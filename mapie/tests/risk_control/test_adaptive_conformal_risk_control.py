import numpy as np
import pytest
import torch

from mapie.risk_control import adaptive_conformal_risk_control as acrc_module
from mapie.risk_control.adaptive_conformal_risk_control import (
    DEVICE,
    AutoAdaptiveConformalRiskControl,
    CustomLoss,
    LogisticHead,
    _import_torch,
    _train_model,
)


def _make_data(n: int = 6, size: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, size, size)).astype(np.float32)
    y = (X > 0.5).astype(np.float32)
    return X, y


def _feature_map(X):
    X = np.asarray(X)
    # Flatten each image into an embedding. A single image (size, size) is
    # mapped to a (1, n_features) row, a batch (n, size, size) to (n, n_features)
    return X.reshape(1, -1) if X.ndim == 2 else X.reshape(X.shape[0], -1)


def _predict_function(X):
    return np.asarray(X)


def test_init_stores_parameters():
    crc = AutoAdaptiveConformalRiskControl(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk=None,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    assert crc.predict_function is _predict_function
    assert crc.feature_map is _feature_map
    assert crc.confidence_level == 0.9
    assert crc.risk is None
    assert np.isclose(crc._alpha, 0.1)
    assert crc.base_model is None
    assert crc.learning_rate == 1e-3
    assert crc.weight_decay == 1e-4


def test_conformalize_initializes_default_base_model():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = AutoAdaptiveConformalRiskControl(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk=None,
    )
    crc.conformalize(X, y, n_epochs=2, batch_size=3)
    assert crc.X_conformalize_embedded.shape == (6, 9)
    np.testing.assert_array_equal(crc.y_conformalize, y)
    assert crc.y_conformalize_pred_proba.shape == (6, 3, 3)
    assert isinstance(crc.base_model, LogisticHead)


def test_conformalize_uses_provided_base_model():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = AutoAdaptiveConformalRiskControl(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk=None,
        base_model=LogisticHead(9),
    )
    crc.conformalize(X, y, n_epochs=1, batch_size=6)
    assert isinstance(crc.base_model, LogisticHead)


def test_predict_returns_binary_masks():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = AutoAdaptiveConformalRiskControl(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk=None,
    )
    crc.conformalize(X, y, n_epochs=2, batch_size=3)
    y_pred = crc.predict(X[:2], n_epochs=1, batch_size=3)
    assert y_pred.shape == (2, 3, 3)
    assert np.all(np.isin(y_pred, [0, 1]))


def test_train_model_returns_eval_module():
    torch.manual_seed(0)
    X, y = _make_data()
    embeddings = _feature_map(X)
    masks_pred = _predict_function(X)
    trained = _train_model(
        LogisticHead(embeddings.shape[1]),
        y,
        masks_pred,
        embeddings,
        lr=1e-2,
        weight_decay=0.0,
        n_epochs=2,
        batch_size=3,
        alpha=0.1,
        x_n_plus_1=embeddings[1:2],
    )
    assert not trained.training
    threshold = trained(torch.tensor(embeddings[1:2].astype(np.float32)).to(DEVICE))
    assert 0.0 <= float(threshold.item()) <= 1.0


def test_train_model_no_improvement_branch():
    # With lr=0 the parameters never update, so the loss is identical across
    # batches: from the second batch on, ``loss.item() < best_loss`` is False.
    torch.manual_seed(0)
    X, y = _make_data()
    embeddings = _feature_map(X)
    masks_pred = _predict_function(X)
    trained = _train_model(
        LogisticHead(embeddings.shape[1]),
        y,
        masks_pred,
        embeddings,
        lr=0.0,
        weight_decay=0.0,
        n_epochs=2,
        batch_size=6,
        alpha=0.1,
        x_n_plus_1=embeddings[0:1],
    )
    assert not trained.training


def test_logistic_head_output_range():
    torch.manual_seed(0)
    head = LogisticHead(4)
    out = head(torch.rand(5, 4))
    assert tuple(out.shape) == (5, 1)
    assert bool(torch.all(out >= 0))
    assert bool(torch.all(out <= 1))


def test_custom_loss_forward_and_integral():
    torch.manual_seed(0)
    loss_fn = CustomLoss(alpha=0.1, n=4)
    masks = torch.rand(4, 3, 3, device=DEVICE)
    masks_pred = torch.rand(4, 3, 3, device=DEVICE)
    preds_th = torch.rand(4, 1, device=DEVICE)
    th_n_plus_1 = torch.rand(1, 1, device=DEVICE)
    loss = loss_fn(masks, masks_pred, preds_th, th_n_plus_1)
    assert loss.numel() == 1
    integrals = loss_fn._compute_integrals(masks, masks_pred, preds_th)
    assert tuple(integrals.shape) == (4,)


def test_import_torch_raises_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="mapie\\[conditional\\]"):
        _import_torch()


def test_import_torch_returns_module():
    assert _import_torch() is torch


def test_risk_control_lazy_getattr_loads_class():
    import mapie.risk_control as rc

    # Drop any cached value so the lazy loader runs.
    rc.__dict__.pop("AutoAdaptiveConformalRiskControl", None)
    loaded = rc.AutoAdaptiveConformalRiskControl
    assert loaded is AutoAdaptiveConformalRiskControl
    assert loaded is acrc_module.AutoAdaptiveConformalRiskControl


def test_risk_control_getattr_unknown_name_raises():
    import mapie.risk_control as rc

    with pytest.raises(AttributeError, match="has no attribute"):
        rc.DefinitelyNotARealExport  # noqa: B018
