import numpy as np
import pytest
import torch

from mapie.risk_control import RiskLoss, miscoverage_loss, recall_loss
from mapie.risk_control import adaptive_conformal_risk_control as acrc_module
from mapie.risk_control.adaptive_conformal_risk_control import (
    DEVICE,
    ConditionalRiskController,
    _AACRCLoss,
    _LinearHead,
    _LogisticHead,
    _import_torch,
    _infer_objective_sign,
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


def _predict_function(X, predict_params=None):
    X = np.asarray(X)
    if predict_params is None:
        return X
    shape = (len(predict_params),) + (1,) * (X.ndim - 1)
    return X >= np.asarray(predict_params).reshape(shape)


def test_init_stores_parameters():
    crc = ConditionalRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk="recall",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    assert crc.predict_function is _predict_function
    assert crc.feature_map is _feature_map
    assert crc.confidence_level == 0.9
    assert crc.risk == "recall"
    assert crc._risk is recall_loss
    assert crc.predict_param_range == (0.0, 1.0)
    assert crc._risk.higher_is_better
    assert np.isclose(crc._alpha, 0.1)
    assert crc.base_model is None
    assert crc.learning_rate == 1e-3
    assert crc.weight_decay == 1e-4


def test_conformalize_initializes_default_base_model():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = ConditionalRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk="recall",
    )
    crc.conformalize(X, y, n_epochs=2, batch_size=3)
    assert crc.X_conformalize_embedded.shape == (6, 9)
    np.testing.assert_array_equal(crc.y_conformalize, y)
    np.testing.assert_array_equal(crc.y_conformalize_pred, X)
    assert isinstance(crc.base_model, _LinearHead)
    assert crc._objective_sign == 1.0


def test_conformalize_uses_provided_base_model():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = ConditionalRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk="recall",
        base_model=_LogisticHead(9),
    )
    crc.conformalize(X, y, n_epochs=1, batch_size=6)
    assert isinstance(crc.base_model, _LogisticHead)


def test_predict_returns_binary_masks():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = ConditionalRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk="recall",
    )
    crc.conformalize(X, y, n_epochs=2, batch_size=3)
    y_pred = crc.predict(X[:2], n_epochs=1, batch_size=3)
    assert y_pred.shape == (2, 3, 3)
    assert np.all(np.isin(y_pred, [0, 1]))


def test_train_model_returns_eval_module():
    torch.manual_seed(0)
    X, y = _make_data()
    embeddings = _feature_map(X)
    trained = _train_model(
        _LogisticHead(embeddings.shape[1]),
        y,
        X,
        embeddings,
        lr=1e-2,
        weight_decay=0.0,
        n_epochs=2,
        batch_size=3,
        alpha=0.1,
        x_n_plus_1=embeddings[1:2],
        risk=recall_loss,
        integration_start=0.0,
    )
    assert not trained.training
    predict_param = trained(torch.tensor(embeddings[1:2].astype(np.float32)).to(DEVICE))
    assert 0.0 <= float(predict_param.item()) <= 1.0


def test_train_model_no_improvement_branch():
    # With lr=0 the parameters never update, so the loss is identical across
    # batches: from the second batch on, ``loss.item() < best_loss`` is False.
    torch.manual_seed(0)
    X, y = _make_data()
    embeddings = _feature_map(X)
    trained = _train_model(
        _LogisticHead(embeddings.shape[1]),
        y,
        X,
        embeddings,
        lr=0.0,
        weight_decay=0.0,
        n_epochs=2,
        batch_size=6,
        alpha=0.1,
        x_n_plus_1=embeddings[0:1],
        risk=recall_loss,
        integration_start=0.0,
    )
    assert not trained.training


def test_logistic_head_output_range():
    torch.manual_seed(0)
    head = _LogisticHead(4)
    out = head(torch.rand(5, 4))
    assert tuple(out.shape) == (5, 1)
    assert bool(torch.all(out >= 0))
    assert bool(torch.all(out <= 1))


def test_custom_loss_forward_and_integral():
    torch.manual_seed(0)
    loss_fn = _AACRCLoss(alpha=0.1, n=4)
    y_true = torch.ones(4, 3, 3, device=DEVICE)
    y_pred = torch.rand(4, 3, 3, device=DEVICE)
    predict_params = torch.rand(4, 1, device=DEVICE, requires_grad=True)
    predict_param_n_plus_1 = torch.rand(1, 1, device=DEVICE)
    loss = loss_fn(y_true, y_pred, predict_params, predict_param_n_plus_1)
    assert loss.numel() == 1
    loss.backward()
    assert predict_params.grad is not None
    assert bool(torch.isfinite(predict_params.grad).all())
    integrals = loss_fn._compute_integrals(y_true, y_pred, predict_params)
    assert tuple(integrals.shape) == (4,)


def test_aacrc_loss_is_same_for_equivalent_metric_and_loss():
    def performance_metric(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.75,
            dtype=y_pred.dtype,
            device=y_pred.device,
        )

    def loss_function(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.25,
            dtype=y_pred.dtype,
            device=y_pred.device,
        )

    y_true = torch.ones(1, 1, device=DEVICE)
    y_pred = torch.ones(1, 1, device=DEVICE)
    predict_params = torch.tensor([[0.5]], device=DEVICE)
    predict_param_n_plus_1 = torch.tensor([[0.5]], device=DEVICE)
    metric_loss = _AACRCLoss(
        alpha=0.1,
        n=1,
        risk=RiskLoss(performance_metric, higher_is_better=True),
    )
    direct_loss = _AACRCLoss(
        alpha=0.1,
        n=1,
        risk=RiskLoss(loss_function, higher_is_better=False),
    )

    assert torch.allclose(
        metric_loss(
            y_true,
            y_pred,
            predict_params,
            predict_param_n_plus_1,
        ),
        direct_loss(
            y_true,
            y_pred,
            predict_params,
            predict_param_n_plus_1,
        ),
    )


def test_aacrc_loss_with_nonzero_integration_start():
    def constant_loss(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.25,
            dtype=y_pred.dtype,
            device=y_pred.device,
        )

    predict_params = torch.tensor([[3.0]], device=DEVICE, requires_grad=True)
    predict_param_n_plus_1 = torch.tensor(
        [[3.0]],
        device=DEVICE,
        requires_grad=True,
    )
    loss = _AACRCLoss(
        alpha=0.1,
        n=1,
        risk=RiskLoss(constant_loss, higher_is_better=False),
        integration_start=2.0,
    )(
        torch.ones(1, 1, device=DEVICE),
        torch.ones(1, 1, device=DEVICE),
        predict_params,
        predict_param_n_plus_1,
    )

    assert torch.isclose(loss, torch.tensor(0.525, device=DEVICE))
    loss.backward()
    assert torch.isclose(
        predict_params.grad.squeeze(),
        torch.tensor(0.075, device=DEVICE),
    )
    assert torch.isclose(
        predict_param_n_plus_1.grad.squeeze(),
        torch.tensor(0.45, device=DEVICE),
    )


def test_aacrc_loss_negates_objective_for_decreasing_risk():
    y_true = torch.ones(1, 1, device=DEVICE)
    y_pred = torch.ones(1, 1, device=DEVICE)
    predict_params = torch.tensor([[0.5]], device=DEVICE)
    predict_param_n_plus_1 = torch.tensor([[0.5]], device=DEVICE)
    increasing_objective = _AACRCLoss(
        alpha=0.1,
        n=1,
        objective_sign=1.0,
    )(y_true, y_pred, predict_params, predict_param_n_plus_1)
    decreasing_objective = _AACRCLoss(
        alpha=0.1,
        n=1,
        objective_sign=-1.0,
    )(y_true, y_pred, predict_params, predict_param_n_plus_1)

    assert torch.allclose(decreasing_objective, -increasing_objective)


@pytest.mark.parametrize(
    "loss_function, message",
    [
        (
            lambda y_true, y_pred, param: torch.ones(
                len(param) + 1,
                device=param.device,
            ),
            "one value per prediction parameter",
        ),
        (
            lambda y_true, y_pred, param: torch.full(
                (len(param),),
                torch.nan,
                device=param.device,
            ),
            "only finite values",
        ),
        (
            lambda y_true, y_pred, param: torch.full(
                (len(param),),
                1.5,
                device=param.device,
            ),
            r"lie in \[0, 1\]",
        ),
        (
            lambda y_true, y_pred, param: (
                0.5 * torch.sin(2 * torch.pi * param.reshape(len(param), -1)[:, 0])
                + 0.5
            ),
            "must be monotone",
        ),
    ],
)
def test_infer_objective_sign_rejects_invalid_risks(loss_function, message):
    risk = RiskLoss(loss_function, higher_is_better=False)
    with pytest.raises(ValueError, match=message):
        _infer_objective_sign(
            risk,
            np.ones(2),
            np.ones(2),
            (0.0, 1.0),
            n_grid_points=5,
        )


def test_predefined_recall_loss():
    y_true = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0]],
        device=DEVICE,
    )
    y_pred = torch.tensor(
        [[0.9, 0.1], [0.9, 0.1]],
        device=DEVICE,
    )
    predict_params = torch.tensor(
        [[0.5], [0.5]],
        device=DEVICE,
        requires_grad=True,
    )
    losses = recall_loss(y_true, y_pred, predict_params)
    np.testing.assert_allclose(
        losses.detach().cpu(),
        [0.5, 0.0],
        atol=1e-6,
    )
    losses.sum().backward()
    assert predict_params.grad is not None
    assert recall_loss.higher_is_better


def test_predefined_miscoverage_loss_is_differentiable():
    y_true = torch.tensor([0.0, 2.0], device=DEVICE)
    y_pred = torch.tensor([0.0, 0.0], device=DEVICE)
    widths = torch.tensor([0.5, 1.0], device=DEVICE, requires_grad=True)
    losses = miscoverage_loss(y_true, y_pred, widths)
    np.testing.assert_allclose(losses.detach().cpu(), [0.0, 1.0], atol=1e-6)
    losses.sum().backward()
    assert widths.grad is not None
    assert not miscoverage_loss.higher_is_better


@pytest.mark.parametrize(
    "overrides, error, message",
    [
        ({"predict_function": None}, TypeError, "`predict_function` must be callable"),
        ({"risk": None}, TypeError, "`risk` must be a string or a RiskLoss"),
        (
            {"risk": "unknown"},
            ValueError,
            "When `risk` is provided as a string",
        ),
        (
            {"predict_param_range": (0.0,)},
            ValueError,
            "`predict_param_range` must contain two values",
        ),
        (
            {"predict_param_range": (1.0, 0.0)},
            ValueError,
            "`predict_param_range` values must be finite and increasing",
        ),
    ],
)
def test_conformalize_input_validation(overrides, error, message):
    X, y = _make_data()
    params = {
        "predict_function": _predict_function,
        "feature_map": _feature_map,
        "confidence_level": 0.9,
        "risk": "recall",
    }
    params.update(overrides)
    with pytest.raises(error, match=message):
        crc = ConditionalRiskController(**params)
        crc.conformalize(X, y, n_epochs=0)


def test_custom_risk_loss():
    X, y = _make_data()

    def constant_loss(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.5,
            dtype=y_pred.dtype,
            device=y_pred.device,
        )

    crc = ConditionalRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        confidence_level=0.9,
        risk=RiskLoss(constant_loss, higher_is_better=False),
    )
    crc.conformalize(X, y, n_epochs=1)
    assert isinstance(crc.risk, RiskLoss)
    assert crc._risk is crc.risk
    assert not crc._risk.higher_is_better


def test_custom_prediction_function_for_regression_intervals():
    X = np.arange(6, dtype=np.float32).reshape(-1, 1)
    y = X.ravel() + np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2])

    def interval_function(X, widths=None):
        y_pred = np.asarray(X).ravel()
        if widths is None:
            return y_pred
        return np.column_stack([y_pred - widths, y_pred + widths])

    crc = ConditionalRiskController(
        predict_function=interval_function,
        feature_map=lambda X: np.column_stack([np.ones(len(X)), np.asarray(X)]),
        confidence_level=0.9,
        risk="miscoverage",
        predict_param_range=(0.5, 2.0),
    )
    crc.conformalize(X, y, n_epochs=1, batch_size=6)
    intervals = crc.predict(X[:2], n_epochs=0, batch_size=6)
    assert intervals.shape == (2, 2)
    assert crc._objective_sign == -1.0
    assert np.all((crc.best_predict_params_ >= 0.5) & (crc.best_predict_params_ <= 2))


def test_linear_head_output():
    head = _LinearHead(2)
    with torch.no_grad():
        head.fc.weight.fill_(1)
        head.fc.bias.fill_(2)
    output = head(torch.zeros(3, 2))
    np.testing.assert_array_equal(output.detach().numpy(), np.full((3, 1), 2))


def test_import_torch_raises_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    assert fake_import("numpy") is np
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="mapie\\[conditional\\]"):
        _import_torch()


def test_import_torch_returns_module():
    assert _import_torch() is torch


def test_risk_control_lazy_getattr_loads_class():
    import mapie.risk_control as rc

    # Drop any cached value so the lazy loader runs.
    rc.__dict__.pop("ConditionalRiskController", None)
    loaded = rc.ConditionalRiskController
    assert loaded is ConditionalRiskController
    assert loaded is acrc_module.ConditionalRiskController


def test_risk_control_getattr_unknown_name_raises():
    import mapie.risk_control as rc

    with pytest.raises(AttributeError, match="has no attribute"):
        rc.DefinitelyNotARealExport  # noqa: B018
