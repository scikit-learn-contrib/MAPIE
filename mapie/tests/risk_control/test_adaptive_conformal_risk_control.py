import numpy as np
import pytest
import torch

from mapie.risk_control import RiskLoss, miscoverage_loss, recall_loss
from mapie.risk_control import adaptive_conformal_risk_control as acrc_module
from mapie.risk_control.adaptive_conformal_risk_control import (
    ConditionalExpectedRiskController,
    _AACRCLoss,
    _LinearHead,
    _LogisticHead,
    _evaluate_aacrc_objective,
    _import_torch,
    _train_model,
)

# These unit tests verify deterministic algorithm semantics, not accelerator
# behavior. Use CPU on every platform to avoid backend-specific MPS results.
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _use_cpu_device(monkeypatch):
    monkeypatch.setattr(acrc_module, "DEVICE", DEVICE)


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
    crc = ConditionalExpectedRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        target_level=0.1,
        risk="recall",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    assert crc.predict_function is _predict_function
    assert crc.feature_map is _feature_map
    assert crc.target_level == 0.1
    assert crc.risk == "recall"
    assert crc._risk is recall_loss
    assert crc.predict_param_range is None
    assert crc._risk.higher_is_better
    assert crc._risk.monotonicity == "increasing"
    assert crc._risk.objective_sign == 1.0
    assert crc.base_model is None
    assert crc.learning_rate == 1e-3
    assert crc.weight_decay == 1e-4


def test_conformalize_initializes_default_base_model():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = ConditionalExpectedRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        target_level=0.1,
        risk="recall",
    )
    crc.conformalize(X, y, n_epochs=2, batch_size=3)
    assert crc.X_conformalize_embedded.shape == (6, 9)
    np.testing.assert_array_equal(crc.y_conformalize, y)
    np.testing.assert_array_equal(crc.y_conformalize_pred, X)
    assert isinstance(crc.base_model, _LinearHead)


@pytest.mark.parametrize(
    "predict_param_range, expected_predict_param",
    [
        (None, 2.0),
        ((0.0, 1.0), 1.0),
        ((0.5, 1.0), 1.0),
    ],
)
def test_predict_param_range_is_optional(
    predict_param_range,
    expected_predict_param,
):
    X, y = _make_data()
    head = _LinearHead(9)
    with torch.no_grad():
        head.fc.weight.zero_()
        head.fc.bias.fill_(2.0)
    crc = ConditionalExpectedRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        target_level=0.1,
        risk="recall",
        predict_param_range=predict_param_range,
        base_model=head,
    )

    crc.conformalize(X, y, n_epochs=0, batch_size=6)
    crc.predict(X[:1], n_epochs=0, batch_size=6)

    np.testing.assert_allclose(
        crc.best_predict_params_,
        expected_predict_param,
    )


def test_conformalize_uses_provided_base_model():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    head = _LogisticHead(9)
    with torch.no_grad():
        head.fc.weight.fill_(0.25)
        head.fc.bias.fill_(-0.5)
    crc = ConditionalExpectedRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        target_level=0.1,
        risk="recall",
        base_model=head,
    )
    crc.conformalize(X, y, n_epochs=0, batch_size=6)
    assert isinstance(crc.base_model, _LogisticHead)
    np.testing.assert_allclose(
        crc.base_model.fc.weight.detach().cpu().numpy(),
        0.25,
    )
    np.testing.assert_allclose(
        crc.base_model.fc.bias.detach().cpu().numpy(),
        -0.5,
    )


@pytest.mark.parametrize(
    "target_level",
    [True, 1, 0.0, 1.0, -0.1, 1.1, np.nan, np.inf],
)
def test_init_rejects_invalid_target_level(target_level):
    with pytest.raises(ValueError, match="`target_level` must be a float"):
        ConditionalExpectedRiskController(
            predict_function=_predict_function,
            feature_map=_feature_map,
            target_level=target_level,
            risk="recall",
        )


@pytest.mark.parametrize(
    "risk_value, message",
    [
        (-0.1, r"lie in \[0, 1\]"),
        (1.1, r"lie in \[0, 1\]"),
        (np.nan, "finite"),
        (np.inf, "finite"),
    ],
)
def test_risk_loss_rejects_values_outside_unit_interval(risk_value, message):
    def invalid_risk(y_true, y_pred, predict_param):
        return predict_param.squeeze(-1) * 0 + risk_value

    risk = RiskLoss(
        invalid_risk,
        higher_is_better=False,
        monotonicity="increasing",
    )

    with pytest.raises(ValueError, match=message):
        risk(None, None, torch.tensor([[0.5]]))


def test_predict_returns_binary_masks():
    np.random.seed(0)
    torch.manual_seed(0)
    X, y = _make_data()
    crc = ConditionalExpectedRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        target_level=0.1,
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
    )
    assert not trained.training
    predict_param = trained(torch.tensor(embeddings[1:2].astype(np.float32)).to(DEVICE))
    assert 0.0 <= float(predict_param.item()) <= 1.0


def test_train_model_with_zero_learning_rate_returns_eval_module():
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
    )
    assert not trained.training


def test_evaluate_aacrc_objective_includes_full_objective_and_regularization():
    def constant_loss(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.4,
            device=predict_param.device,
        )

    risk = RiskLoss(
        constant_loss,
        higher_is_better=False,
        monotonicity="increasing",
    )
    embeddings = np.ones((2, 1), dtype=np.float32)
    model = _LinearHead(1).to(DEVICE)
    torch.nn.init.zeros_(model.fc.weight)
    torch.nn.init.constant_(model.fc.bias, 2.0)

    objective = _evaluate_aacrc_objective(
        model,
        torch.zeros(2, device=DEVICE),
        torch.zeros(2, device=DEVICE),
        torch.tensor(embeddings, device=DEVICE),
        torch.tensor(embeddings[0:1], device=DEVICE),
        alpha=0.1,
        risk=risk,
        weight_decay=0.5,
    )

    # Integrated terms: 2 * (0.4 - 0.1) * 2 = 1.2.
    # Worst-case term: (1 - 0.1) * 2 = 1.8. Dividing their sum by
    # n + 1 gives 1.0. The L2 penalty is 0.5 / 2 * 2**2 = 1.0.
    assert objective == pytest.approx(2.0)


def test_evaluate_aacrc_objective_restores_training_mode():
    X, y = _make_data()
    embeddings = _feature_map(X)
    model = _LogisticHead(embeddings.shape[1]).to(DEVICE)
    model.train()

    _evaluate_aacrc_objective(
        model,
        torch.tensor(y, device=DEVICE),
        torch.tensor(X, device=DEVICE),
        torch.tensor(embeddings, device=DEVICE),
        torch.tensor(embeddings[0:1], device=DEVICE),
        alpha=0.1,
        risk=recall_loss,
        weight_decay=0.0,
    )

    assert model.training


def test_train_model_returns_best_epoch(monkeypatch):
    def constant_loss(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.5,
            device=predict_param.device,
        )

    risk = RiskLoss(
        constant_loss,
        higher_is_better=False,
        monotonicity="increasing",
    )
    embeddings = np.ones((2, 1), dtype=np.float32)
    model = _LinearHead(1)
    torch.nn.init.zeros_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)
    evaluated_biases = []
    objective_values = iter([2.0, 0.0, 1.0])

    def evaluate_objective(model, *args, **kwargs):
        evaluated_biases.append(model.fc.bias.detach().clone())
        return next(objective_values)

    monkeypatch.setattr(
        acrc_module,
        "_evaluate_aacrc_objective",
        evaluate_objective,
    )

    trained = _train_model(
        model,
        y_true=np.zeros(2, dtype=np.float32),
        y_pred=np.zeros(2, dtype=np.float32),
        embeddings=embeddings,
        lr=0.1,
        weight_decay=0.0,
        n_epochs=2,
        batch_size=2,
        alpha=0.1,
        x_n_plus_1=embeddings[0:1],
        risk=risk,
    )

    assert trained is model
    assert len(evaluated_biases) == 3
    assert torch.equal(trained.fc.bias, evaluated_biases[1])
    assert not torch.equal(trained.fc.bias, evaluated_biases[2])


@pytest.mark.parametrize(
    "batch_size, error, message",
    [
        (0, ValueError, "must be strictly positive"),
        (-1, ValueError, "must be strictly positive"),
        (1.5, TypeError, "must be an integer"),
        (True, TypeError, "must be an integer"),
    ],
)
def test_train_model_rejects_invalid_batch_size(batch_size, error, message):
    X, y = _make_data()
    embeddings = _feature_map(X)
    with pytest.raises(error, match=message):
        _train_model(
            _LogisticHead(embeddings.shape[1]),
            y,
            X,
            embeddings,
            lr=1e-2,
            weight_decay=0.0,
            n_epochs=1,
            batch_size=batch_size,
            alpha=0.1,
            x_n_plus_1=embeddings[0:1],
            risk=recall_loss,
        )


def test_logistic_head_output_range():
    torch.manual_seed(0)
    head = _LogisticHead(4, predict_param_range=(0.5, 2.0))
    out = head(torch.rand(5, 4))
    assert tuple(out.shape) == (5, 1)
    assert bool(torch.all(out >= 0.5))
    assert bool(torch.all(out <= 2.0))


def test_logistic_head_zero_initialization_outputs_range_midpoint():
    head = _LogisticHead(4, predict_param_range=(0.5, 2.0))
    torch.nn.init.zeros_(head.fc.weight)
    torch.nn.init.zeros_(head.fc.bias)
    out = head(torch.rand(5, 4))
    np.testing.assert_allclose(out.detach().numpy(), 1.25)


def test_aacrc_gradient_surrogate_is_zero_and_differentiable():
    torch.manual_seed(0)
    loss_fn = _AACRCLoss(alpha=0.1, n=4)
    y_true = torch.ones(4, 3, 3, device=DEVICE)
    y_pred = torch.rand(4, 3, 3, device=DEVICE)
    predict_params = torch.rand(4, 1, device=DEVICE, requires_grad=True)
    predict_param_n_plus_1 = torch.rand(
        1,
        1,
        device=DEVICE,
        requires_grad=True,
    )
    loss = loss_fn(y_true, y_pred, predict_params, predict_param_n_plus_1)
    assert torch.isclose(loss, torch.tensor(0.0, device=DEVICE))
    loss.backward()
    assert predict_params.grad is not None
    assert bool(torch.isfinite(predict_params.grad).all())
    assert predict_param_n_plus_1.grad is not None
    assert bool(torch.isfinite(predict_param_n_plus_1.grad).all())


def test_aacrc_gradient_surrogate_evaluates_endpoint_risk_once():
    parameter_shapes = []

    def endpoint_loss(y_true, y_pred, predict_param):
        parameter_shapes.append(tuple(predict_param.shape))
        return torch.full(
            (len(predict_param),),
            0.25,
            dtype=y_pred.dtype,
            device=y_pred.device,
        )

    loss_fn = _AACRCLoss(
        alpha=0.1,
        n=4,
        risk=RiskLoss(
            endpoint_loss,
            higher_is_better=False,
            monotonicity="increasing",
        ),
    )
    loss_fn(
        torch.ones(4, 3, 3, device=DEVICE),
        torch.ones(4, 3, 3, device=DEVICE),
        torch.ones(4, 1, device=DEVICE, requires_grad=True),
        torch.ones(1, 1, device=DEVICE, requires_grad=True),
    )

    assert parameter_shapes == [(4, 1, 1)]


def test_aacrc_gradient_surrogate_uses_endpoint_identity():
    loss_fn = _AACRCLoss(alpha=0.1, n=1, risk=miscoverage_loss)
    y_true = torch.tensor([[1.0]], device=DEVICE)
    y_pred = torch.tensor([[0.0]], device=DEVICE)
    predict_params = torch.tensor(
        [[2.0]],
        device=DEVICE,
        requires_grad=True,
    )
    predict_param_n_plus_1 = torch.tensor(
        [[2.0]],
        device=DEVICE,
        requires_grad=True,
    )

    loss = loss_fn(
        y_true,
        y_pred,
        predict_params,
        predict_param_n_plus_1,
    )

    assert torch.isclose(loss, torch.tensor(0.0, device=DEVICE))
    loss.backward()
    assert torch.isclose(
        predict_params.grad.squeeze(),
        torch.tensor(0.05, device=DEVICE),
    )
    assert torch.isclose(
        predict_param_n_plus_1.grad.squeeze(),
        torch.tensor(-0.45, device=DEVICE),
    )


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

    metric_loss = _AACRCLoss(
        alpha=0.1,
        n=1,
        risk=RiskLoss(
            performance_metric,
            higher_is_better=True,
            monotonicity="increasing",
        ),
    )
    direct_loss = _AACRCLoss(
        alpha=0.1,
        n=1,
        risk=RiskLoss(
            loss_function,
            higher_is_better=False,
            monotonicity="increasing",
        ),
    )

    def get_gradients(loss_fn):
        predict_params = torch.tensor(
            [[0.5]],
            device=DEVICE,
            requires_grad=True,
        )
        predict_param_n_plus_1 = torch.tensor(
            [[0.5]],
            device=DEVICE,
            requires_grad=True,
        )
        loss = loss_fn(
            torch.ones(1, 1, device=DEVICE),
            torch.ones(1, 1, device=DEVICE),
            predict_params,
            predict_param_n_plus_1,
        )
        return torch.autograd.grad(
            loss,
            (predict_params, predict_param_n_plus_1),
        )

    metric_gradients = get_gradients(metric_loss)
    direct_gradients = get_gradients(direct_loss)
    assert all(
        torch.allclose(metric_gradient, direct_gradient)
        for metric_gradient, direct_gradient in zip(
            metric_gradients,
            direct_gradients,
        )
    )


def test_aacrc_gradient_surrogate_calibration_and_worst_case_gradients():
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
        risk=RiskLoss(
            constant_loss,
            higher_is_better=False,
            monotonicity="increasing",
        ),
    )(
        torch.ones(1, 1, device=DEVICE),
        torch.ones(1, 1, device=DEVICE),
        predict_params,
        predict_param_n_plus_1,
    )

    assert torch.isclose(loss, torch.tensor(0.0, device=DEVICE))
    loss.backward()
    assert torch.isclose(
        predict_params.grad.squeeze(),
        torch.tensor(0.075, device=DEVICE),
    )
    assert torch.isclose(
        predict_param_n_plus_1.grad.squeeze(),
        torch.tensor(0.45, device=DEVICE),
    )


def test_aacrc_loss_scales_minibatch_to_full_objective():
    def constant_loss(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.5,
            dtype=y_pred.dtype,
            device=y_pred.device,
        )

    predict_params = torch.ones(
        2,
        1,
        device=DEVICE,
        requires_grad=True,
    )
    predict_param_n_plus_1 = torch.ones(
        1,
        1,
        device=DEVICE,
        requires_grad=True,
    )
    loss = _AACRCLoss(
        alpha=0.1,
        n=6,
        risk=RiskLoss(
            constant_loss,
            higher_is_better=False,
            monotonicity="increasing",
        ),
    )(
        torch.ones(2, 1, device=DEVICE),
        torch.ones(2, 1, device=DEVICE),
        predict_params,
        predict_param_n_plus_1,
    )

    assert torch.isclose(loss, torch.tensor(0.0, device=DEVICE))
    loss.backward()
    expected_calibration_gradient = torch.tensor(3 * 0.4 / 7, device=DEVICE)
    assert bool(
        torch.allclose(
            predict_params.grad,
            torch.full_like(predict_params, expected_calibration_gradient),
        )
    )
    assert torch.isclose(
        predict_param_n_plus_1.grad.squeeze(),
        torch.tensor(0.9 / 7, device=DEVICE),
    )


def test_aacrc_loss_negates_objective_for_decreasing_risk():
    def constant_loss(y_true, y_pred, predict_param):
        return torch.full(
            (len(predict_param),),
            0.25,
            dtype=y_pred.dtype,
            device=y_pred.device,
        )

    def get_gradients(monotonicity):
        predict_params = torch.tensor(
            [[0.5]],
            device=DEVICE,
            requires_grad=True,
        )
        predict_param_n_plus_1 = torch.tensor(
            [[0.5]],
            device=DEVICE,
            requires_grad=True,
        )
        loss = _AACRCLoss(
            alpha=0.1,
            n=1,
            risk=RiskLoss(
                constant_loss,
                higher_is_better=False,
                monotonicity=monotonicity,
            ),
        )(
            torch.ones(1, 1, device=DEVICE),
            torch.ones(1, 1, device=DEVICE),
            predict_params,
            predict_param_n_plus_1,
        )
        return torch.autograd.grad(
            loss,
            (predict_params, predict_param_n_plus_1),
        )

    increasing_gradients = get_gradients("increasing")
    decreasing_gradients = get_gradients("decreasing")
    assert all(
        torch.allclose(decreasing_gradient, -increasing_gradient)
        for increasing_gradient, decreasing_gradient in zip(
            increasing_gradients,
            decreasing_gradients,
        )
    )


def test_risk_loss_rejects_invalid_monotonicity():
    with pytest.raises(ValueError, match="`monotonicity` must be either"):
        RiskLoss(
            lambda y_true, y_pred, param: param,
            higher_is_better=False,
            monotonicity="invalid",
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
    assert recall_loss.monotonicity == "increasing"
    assert recall_loss.objective_sign == 1.0


def test_predefined_miscoverage_loss_is_differentiable():
    y_true = torch.tensor([0.0, 2.0], device=DEVICE)
    y_pred = torch.tensor([0.0, 0.0], device=DEVICE)
    widths = torch.tensor([0.5, 1.0], device=DEVICE, requires_grad=True)
    losses = miscoverage_loss(y_true, y_pred, widths)
    np.testing.assert_allclose(losses.detach().cpu(), [0.0, 1.0], atol=1e-6)
    losses.sum().backward()
    assert widths.grad is not None
    assert not miscoverage_loss.higher_is_better
    assert miscoverage_loss.monotonicity == "decreasing"
    assert miscoverage_loss.objective_sign == -1.0


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
        "target_level": 0.1,
        "risk": "recall",
    }
    params.update(overrides)
    with pytest.raises(error, match=message):
        crc = ConditionalExpectedRiskController(**params)
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

    crc = ConditionalExpectedRiskController(
        predict_function=_predict_function,
        feature_map=_feature_map,
        target_level=0.1,
        risk=RiskLoss(
            constant_loss,
            higher_is_better=False,
            monotonicity="increasing",
        ),
    )
    crc.conformalize(X, y, n_epochs=1)
    assert isinstance(crc.risk, RiskLoss)
    assert crc._risk is crc.risk
    assert not crc._risk.higher_is_better
    assert crc._risk.monotonicity == "increasing"


def test_custom_prediction_function_for_regression_intervals():
    X = np.arange(6, dtype=np.float32).reshape(-1, 1)
    y = X.ravel() + np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2])

    def interval_function(X, widths=None):
        y_pred = np.asarray(X).ravel()
        if widths is None:
            return y_pred
        return np.column_stack([y_pred - widths, y_pred + widths])

    crc = ConditionalExpectedRiskController(
        predict_function=interval_function,
        feature_map=lambda X: np.column_stack([np.ones(len(X)), np.asarray(X)]),
        target_level=0.1,
        risk="miscoverage",
        predict_param_range=(0.5, 2.0),
    )
    crc.conformalize(X, y, n_epochs=1, batch_size=6)
    intervals = crc.predict(X[:2], n_epochs=0, batch_size=6)
    assert intervals.shape == (2, 2)
    assert crc._risk.objective_sign == -1.0
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
    rc.__dict__.pop("ConditionalExpectedRiskController", None)
    loaded = rc.ConditionalExpectedRiskController
    assert loaded is ConditionalExpectedRiskController
    assert loaded is acrc_module.ConditionalExpectedRiskController


def test_risk_control_getattr_unknown_name_raises():
    import mapie.risk_control as rc

    with pytest.raises(AttributeError, match="has no attribute"):
        rc.DefinitelyNotARealExport  # noqa: B018
