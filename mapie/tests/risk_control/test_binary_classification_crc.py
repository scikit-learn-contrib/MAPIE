import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from mapie.risk_control import BinaryClassificationCRController
from mapie.utils import NotFittedError

# Hand-computable calibration set (n=6, B=1). Positives have probabilities
# 0.4, 0.6, 0.8, 0.9 and negatives 0.1, 0.2.
PROBA = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
Y = np.array([0, 0, 1, 1, 1, 1])
X = PROBA.reshape(-1, 1)


def predict_proba(X):
    p = np.asarray(X)[:, 0]
    return np.column_stack([1 - p, p])


def false_positive_loss(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 1)).astype(float)


def make(**kwargs):
    kwargs.setdefault("predict_function", predict_proba)
    kwargs.setdefault("target_level", 0.3)
    return BinaryClassificationCRController(**kwargs)


def test_false_negative_threshold_matches_hand_computation():
    # corrected risk = (#FN + 1) / 7 at thresholds
    # [-inf, .1, .2, .4, .6, .8, .9, inf] -> 1/7, 1/7, 1/7, 1/7, 2/7, 3/7, 4/7, 5/7
    controller = make(target_level=0.3).calibrate(X, Y)
    np.testing.assert_allclose(
        controller.thresholds_, [-np.inf, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, np.inf]
    )
    np.testing.assert_allclose(
        controller.risk_bound_, np.array([1, 1, 1, 1, 2, 3, 4, 5]) / 7
    )
    assert controller.threshold_ == 0.6
    assert make(target_level=0.2).calibrate(X, Y).threshold_ == 0.4


def test_predict_uses_greater_or_equal_rule_when_increasing():
    controller = make(target_level=0.3).calibrate(X, Y)
    np.testing.assert_array_equal(controller.predict(X), [0, 0, 0, 1, 1, 1])


def test_false_positive_threshold_matches_hand_computation():
    # corrected risk = (#FP + 1) / 7 at the same thresholds
    # -> 3/7, 2/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7
    kwargs = dict(risk_function=false_positive_loss, monotonicity="decreasing")
    controller = make(target_level=0.3, **kwargs).calibrate(X, Y)
    np.testing.assert_allclose(
        controller.risk_bound_, np.array([3, 2, 1, 1, 1, 1, 1, 1]) / 7
    )
    assert controller.threshold_ == 0.1
    assert make(target_level=0.2, **kwargs).calibrate(X, Y).threshold_ == 0.2
    # strict rule when decreasing
    np.testing.assert_array_equal(
        make(target_level=0.2, **kwargs).calibrate(X, Y).predict(X),
        [0, 0, 1, 1, 1, 1],
    )


def test_upper_bound_rescales_the_risk():
    def double_fn(y_true, y_pred):
        return 2 * ((y_true == 1) & (y_pred == 0)).astype(float)

    controller = make(
        risk_function=double_fn, upper_bound=2.0, target_level=0.4
    ).calibrate(X, Y)
    assert controller.threshold_ == make(target_level=0.2).calibrate(X, Y).threshold_


def test_uncontrollable_risk_returns_safest_threshold_with_warning():
    with pytest.warns(UserWarning, match="cannot be controlled"):
        controller = make(target_level=0.1).calibrate(X, Y)  # 0.1 < 1 / 7
    assert controller.threshold_ == -np.inf
    np.testing.assert_array_equal(controller.predict(X), np.ones(6))

    with pytest.warns(UserWarning, match="cannot be controlled"):
        controller = make(
            target_level=0.1,
            risk_function=false_positive_loss,
            monotonicity="decreasing",
        ).calibrate(X, Y)
    assert controller.threshold_ == np.inf
    np.testing.assert_array_equal(controller.predict(X), np.zeros(6))


def test_terminal_condition_warning():
    def constant_loss(y_true, y_pred):
        return np.full(len(y_true), 0.5)

    with pytest.warns(UserWarning, match="cannot be controlled"):
        with pytest.warns(UserWarning, match="terminal threshold"):
            make(risk_function=constant_loss).calibrate(X, Y)


def test_positive_class_with_arbitrary_labels():
    y_str = np.where(Y == 1, "spam", "ham")
    controller = make(positive_class="spam").calibrate(X, y_str)
    assert controller.threshold_ == make().calibrate(X, Y).threshold_
    # a single class in the calibration labels is accepted
    make(positive_class="spam").calibrate(X, np.array(["spam"] * 6))


@pytest.mark.parametrize(
    "kwargs, y, match",
    [
        ({"target_level": 0.0}, Y, "target_level"),
        ({"target_level": 1.0}, Y, "target_level"),
        ({"upper_bound": 0.0}, Y, "upper_bound"),
        ({"upper_bound": np.inf}, Y, "upper_bound"),
        ({"monotonicity": "flat"}, Y, "monotonicity"),
        ({"monotonicity": "decreasing"}, Y, "default false negative"),
        ({}, np.array([0, 1, 2, 0, 1, 2]), "only 0 and 1"),
        ({"positive_class": 1}, np.array([0, 1, 2, 0, 1, 2]), "at most two"),
        ({"positive_class": "c"}, np.array(["a", "b"] * 3), "not among"),
        ({}, Y[:-1], "same number"),
    ],
)
def test_invalid_inputs(kwargs, y, match):
    with pytest.raises(ValueError, match=match):
        make(**kwargs).calibrate(X, y)


def test_predict_function_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        make(predict_function=LogisticRegression()).calibrate(X, Y)


@pytest.mark.parametrize(
    "proba, match",
    [
        (np.ones(6), "shape"),
        (np.ones((6, 3)) / 3, "shape"),
        (np.full((6, 2), 1.5), r"\[0, 1\]"),
        (np.full((6, 2), np.nan), r"\[0, 1\]"),
    ],
)
def test_invalid_probabilities(proba, match):
    with pytest.raises(ValueError, match=match):
        make(predict_function=lambda X: proba).calibrate(X, Y)


@pytest.mark.parametrize(
    "loss, match",
    [
        (lambda y_true, y_pred: np.mean(y_true), "one loss per sample"),
        (lambda y_true, y_pred: np.full(len(y_true), 2.0), r"in \[0, 1.0\]"),
        (lambda y_true, y_pred: np.full(len(y_true), np.nan), r"in \[0, 1.0\]"),
        (lambda y_true, y_pred: (y_pred == 1).astype(float), "not increasing"),
    ],
)
def test_invalid_risk_functions(loss, match):
    with pytest.raises(ValueError, match=match):
        make(risk_function=loss).calibrate(X, Y)


def test_not_decreasing_loss_is_rejected():
    with pytest.raises(ValueError, match="not decreasing"):
        make(
            risk_function=lambda y_true, y_pred: (y_pred == 0).astype(float),
            monotonicity="decreasing",
        ).calibrate(X, Y)


def test_predict_before_calibrate_raises():
    controller = make()
    assert not controller.is_fitted
    with pytest.raises(NotFittedError):
        controller.predict(X)


def test_clone_and_params():
    controller = make(target_level=0.2, positive_class=1)
    cloned = clone(controller)
    assert cloned.get_params() == controller.get_params()
    assert not cloned.is_fitted
    assert controller.calibrate(X, Y).is_fitted


def test_works_with_scikit_learn_classifier():
    rng = np.random.RandomState(0)
    X_all = rng.normal(size=(600, 2))
    y_all = (X_all[:, 0] + rng.normal(scale=0.5, size=600) > 0).astype(int)
    clf = LogisticRegression().fit(X_all[:200], y_all[:200])
    controller = make(predict_function=clf.predict_proba, target_level=0.1).calibrate(
        X_all[200:400], y_all[200:400]
    )
    y_pred = controller.predict(X_all[400:])
    assert set(np.unique(y_pred)) <= {0, 1}


@pytest.mark.parametrize("monotonicity", ["increasing", "decreasing"])
def test_expected_risk_is_controlled_on_synthetic_data(monotonicity):
    """
    Monte Carlo check of E[L_{n+1}(lambda_hat)] <= alpha, where the expectation
    is over calibration sets and a new example. The classifier outputs the true
    conditional probability, but it is deliberately miscalibrated (not needed
    for the guarantee) to show that the guarantee is distribution-free.
    """
    alpha, n, n_test, n_trials = 0.1, 100, 2000, 300
    rng = np.random.RandomState(1)

    def draw(size):
        p = rng.uniform(size=size)
        return p, (rng.uniform(size=size) < p).astype(int)

    def predict_function(X):
        p = np.asarray(X)[:, 0] ** 2
        return np.column_stack([1 - p, p])

    if monotonicity == "increasing":
        kwargs = {}
        loss_fn = lambda y, y_pred: ((y == 1) & (y_pred == 0)).astype(float)  # noqa
    else:
        kwargs = dict(risk_function=false_positive_loss, monotonicity="decreasing")
        loss_fn = false_positive_loss

    test_risks = []
    for _ in range(n_trials):
        p_cal, y_cal = draw(n)
        controller = BinaryClassificationCRController(
            predict_function=predict_function, target_level=alpha, **kwargs
        ).calibrate(p_cal.reshape(-1, 1), y_cal)
        p_test, y_test = draw(n_test)
        y_pred = controller.predict(p_test.reshape(-1, 1))
        test_risks.append(loss_fn(y_test, y_pred).mean())

    mean_risk = np.mean(test_risks)
    standard_error = np.std(test_risks) / np.sqrt(n_trials)
    assert mean_risk <= alpha + 3 * standard_error
    # the procedure should not be trivially conservative
    assert mean_risk >= alpha - 0.05
