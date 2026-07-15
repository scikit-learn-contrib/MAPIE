"""Tests for the conditional split conformal regressor and classifier."""

import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from mapie.conditional_conformal_prediction import (
    ConditionalSplitConformalClassifier,
    ConditionalSplitConformalRegressor,
    _import_cvxpy,
    _solve_dual,
    binary_search,
    finish_dual_setup,
    setup_cvx_problem,
)
from mapie.conformity_scores import AbsoluteConformityScore


def _make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 1))
    y = 2 * X[:, 0] + rng.normal(0, 0.5 * (1 + X[:, 0] ** 2), size=n)
    n_half = n // 2
    return X[:n_half], y[:n_half], X[n_half:], y[n_half:]


def _phi(x):
    """Finite basis: intercept + linear term."""
    return np.column_stack([np.ones(len(x)), x[:, 0]])


def _fitted_regressor(confidence_level=0.9, conformity_score="absolute", **kwargs):
    X_conf, y_conf, _, _ = _make_data()
    estimator = LinearRegression().fit(X_conf, y_conf)
    regressor = ConditionalSplitConformalRegressor(
        feature_map=_phi,
        estimator=estimator,
        confidence_level=confidence_level,
        conformity_score=conformity_score,
        prefit=True,
        **kwargs,
    )
    return regressor.conformalize(X_conf, y_conf)


def test_predict_interval_shapes_and_order():
    regressor = _fitted_regressor(confidence_level=[0.8, 0.9])
    _, _, X_test, _ = _make_data()
    points, intervals = regressor.predict_interval(X_test)
    assert points.shape == (len(X_test),)
    assert intervals.shape == (len(X_test), 2, 2)
    assert np.all(intervals[:, 0, :] <= intervals[:, 1, :])
    # Higher confidence level yields wider intervals on average.
    width_80 = np.mean(intervals[:, 1, 0] - intervals[:, 0, 0])
    width_90 = np.mean(intervals[:, 1, 1] - intervals[:, 0, 1])
    assert width_90 >= width_80


def test_predict_interval_marginal_coverage():
    regressor = _fitted_regressor(confidence_level=0.9)
    _, _, X_test, y_test = _make_data()
    _, intervals = regressor.predict_interval(X_test)
    covered = (y_test >= intervals[:, 0, 0]) & (y_test <= intervals[:, 1, 0])
    assert np.mean(covered) >= 0.8


def test_predict_returns_point_predictions():
    regressor = _fitted_regressor()
    _, _, X_test, _ = _make_data()
    points = regressor.predict(X_test)
    assert points.shape == (len(X_test),)


def test_conformity_scores_property():
    regressor = _fitted_regressor()
    X_conf, _, _, _ = _make_data()
    assert regressor.conformity_scores.shape == (len(X_conf),)


def test_non_prefit_fit_then_conformalize():
    X_conf, y_conf, X_test, _ = _make_data()
    regressor = ConditionalSplitConformalRegressor(
        feature_map=_phi,
        estimator=LinearRegression(),
        confidence_level=0.9,
        prefit=False,
    )
    regressor.fit(X_conf, y_conf).conformalize(X_conf, y_conf)
    _, intervals = regressor.predict_interval(X_test)
    assert intervals.shape == (len(X_test), 2, 1)


def test_asymmetric_score_two_sided():
    regressor = _fitted_regressor(
        confidence_level=0.9, conformity_score=AbsoluteConformityScore(sym=False)
    )
    _, _, X_test, y_test = _make_data()
    _, intervals = regressor.predict_interval(X_test)
    assert np.all(intervals[:, 0, 0] <= intervals[:, 1, 0])
    covered = (y_test >= intervals[:, 0, 0]) & (y_test <= intervals[:, 1, 0])
    assert np.mean(covered) >= 0.8


def test_binary_search_path():
    regressor = _fitted_regressor(confidence_level=0.9, exact=False)
    _, _, X_test, _ = _make_data()
    _, intervals = regressor.predict_interval(X_test[:10])
    assert np.all(intervals[:, 0, 0] <= intervals[:, 1, 0])


def test_binary_search_path_asymmetric():
    # Asymmetric score sends one cutoff through the quantile < 0.5 branch.
    regressor = _fitted_regressor(
        confidence_level=0.9,
        exact=False,
        conformity_score=AbsoluteConformityScore(sym=False),
    )
    _, _, X_test, _ = _make_data()
    _, intervals = regressor.predict_interval(X_test[:8])
    assert np.all(intervals[:, 0, 0] <= intervals[:, 1, 0])


def test_predict_conditional_cutoff_with_explicit_bounds():
    regressor = _fitted_regressor(confidence_level=0.9, exact=False)
    _, _, X_test, _ = _make_data()
    cutoff = regressor._predict_conditional_cutoff(
        0.9, X_test[0].reshape(1, -1), S_min=0.0, S_max=3.0
    )
    assert np.isfinite(cutoff)


def test_randomize():
    regressor = _fitted_regressor(confidence_level=0.9, randomize=True)
    _, _, X_test, _ = _make_data()
    _, intervals = regressor.predict_interval(X_test[:10])
    assert intervals.shape == (10, 2, 1)


def test_predict_interval_before_conformalize_raises():
    regressor = ConditionalSplitConformalRegressor(
        feature_map=_phi, estimator=LinearRegression().fit(*_make_data()[:2])
    )
    with pytest.raises(ValueError, match="conformalize"):
        regressor.predict_interval(_make_data()[2])


def test_minimize_interval_width_not_supported():
    regressor = _fitted_regressor()
    _, _, X_test, _ = _make_data()
    with pytest.raises(NotImplementedError, match="minimize_interval_width"):
        regressor.predict_interval(X_test, minimize_interval_width=True)


def test_rank_deficient_basis():
    # A redundant column makes Phi rank-deficient, triggering the SVD reduction.
    def phi_redundant(x):
        return np.column_stack([np.ones(len(x)), x[:, 0], 2 * x[:, 0]])

    X_conf, y_conf, X_test, _ = _make_data()
    regressor = ConditionalSplitConformalRegressor(
        feature_map=phi_redundant,
        estimator=LinearRegression().fit(X_conf, y_conf),
        confidence_level=0.9,
    )
    regressor.conformalize(X_conf, y_conf)
    _, intervals = regressor.predict_interval(X_test[:10])
    assert np.all(intervals[:, 0, 0] <= intervals[:, 1, 0])


def test_infinite_bounds_when_basis_degenerate():
    # Phi without intercept maps the origin to the zero vector, so the exact
    # cutoff is infinite there. Use an asymmetric score to exercise both the
    # +inf (upper quantile) and -inf (lower quantile) returns.
    def phi_no_intercept(x):
        return x

    X_conf, y_conf, _, _ = _make_data()
    regressor = ConditionalSplitConformalRegressor(
        feature_map=phi_no_intercept,
        estimator=LinearRegression().fit(X_conf, y_conf),
        confidence_level=0.9,
        conformity_score=AbsoluteConformityScore(sym=False),
    )
    regressor.conformalize(X_conf, y_conf)
    X_test = np.array([[0.0], [0.5]])
    _, intervals = regressor.predict_interval(X_test)
    assert not np.isfinite(intervals[0, 0, 0])
    assert not np.isfinite(intervals[0, 1, 0])


def test_rkhs_kernel_path():
    X_conf, y_conf, X_test, _ = _make_data(n=120)
    regressor = ConditionalSplitConformalRegressor(
        feature_map=_phi,
        estimator=LinearRegression().fit(X_conf, y_conf),
        confidence_level=0.9,
        exact=False,
        infinite_params={"kernel": "rbf", "gamma": 0.1, "lambda": 1},
    )
    regressor.conformalize(X_conf, y_conf)
    _, intervals = regressor.predict_interval(X_test[:5])
    assert np.all(intervals[:, 0, 0] <= intervals[:, 1, 0])


def test_exact_with_kernel_raises():
    X_conf, y_conf, X_test, _ = _make_data(n=120)
    regressor = ConditionalSplitConformalRegressor(
        feature_map=_phi,
        estimator=LinearRegression().fit(X_conf, y_conf),
        confidence_level=0.9,
        exact=True,
        infinite_params={"kernel": "rbf", "gamma": 0.1, "lambda": 1},
    )
    regressor.conformalize(X_conf, y_conf)
    with pytest.raises(ValueError, match="RKHS"):
        regressor.predict_interval(X_test[:2])


def _make_multiclass_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    logits = np.column_stack([3 * X[:, 0], -1.5 * X[:, 0] + 2 * X[:, 1], np.zeros(n)])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=p) for p in probs])
    n_half = n // 2
    return X[:n_half], y[:n_half], X[n_half:], y[n_half:]


def _fitted_classifier(confidence_level=0.9, conformity_score="lac", **kwargs):
    X_conf, y_conf, _, _ = _make_multiclass_data()
    estimator = LogisticRegression().fit(X_conf, y_conf)
    classifier = ConditionalSplitConformalClassifier(
        feature_map=_phi,
        estimator=estimator,
        confidence_level=confidence_level,
        conformity_score=conformity_score,
        prefit=True,
        **kwargs,
    )
    return classifier.conformalize(X_conf, y_conf)


class TestConditionalSplitConformalClassifier:
    def test_predict_set_shapes_and_nesting(self):
        classifier = _fitted_classifier(confidence_level=[0.8, 0.9])
        _, _, X_test, _ = _make_multiclass_data()
        labels, sets = classifier.predict_set(X_test[:20])
        assert labels.shape == (20,)
        assert sets.shape == (20, 3, 2)
        assert sets.dtype == bool
        # Higher confidence level yields larger sets on average.
        assert sets[:, :, 1].sum() >= sets[:, :, 0].sum()

    def test_predict_set_marginal_coverage(self):
        classifier = _fitted_classifier(confidence_level=0.9)
        _, _, X_test, y_test = _make_multiclass_data()
        _, sets = classifier.predict_set(X_test)
        covered = sets[np.arange(len(y_test)), y_test, 0]
        assert np.mean(covered) >= 0.8

    def test_predict_returns_labels(self):
        classifier = _fitted_classifier()
        _, _, X_test, _ = _make_multiclass_data()
        labels = classifier.predict(X_test)
        assert labels.shape == (len(X_test),)
        assert set(labels) <= {0, 1, 2}

    def test_aps_conformity_score(self):
        classifier = _fitted_classifier(confidence_level=0.9, conformity_score="aps")
        _, _, X_test, y_test = _make_multiclass_data()
        _, sets = classifier.predict_set(X_test[:20])
        assert sets.shape == (20, 3, 1)
        covered = sets[np.arange(20), y_test[:20], 0]
        assert np.mean(covered) >= 0.8

    def test_non_prefit_fit_then_conformalize(self):
        X_conf, y_conf, X_test, _ = _make_multiclass_data()
        classifier = ConditionalSplitConformalClassifier(
            feature_map=_phi,
            estimator=LogisticRegression(),
            confidence_level=0.9,
            prefit=False,
        )
        classifier.fit(X_conf, y_conf).conformalize(X_conf, y_conf)
        _, sets = classifier.predict_set(X_test[:10])
        assert sets.shape == (10, 3, 1)

    def test_binary_search_path(self):
        classifier = _fitted_classifier(confidence_level=0.9, exact=False)
        _, _, X_test, _ = _make_multiclass_data()
        _, sets = classifier.predict_set(X_test[:10])
        assert sets.shape == (10, 3, 1)

    @pytest.mark.parametrize("conformity_score", ["top_k", "raps"])
    def test_unsupported_conformity_score_raises(self, conformity_score):
        with pytest.raises(ValueError, match="thresholding"):
            ConditionalSplitConformalClassifier(
                feature_map=_phi, conformity_score=conformity_score
            )

    def test_predict_set_before_conformalize_raises(self):
        X_conf, y_conf, X_test, _ = _make_multiclass_data()
        classifier = ConditionalSplitConformalClassifier(
            feature_map=_phi, estimator=LogisticRegression().fit(X_conf, y_conf)
        )
        with pytest.raises(ValueError, match="conformalize"):
            classifier.predict_set(X_test)


def test_binary_search_function():
    # f(x) = x - 2 crosses zero at 2.
    lower, upper = binary_search(lambda x: x - 2.0, 0.0, 5.0, tol=1e-4)
    assert lower <= 2.0 <= upper
    assert upper - lower <= 1e-3


def test_import_cvxpy_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "cvxpy", None)
    with pytest.raises(ImportError, match="mapie\\[conditional\\]"):
        _import_cvxpy()


@pytest.mark.parametrize("quantile_level", [0.9, 0.3])
def test_solve_dual_default_threshold(quantile_level):
    regressor = _fitted_regressor(confidence_level=0.9)
    _, _, X_test, _ = _make_data()
    n_calib = len(regressor.scores_calib)
    quantiles = np.ones((n_calib + 1, 1)) * quantile_level
    value = _solve_dual(
        float(np.median(regressor.scores_calib)),
        gcc=regressor,
        x_test=X_test[0].reshape(1, -1),
        quantiles=quantiles,
        threshold=None,
    )
    assert np.isfinite(value)


def test_setup_cvx_problem_default_phi():
    X_conf, _, _, _ = _make_data()
    scores = np.abs(np.random.default_rng(0).normal(size=len(X_conf)))
    problem = setup_cvx_problem(X_conf, scores, None)
    assert "weights" in problem.var_dict


def test_finish_dual_setup_without_kernel():
    X_conf, _, _, _ = _make_data()
    scores = np.abs(np.random.default_rng(0).normal(size=len(X_conf)))
    phi = _phi(X_conf)
    problem = setup_cvx_problem(X_conf, scores, phi)
    x_row = X_conf[0].reshape(1, -1)
    out = finish_dual_setup(problem, 0.5, x_row, 0.9, _phi(x_row), X_conf, {})
    assert out.param_dict["quantile"].value == 0.9
