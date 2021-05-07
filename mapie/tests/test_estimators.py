"""
Testing for mapieregressor module.

List of tests:
- Test input parameters
- Test created attributes depending on the method
- Test output and results
"""
from typing import Any

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, Kfold
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from mapie.estimators import MapieRegressor
from mapie.metrics import coverage_score


X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X_reg, y_reg = make_regression(n_samples=500, n_features=10, random_state=1)

SEED = 59
np.random.seed(SEED)
y_reg = y_reg + np.random.normal(0, 1, 500)

ALL_METHODS = ["naive", "base", "plus", "minmax"]
NAIVE_METHOD = "naive"
NON_NAIVE_METHODS = ["base", "plus", "minmax"]
EXPECTED_WIDTHS = {
    "naive": 3.76,
    "jackknife": 3.85,
    "jackknife_plus": 3.86,
    "jackknife_minmax": 3.91,
    "cv": 3.92,
    "cv_plus": 3.99,
    "cv_minmax": 4.13
}
EXPECTED_COVERAGES = {
    "naive": 0.952,
    "jackknife": 0.952,
    "jackknife_plus": 0.952,
    "jackknife_minmax": 0.952,
    "cv": 0.958,
    "cv_plus": 0.956,
    "cv_minmax": 0.966
}

SKLEARN_EXCLUDED_CHECKS = {
    "check_regressors_train",
    "check_pipeline_consistency",
    "check_fit_score_takes_y",
}


# TODO:
# beware of ALL_METHODS, you should also check for cv et return_pred
# rethink all tests, especially the last ones with expected widths/coverage
# assert checks are called during fit
# assert checks are called during predict
# assert all corner cases of predict
# update documentation
# update examples

def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieRegressor()


def test_fit() -> None:
    """Test that fit raises no errors."""
    mapie = MapieRegressor()
    mapie.fit(X_toy, y_toy)


def test_fit_predict() -> None:
    """Test that fit-predict raises no errors."""
    mapie = MapieRegressor()
    mapie.fit(X_toy, y_toy)
    mapie.predict(X_toy)


def test_no_fit_predict() -> None:
    """Test that predict before fit raises errors"""
    mapie = MapieRegressor(estimator=DummyRegressor())
    with pytest.raises(NotFittedError, match=r".*not fitted.*"):
        mapie.predict(X_toy)


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie = MapieRegressor(estimator=DummyRegressor())
    assert mapie.estimator is None
    assert mapie.alpha == 0.1
    assert mapie.method == "plus"
    assert mapie.cv == 5
    assert not mapie.ensemble


@pytest.mark.parametrize("estimator", [0, "estimator", Kfold()])
def test_invalid_estimator(estimator: Any) -> None:
    """Test that invalid estimators raise errors."""
    mapie = MapieRegressor(estimator=estimator)
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie.fit(X_toy, y_toy)


def test_none_estimator() -> None:
    """Test that None estimator defaults to LinearRegression."""
    mapie = MapieRegressor(estimator=None)
    mapie.fit(X_toy, y_toy)
    assert isinstance(mapie.single_estimator_, LinearRegression)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_valid_estimator(method: str) -> None:
    """Test that valid estimators are not corrupted."""
    mapie = MapieRegressor(estimator=DummyRegressor(), method=method)
    mapie.fit(X_toy, y_toy)
    assert isinstance(mapie.single_estimator_, DummyRegressor)
    for estimator in mapie.estimators_:
        assert isinstance(estimator, DummyRegressor)


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2])
def test_invalid_alpha(alpha: int) -> None:
    """Test that invalid alphas raise errors."""
    mapie = MapieRegressor(alpha=alpha)
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("alpha", np.linspace(0, 1, 5))
def test_valid_alpha(alpha: int) -> None:
    """Test that valid alphas raise no errors."""
    mapie = MapieRegressor(alpha=alpha)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", ["dummy", "cv_dummy", "jackknife_dummy", "dummy_plus", "dummy_minmax"])
def test_invalid_method(method: str) -> None:
    """Test that invalid methods raise errors."""
    mapie = MapieRegressor(method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_valid_method(method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie = MapieRegressor(method=method)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("ensemble", ["dummy", 1, 2., [1, 2]])
def test_invalid_ensemble(ensemble: Any) -> None:
    """Test that invalid return_pred raise errors."""
    mapie = MapieRegressor(DummyRegressor(), ensemble=ensemble)
    with pytest.raises(ValueError, match=r".*Invalid ensemble.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("ensemble", [True, False])
def test_valid_ensemble(ensemble: str) -> None:
    """Test that valid ensemble raise no errors."""
    mapie = MapieRegressor(ensemble=ensemble)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [-3.14, -2, -1, 0, 1, "cv", DummyRegressor()])
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie = MapieRegressor(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [None, 2, 3, 4, 5, 10, Kfold(), LeaveOneOut()])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = MapieRegressor(cv=cv)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_fit_attributes(method: str) -> None:
    """Test fit attributes shared by all PI methods."""
    mapie = MapieRegressor(method=method)
    mapie.fit(X_toy, y_toy)
    assert hasattr(mapie, 'single_estimator_')
    assert hasattr(mapie, 'estimators_')
    assert hasattr(mapie, 'residuals_')
    assert hasattr(mapie, 'k_')


@pytest.mark.parametrize("method", ALL_METHODS)
def test_predict_output_shape(method: str) -> None:
    """Test predict output shape."""
    mapie = MapieRegressor(method=method)
    mapie.fit(X_reg, y_reg)
    assert mapie.predict(X_reg).shape[0] == X_reg.shape[0]
    assert mapie.predict(X_reg).shape[1] == 3


@pytest.mark.parametrize("method", ALL_METHODS)
def test_linear_confidence_interval(method: str) -> None:
    """
    Test that MapieRegressor applied on a linear regression model
    fitted on a linear curve results in null uncertainty.
    """
    mapie = MapieRegressor(estimator=LinearRegression(), method=method, n_splits=3)
    mapie.fit(X_toy, y_toy)
    y_preds = mapie.predict(X_toy)
    y_low, y_up = y_preds[:, 1], y_preds[:, 2]
    np.testing.assert_almost_equal(y_up, y_low)


@pytest.mark.parametrize("ensemble", [True, False])
def test_prediction_between_low_up(ensemble: bool) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    mapie = MapieRegressor(estimator=LinearRegression(), ensemble=ensemble)
    mapie.fit(X_reg, y_reg)
    y_preds = mapie.predict(X_reg)
    y_pred, y_low, y_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]
    assert (y_pred >= y_low).all() & (y_pred <= y_up).all()


@pytest.mark.parametrize("method", ALL_METHODS)
def test_linreg_results(method: str) -> None:
    """Test expected PIs for a multivariate linear regression problem with fixed random seed."""
    mapie = MapieRegressor(estimator=LinearRegression(), method=method, alpha=0.05, random_state=SEED)
    mapie.fit(X_reg, y_reg)
    y_preds = mapie.predict(X_reg)
    preds_low, preds_up = y_preds[:, 1], y_preds[:, 2]
    width_mean = (preds_up-preds_low).mean()
    coverage = coverage_score(y_reg, preds_low, preds_up)
    np.testing.assert_almost_equal(width_mean, EXPECTED_WIDTHS[method], 2)
    np.testing.assert_almost_equal(coverage, EXPECTED_COVERAGES[method], 2)


@parametrize_with_checks([MapieRegressor(LinearRegression())])  # type: ignore
def test_sklearn_compatible_estimator(estimator: Any, check: Any) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    if check.func.__name__ not in SKLEARN_EXCLUDED_CHECKS:
        check(estimator)
