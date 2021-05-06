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
from sklearn.datasets import load_boston, make_regression
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import assert_almost_equal
from sklearn.model_selection import LeaveOneOut, Kfold
from sklearn.utils.estimator_checks import parametrize_with_checks

from mapie.estimators import MapieRegressor
from mapie.metrics import coverage_score


X_boston, y_boston = load_boston(return_X_y=True)
X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X_reg, y_reg = make_regression(n_samples=500, n_features=10, random_state=1)

SEED = 59
np.random.seed(SEED)
y_reg = y_reg + np.random.normal(0, 1, 500)

ALL_RETURN_PREDS = ["single", "median"]
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
    assert mapie.return_pred == "single"


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


@pytest.mark.parametrize("return_pred", ["dummy", "ensemble", "multi", "mean"])
def test_invalid_return_pred(return_pred: str) -> None:
    """Test that invalid return_pred raise errors."""
    mapie = MapieRegressor(return_pred=return_pred)
    with pytest.raises(ValueError, match=r".*Invalid return_pred.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("return_pred", ALL_RETURN_PREDS)
def test_valid_return_pred(return_pred: str) -> None:
    """Test that valid return_pred raise no errors."""
    mapie = MapieRegressor(return_pred=return_pred)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", ["cv", DummyRegressor()])
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie = MapieRegressor(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [None, 5, Kfold(), LeaveOneOut()])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = MapieRegressor(cv=cv)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", ["dummy"])
def test_invalid_method_in_fit(monkeypatch: Any, method: str) -> None:
    """Test error in select_cv when invalid method is selected."""
    monkeypatch.setattr(MapieRegressor, "_check_parameters", lambda _: None)
    mapie = MapieRegressor(DummyRegressor(), method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.fit(X_boston, y_boston)


@pytest.mark.parametrize("method", ["dummy"])
def test_invalid_method_in_predict(monkeypatch: Any, method: str) -> None:
    """Test message in predict when invalid method is selected."""
    monkeypatch.setattr(MapieRegressor, "_check_parameters", lambda _: None)
    monkeypatch.setattr(MapieRegressor, "_select_cv", lambda _: LeaveOneOut())
    mapie = MapieRegressor(DummyRegressor(), method=method)
    mapie.fit(X_boston, y_boston)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.predict(X_boston)


@pytest.mark.parametrize("method", all_methods)
def test_fit_attribute(method: str) -> None:
    """Test class attributes shared by all PI methods."""
    mapie = MapieRegressor(DummyRegressor(), method=method)
    mapie.fit(X_reg, y_reg)
    assert hasattr(mapie, 'single_estimator_')
    assert hasattr(mapie, 'residuals_')


@pytest.mark.parametrize("method", jackknife_methods + cv_methods)
def test_jkcv_fit_attribute(method: str) -> None:
    """Test class attributes shared by jackknife and CV methods."""
    mapie = MapieRegressor(DummyRegressor(), method=method)
    mapie.fit(X_reg, y_reg)
    assert hasattr(mapie, 'estimators_')
    assert hasattr(mapie, 'k_')


@pytest.mark.parametrize("method", cv_methods)
def test_cv_attributes(method: str) -> None:
    """Test class attributes shared by CV methods."""
    mapie = MapieRegressor(DummyRegressor(), method=method, shuffle=False)
    mapie.fit(X_reg, y_reg)
    assert mapie.random_state is None


def test_predinterv_outputshape() -> None:
    """Test that number of observations given by predict method is equal to input data."""
    mapie = MapieRegressor(DummyRegressor())
    mapie.fit(X_reg, y_reg)
    assert mapie.predict(X_reg).shape[0] == X_reg.shape[0]
    assert mapie.predict(X_reg).shape[1] == 3


@pytest.mark.parametrize("method", all_methods)
def test_results(method: str) -> None:
    """
    Test that MapieRegressor applied on a linear regression model
    fitted on a linear curve results in null uncertainty.
    """
    mapie = MapieRegressor(LinearRegression(), method=method, n_splits=3)
    mapie.fit(X_toy, y_toy)
    y_preds = mapie.predict(X_toy)
    y_low, y_up = y_preds[:, 1], y_preds[:, 2]
    assert_almost_equal(y_up, y_low, 10)


@pytest.mark.parametrize("return_pred", ["single", "median"])
def test_prediction_between_low_up(return_pred: str) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    mapie = MapieRegressor(LinearRegression(), return_pred=return_pred)
    mapie.fit(X_boston, y_boston)
    y_preds = mapie.predict(X_boston)
    y_pred, y_low, y_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]
    assert (y_pred >= y_low).all() & (y_pred <= y_up).all()


@pytest.mark.parametrize("method", all_methods)
def test_linreg_results(method: str) -> None:
    """Test expected PIs for a multivariate linear regression problem with fixed random seed."""
    mapie = MapieRegressor(
        LinearRegression(), method=method, alpha=0.05, random_state=SEED
    )
    mapie.fit(X_reg, y_reg)
    y_preds = mapie.predict(X_reg)
    preds_low, preds_up = y_preds[:, 1], y_preds[:, 2]
    assert_almost_equal((preds_up-preds_low).mean(), expected_widths[method], 2)
    assert_almost_equal(coverage_score(y_reg, preds_low, preds_up), expected_coverages[method], 2)


@parametrize_with_checks([MapieRegressor(LinearRegression())])  # type: ignore
def test_sklearn_compatible_estimator(estimator: Any, check: Any) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    if check.func.__name__ not in SKLEARN_EXCLUDED_CHECKS:
        check(estimator)
