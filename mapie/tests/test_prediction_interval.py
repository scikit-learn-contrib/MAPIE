"""
Testing for  prediction_interval module.

List of tests:
- Test input parameters
- Test created attributes depending on the method
- Test output and results
"""
from typing import Any

import pytest
import numpy as np
from sklearn.datasets import load_boston, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import assert_almost_equal
from sklearn.model_selection import LeaveOneOut

from mapie.prediction_interval import PredictionInterval


X_boston, y_boston = load_boston(return_X_y=True)
X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X_reg, y_reg = make_regression(
    n_samples=500, n_features=10, random_state=1
)
SEED = 59
np.random.seed(SEED)
y_reg = y_reg + np.random.normal(0, 1, 500)
all_methods = [
    "naive", "jackknife", "jackknife_plus", "jackknife_minmax", "cv", "cv_plus", "cv_minmax"
]
cv_methods = ["cv", "cv_plus", "cv_minmax"]
jackknife_methods = ["jackknife", "jackknife_plus", "jackknife_minmax"]
standard_methods = ["naive", "jackknife", "cv"]
plus_methods = ["jackknife_plus", "cv_plus"]
minmax_methods = ["jackknife_minmax", "cv_minmax"]
expected_widths = {
    "naive": 3.76,
    "jackknife": 3.85,
    "jackknife_plus": 3.86,
    "jackknife_minmax": 3.91,
    "cv": 4.02,
    "cv_plus": 4.01,
    "cv_minmax": 4.21
}
expected_coverages = {
    "naive": 0.952,
    "jackknife": 0.952,
    "jackknife_plus": 0.952,
    "jackknife_minmax": 0.952,
    "cv": 0.958,
    "cv_plus": 0.956,
    "cv_minmax": 0.966
}


def test_optional_input_values() -> None:
    """Test default values of input parameters."""
    pireg = PredictionInterval(DummyRegressor())
    assert pireg.method == "jackknife_plus"
    assert pireg.alpha == 0.1
    assert pireg.n_splits == 10
    assert pireg.shuffle
    assert pireg.return_pred == "single"
    assert pireg.random_state is None


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2])
def test_invalid_alpha(alpha: int) -> None:
    pireg = PredictionInterval(DummyRegressor(), alpha=alpha)
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        pireg.fit(X_boston, y_boston)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    PredictionInterval(DummyRegressor())


def test_fitted() -> None:
    """Test that fit does not crash."""
    pireg = PredictionInterval(DummyRegressor())
    pireg.fit(X_reg, y_reg)


def test_predicted() -> None:
    """Test that predict does not crash."""
    pireg = PredictionInterval(DummyRegressor())
    pireg.fit(X_reg, y_reg)
    pireg.predict(X_reg)


def test_not_fitted() -> None:
    """Test error message when predict is called before fit."""
    pireg = PredictionInterval(DummyRegressor())
    with pytest.raises(NotFittedError, match=r".*not fitted.*"):
        pireg.predict(X_reg)


@pytest.mark.parametrize("method", ["dummy", "cv_dummy", "jackknife_dummy"])
def test_invalid_method_in_check_parameters(method: str) -> None:
    """Test error in check_parameters when invalid method is selected."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        pireg.fit(X_boston, y_boston)


@pytest.mark.parametrize("method", ["dummy"])
def test_invalid_method_in_fit(monkeypatch: Any, method: str) -> None:
    """Test error in select_cv when invalid method is selected."""
    monkeypatch.setattr(PredictionInterval, "_check_parameters", lambda _: None)
    pireg = PredictionInterval(DummyRegressor(), method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        pireg.fit(X_boston, y_boston)


@pytest.mark.parametrize("method", ["dummy"])
def test_invalid_method_in_predict(monkeypatch: Any, method: str) -> None:
    """Test message in predict when invalid method is selected."""
    monkeypatch.setattr(PredictionInterval, "_check_parameters", lambda _: None)
    monkeypatch.setattr(PredictionInterval, "_select_cv", lambda _: LeaveOneOut())
    pireg = PredictionInterval(DummyRegressor(), method=method)
    pireg.fit(X_boston, y_boston)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        pireg.predict(X_boston)


@pytest.mark.parametrize("method", all_methods)
def test_single_estimator_attribute(method: str) -> None:
    """Test class attributes shared by all PI methods."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'single_estimator_')


@pytest.mark.parametrize("method", standard_methods)
def test_quantile_attribute(method: str) -> None:
    """Test quantile attribute."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'quantile_')
    assert (pireg.quantile_ >= 0)


@pytest.mark.parametrize("method", jackknife_methods + cv_methods)
def test_jkcv_attribute(method: str) -> None:
    """Test class attributes shared by jackknife and CV methods."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'estimators_')
    assert hasattr(pireg, 'residuals_split_')
    assert hasattr(pireg, 'y_train_pred_split_')


@pytest.mark.parametrize("method", cv_methods)
def test_cv_attributes(method: str) -> None:
    """Test class attributes shared by CV methods."""
    pireg = PredictionInterval(DummyRegressor(), method=method, shuffle=False)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'val_fold_ids_')
    assert pireg.random_state is None


def test_none_estimator() -> None:
    """Test error raised when estimator is None."""
    pireg = PredictionInterval(None)
    with pytest.raises(ValueError, match=r".*Invalid none estimator.*"):
        pireg.fit(X_boston, y_boston)


def test_predinterv_outputshape() -> None:
    """
    Test that number of observations given by predict method is equal to
    input data.
    """
    pireg = PredictionInterval(DummyRegressor())
    pireg.fit(X_reg, y_reg)
    assert pireg.predict(X_reg).shape[0] == X_reg.shape[0]
    assert pireg.predict(X_reg).shape[1] == 3


@pytest.mark.parametrize("method", all_methods)
def test_results(method: str) -> None:
    """
    Test that PredictionInterval applied on a linear regression model
    fitted on a linear curve results in null uncertainty.
    """
    pireg = PredictionInterval(LinearRegression(), method=method, n_splits=3)
    pireg.fit(X_toy, y_toy)
    y_preds = pireg.predict(X_toy)
    y_low, y_up = y_preds[:, 1], y_preds[:, 2]
    assert_almost_equal(y_up, y_low, 10)


@pytest.mark.parametrize("return_pred", ["ensemble", "single"])
def test_prediction_between_low_up(return_pred: str) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    pireg = PredictionInterval(LinearRegression(), return_pred=return_pred)
    pireg.fit(X_boston, y_boston)
    y_preds = pireg.predict(X_boston)
    y_pred, y_low, y_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]
    assert (y_pred >= y_low).all() & (y_pred <= y_up).all()


@pytest.mark.parametrize("method", all_methods)
def test_linreg_results(method: str) -> None:
    """
    Test expected PIs for a multivariate linear regression problem
    with fixed random seed.
    """
    pireg = PredictionInterval(
        LinearRegression(), method=method, alpha=0.05, random_state=SEED
    )
    pireg.fit(X_reg, y_reg)
    y_preds = pireg.predict(X_reg)
    preds_low, preds_up = y_preds[:, 1], y_preds[:, 2]
    assert_almost_equal((preds_up-preds_low).mean(), expected_widths[method], 2)
    assert_almost_equal(((preds_up >= y_reg) & (preds_low <= y_reg)).mean(), expected_coverages[method], 2)
