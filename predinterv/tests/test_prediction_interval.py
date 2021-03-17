"""
Testing for  prediction_interval module.

List of tests:
- Test input parameters
- Test created attributes depending on the method
- Test output and results
"""
import numpy as np
import pytest

from sklearn.datasets import load_boston, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.utils._testing import assert_almost_equal

from ..simai.prediction_interval import PredictionInterval


X_boston, y_boston = load_boston(return_X_y=True)
X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X_reg, y_reg = make_regression(
    n_samples=500, n_features=10, random_state=1
)
all_methods = [
    "naive", "jackknife", "jackknife_plus", "jackknife_minmax", "cv", "cv_plus", "cv_minmax"
]
cv_methods = ["cv", "cv_plus", "cv_minmax"]
jackknife_methods = ["jackknife", "jackknife_plus", "jackknife_minmax"]
standard_methods = ["naive", "jackknife", "cv"]
plus_methods = ["jackknife_plus", "cv_plus"]
minmax_methods = ["jackknife_minmax", "cv_minmax"]


def test_optional_input_values():
    """Test default values of input parameters."""
    pireg = PredictionInterval(DummyRegressor())
    assert pireg.method == "jackknife_plus"
    assert pireg.alpha == 0.1
    assert pireg.n_splits == 10
    assert pireg.shuffle
    assert pireg.return_pred == "single"
    assert pireg.random_state is None


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2])
def test_invalid_alpha(alpha):
    pireg = PredictionInterval(DummyRegressor(), alpha=alpha)
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        pireg.fit(X_boston, y_boston)


def test_initialized():
    """Test that initialization does not crash."""
    PredictionInterval(DummyRegressor())


def test_fitted():
    """Test that fit does not crash."""
    pireg = PredictionInterval(DummyRegressor())
    pireg.fit(X_reg, y_reg)


def test_predicted():
    """Test that predict does not crash."""
    pireg = PredictionInterval(DummyRegressor())
    pireg.fit(X_reg, y_reg)
    pireg.predict(X_reg)


def test_not_fitted():
    """Test error message when predict is called before fit."""
    pireg = PredictionInterval(DummyRegressor())
    with pytest.raises(NotFittedError, match=r".*not fitted.*"):
        pireg.predict(X_reg)


@pytest.mark.parametrize("method", ["dummy", "cv_dummy", "jackknife_dummy"])
def test_invalid_method(method):
    """Test when invalid method is selected."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        pireg.fit(X_boston, y_boston)


@pytest.mark.parametrize("method", all_methods)
def test_estimator(method):
    """Test class attributes shared by all PI methods."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'single_estimator_')


@pytest.mark.parametrize("method", standard_methods)
def test_quantile_attribute(method):
    """Test quantile attribute."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'quantile_')
    assert (pireg.quantile_ >= 0)


@pytest.mark.parametrize("method", jackknife_methods + cv_methods)
def test_jkcv_attribute(method):
    """Test class attributes shared by jackknife and CV methods."""
    pireg = PredictionInterval(DummyRegressor(), method=method)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'estimators_')
    assert hasattr(pireg, 'residuals_split_')
    assert hasattr(pireg, 'y_train_pred_split_')


@pytest.mark.parametrize("method", cv_methods)
def test_cv_attributes(method):
    """Test class attributes shared by CV methods."""
    pireg = PredictionInterval(DummyRegressor(), method=method, shuffle=False)
    pireg.fit(X_reg, y_reg)
    assert hasattr(pireg, 'val_fold_ids_')
    assert pireg.random_state is None


def test_none_estimator():
    """Test error raised when estimator is None."""
    pireg = PredictionInterval(None)
    with pytest.raises(ValueError, match=r".*Invalid none estimator.*"):
        pireg.fit(X_boston, y_boston)


def test_predinterv_outputshape():
    """
    Test that number of observations given by predict method is equal to
    input data.
    """
    pireg = PredictionInterval(DummyRegressor())
    pireg.fit(X_reg, y_reg)
    assert pireg.predict(X_reg).shape[0] == X_reg.shape[0]
    assert pireg.predict(X_reg).shape[1] == 3


@pytest.mark.parametrize("method", all_methods)
def test_results(method):
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
def test_ensemble(return_pred):
    """Test that prediction lies between low and up prediction intervals."""
    pireg = PredictionInterval(LinearRegression(), return_pred=return_pred)
    pireg.fit(X_boston, y_boston)
    y_preds = pireg.predict(X_boston)
    y_pred, y_low, y_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]
    assert (y_pred >= y_low).all() & (y_pred <= y_up).all()
