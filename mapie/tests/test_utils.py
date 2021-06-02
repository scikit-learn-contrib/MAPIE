from __future__ import annotations
from typing import Optional, Any

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from mapie.utils import check_null_weight, fit_estimator


X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])


class DumbRegressor:

    def fit(
        self, X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> DumbRegressor:
        self.fitted_ = True
        return self


def test_check_null_weight_with_none() -> None:
    """Test that the function has no effect if sample weight is None."""
    sw_out, X_out, y_out = check_null_weight(None, X_toy, y_toy)
    assert sw_out is None
    np.testing.assert_almost_equal(X_out, X_toy)
    np.testing.assert_almost_equal(y_out, y_toy)


def test_check_null_weight_with_nonzeros() -> None:
    """Test that the function has no effect if sample weight is never zero."""
    sample_weight = np.ones_like(y_toy)
    sw_out, X_out, y_out = check_null_weight(sample_weight, X_toy, y_toy)
    np.testing.assert_almost_equal(sw_out, sample_weight)
    np.testing.assert_almost_equal(X_out, X_toy)
    np.testing.assert_almost_equal(y_out, y_toy)


def test_check_null_weight_with_zeros() -> None:
    """Test that the function reduces the shape if there are zeros."""
    sample_weight = np.ones_like(y_toy)
    sample_weight[:1] = 0.0
    sw_out, X_out, y_out = check_null_weight(sample_weight, X_toy, y_toy)
    np.testing.assert_almost_equal(sw_out, np.array([1, 1, 1, 1, 1]))
    np.testing.assert_almost_equal(X_out, np.array([[1], [2], [3], [4], [5]]))
    np.testing.assert_almost_equal(y_out, np.array([7, 9, 11, 13, 15]))


@pytest.mark.parametrize("estimator", [LinearRegression(), DumbRegressor()])
@pytest.mark.parametrize("sample_weight", [None, np.ones_like(y_toy)])
def test_fit_estimator(
    estimator: Any,
    sample_weight: Optional[np.ndarray]
) -> None:
    """Test that the returned estimator is always fitted."""
    estimator = fit_estimator(estimator, X_toy, y_toy, sample_weight)
    check_is_fitted(estimator)


def test_fit_estimator_sample_weight() -> None:
    """Test that sample weight is taken into account if possible."""
    X = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([21, 7, 9, 11, 13, 15])
    sample_weight = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    estimator_1 = fit_estimator(LinearRegression(), X, y)
    estimator_2 = fit_estimator(LinearRegression(), X, y, sample_weight)
    y_pred_1 = estimator_1.predict(X)
    y_pred_2 = estimator_2.predict(X)
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(y_pred_1, y_pred_2)
