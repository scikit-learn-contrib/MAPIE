from typing import Optional

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression

from mapie.utils import check_null_weight


X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])

X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=1.0, random_state=1)


# TODO: add doctests to utils.py
# TODO: document interaction between cv and method parameters
# TODO: write an example using cv="prefit"
# TODO: complete unit tests for utils
# TODO: complete unit test for estimators


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


def test_fit_estimator():
    pass
