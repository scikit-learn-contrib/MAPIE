from __future__ import annotations
from typing import Optional, Any

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import make_regression

from mapie.utils import (
    check_null_weight,
    fit_estimator,
    check_alpha,
    check_n_features_in,
    check_alpha_and_n_samples,
    check_n_jobs,
    check_verbose
)

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])

n_features = 10

X, y = make_regression(
    n_samples=500,
    n_features=n_features,
    noise=1.0,
    random_state=1
)


class DumbEstimator:

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> DumbEstimator:
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


@pytest.mark.parametrize("estimator", [LinearRegression(), DumbEstimator()])
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


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2, 2.5, "a", [[0.5]], ["a", "b"]])
def test_invalid_alpha(alpha: Any) -> None:
    """Test that invalid alphas raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        check_alpha(alpha=alpha)


@pytest.mark.parametrize(
    "alpha",
    [
        np.linspace(0.05, 0.95, 5),
        [0.05, 0.95],
        (0.05, 0.95),
        np.array([0.05, 0.95]),
        None
    ]
)
def test_valid_alpha(alpha: Any) -> None:
    """Test that valid alphas raise no errors."""
    check_alpha(alpha=alpha)


@pytest.mark.parametrize("cv", ["prefit", None])
def test_valid_shape_no_n_features_in(cv: Any) -> None:
    """
    Test that estimators fitted with a right number of features
    but missing an n_features_in_ attribute raise no errors.
    """
    estimator = DumbEstimator()
    n_features_in = check_n_features_in(
        X=X, cv=cv, estimator=estimator
    )
    assert n_features_in == n_features


@pytest.mark.parametrize(
    "alpha",
    [
        np.linspace(0.05, 0.95, 5),
        [0.05, 0.95],
        (0.05, 0.95),
        np.array([0.05, 0.95])
    ]
)
def test_valid_calculation_of_quantile(alpha: Any) -> None:
    """Test that valid alphas raise no errors."""
    n = 30
    check_alpha_and_n_samples(alpha, n)


@pytest.mark.parametrize(
    "alpha",
    [
        np.linspace(0.05, 0.07),
        [0.05, 0.07, 0.9],
        (0.05, 0.07, 0.9),
        np.array([0.05, 0.07, 0.9])
    ]
)
def test_invalid_calculation_of_quantile(alpha: Any) -> None:
    """Test that alpha with 1/alpha > number of samples  raise errors."""
    n = 10
    with pytest.raises(
        ValueError, match=r".*Number of samples of the score is too low*"
    ):
        check_alpha_and_n_samples(alpha, n)


def test_invalid_prefit_estimator_shape() -> None:
    """
    Test that estimators fitted with a wrong number of features raise errors.
    """
    estimator = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match=r".*mismatch between.*"):
        check_n_features_in(
            X_toy, cv="prefit", estimator=estimator
        )


@pytest.mark.parametrize("n_jobs", ["dummy", 0, 1.5, [1, 2]])
def test_invalid_n_jobs(n_jobs: Any) -> None:
    """Test that invalid n_jobs raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid n_jobs argument.*"):
        check_n_jobs(n_jobs)


@pytest.mark.parametrize("n_jobs", [-5, -1, 1, 4])
def test_valid_n_jobs(n_jobs: Any) -> None:
    """Test that valid n_jobs raise no errors."""
    check_n_jobs(n_jobs)


@pytest.mark.parametrize("verbose", ["dummy", -1, 1.5, [1, 2]])
def test_invalid_verbose(verbose: Any) -> None:
    """Test that invalid verboses raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid verbose argument.*"):
        check_verbose(verbose)


@pytest.mark.parametrize("verbose", [0, 10, 50])
def test_valid_verbose(verbose: Any) -> None:
    """Test that valid verboses raise no errors."""
    check_verbose(verbose)
