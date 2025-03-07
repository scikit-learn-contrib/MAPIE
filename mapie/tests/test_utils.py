from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple

import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (BaseCrossValidator, KFold, LeaveOneOut,
                                     ShuffleSplit)
from sklearn.utils.validation import check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.regression import MapieQuantileRegressor
from mapie.utils import (check_alpha, check_alpha_and_n_samples,
                         check_array_inf, check_array_nan, check_arrays_length,
                         check_binary_zero_one, check_cv, check_gamma,
                         check_lower_upper_bounds, check_n_features_in,
                         check_n_jobs, check_n_samples, check_no_agg_cv,
                         check_null_weight, check_number_bins,
                         check_split_strategy, check_verbose,
                         compute_quantiles, fit_estimator, get_binning_groups)

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])

n_features = 10

X, y = make_regression(
    n_samples=500, n_features=n_features, noise=1.0, random_state=1
)
ALPHAS = [
    np.array([0.1]),
    np.array([0.05, 0.1, 0.2]),
]

random_state = 1234567890
prng = RandomState(random_state)
y_score = prng.random(51)
y_scores = prng.random((51, 5))
y_true = prng.randint(0, 2, 51)

results_binning = {
    "quantile":
        [
            0.03075388, 0.17261836, 0.33281326, 0.43939618,
            0.54867626, 0.64881987, 0.73440899, 0.77793816,
            0.89000413, 0.99610621
        ],
    "uniform":
        [
            0, 0.11111111, 0.22222222, 0.33333333, 0.44444444,
            0.55555556, 0.66666667, 0.77777778, 0.88888889, 1
        ],
    "array split":
        [
            0.62689056, 0.74743526, 0.87642114, 0.88321124,
            0.8916548,  0.94083846, 0.94999075, 0.98759822,
            0.99610621, np.inf
        ],
}


class DumbEstimator:
    def fit(
            self,
            X: ArrayLike,
            y: Optional[ArrayLike] = None) -> DumbEstimator:
        self.fitted_ = True
        return self


def test_check_null_weight_with_none() -> None:
    """Test that the function has no effect if sample weight is None."""
    sw_out, X_out, y_out = check_null_weight(None, X_toy, y_toy)
    assert sw_out is None
    np.testing.assert_almost_equal(np.array(X_out), X_toy)
    np.testing.assert_almost_equal(np.array(y_out), y_toy)


def test_check_null_weight_with_nonzeros() -> None:
    """Test that the function has no effect if sample weight is never zero."""
    sample_weight = np.ones_like(y_toy)
    sw_out, X_out, y_out = check_null_weight(sample_weight, X_toy, y_toy)
    np.testing.assert_almost_equal(np.array(sw_out), sample_weight)
    np.testing.assert_almost_equal(np.array(X_out), X_toy)
    np.testing.assert_almost_equal(np.array(y_out), y_toy)


def test_check_null_weight_with_zeros() -> None:
    """Test that the function reduces the shape if there are zeros."""
    sample_weight = np.ones_like(y_toy)
    sample_weight[:1] = 0.0
    sw_out, X_out, y_out = check_null_weight(sample_weight, X_toy, y_toy)
    np.testing.assert_almost_equal(np.array(sw_out), np.array([1, 1, 1, 1, 1]))
    np.testing.assert_almost_equal(
        np.array(X_out), np.array([[1], [2], [3], [4], [5]])
    )
    np.testing.assert_almost_equal(
        np.array(y_out), np.array([7, 9, 11, 13, 15])
    )


@pytest.mark.parametrize("estimator", [LinearRegression(), DumbEstimator()])
@pytest.mark.parametrize("sample_weight", [None, np.ones_like(y_toy)])
def test_fit_estimator(
    estimator: Any,
    sample_weight: Optional[NDArray]
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


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2, 2.5, "a", ["a", "b"]])
def test_invalid_alpha(alpha: Any) -> None:
    """Test that invalid alphas raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        check_alpha(alpha=alpha)


@pytest.mark.parametrize(
    "alpha",
    [
        0.95,
        [0.05, 0.95],
        (0.05, 0.95),
        np.array([0.05, 0.95]),
        None,
    ],
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
    n_features_in = check_n_features_in(X=X, cv=cv, estimator=estimator)
    assert n_features_in == n_features


@pytest.mark.parametrize(
    "alpha",
    [
        np.linspace(0.05, 0.95, 5),
        [0.05, 0.95],
        (0.05, 0.95),
        np.array([0.05, 0.95]),
    ],
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
        np.array([0.05, 0.07, 0.9]),
    ],
)
def test_invalid_calculation_of_quantile(alpha: Any) -> None:
    """Test that alpha with 1/alpha > number of samples  raise errors."""
    n = 10
    with pytest.raises(
        ValueError, match=r".*Number of samples of the score is too low.*"
    ):
        check_alpha_and_n_samples(alpha, n)


def test_invalid_prefit_estimator_shape() -> None:
    """
    Test that estimators fitted with a wrong number of features raise errors.
    """
    estimator = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match=r".*mismatch between.*"):
        check_n_features_in(X_toy, cv="prefit", estimator=estimator)


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


def test_initial_low_high_pred(caplog) -> None:
    """Test lower/upper predictions of the quantiles regression crossing"""
    y_preds = np.array([[4, 5, 2], [4, 4, 4], [2, 3, 4]])
    with caplog.at_level(logging.INFO):
        check_lower_upper_bounds(y_preds[0], y_preds[1], y_preds[2])
    assert "The predictions are ill-sorted" in caplog.text


def test_final_low_high_pred(caplog) -> None:
    """Test lower/upper predictions crossing"""
    y_preds = np.array(
        [[4, 3, 2], [3, 3, 3], [2, 3, 4]]
    )
    y_pred_low = np.array([4, 7, 2])
    y_pred_up = np.array([3, 3, 3])
    with caplog.at_level(logging.INFO):
        check_lower_upper_bounds(y_pred_low, y_pred_up, y_preds[2])
    assert "The predictions are ill-sorted" in caplog.text


def test_ensemble_in_predict() -> None:
    """Checking for ensemble defined in predict of CQR"""
    mapie_reg = MapieQuantileRegressor()
    mapie_reg.fit(X, y)
    with pytest.warns(
        UserWarning, match=r"WARNING: Alpha should not be spec.*"
    ):
        mapie_reg.predict(X, alpha=0.2)


def test_alpha_in_predict() -> None:
    """Checking for alpha defined in predict of CQR"""
    mapie_reg = MapieQuantileRegressor()
    mapie_reg.fit(X, y)
    with pytest.warns(UserWarning, match=r"WARNING: ensemble is not util*"):
        mapie_reg.predict(X, ensemble=True)


def test_compute_quantiles_value_error():
    """Test that if the size of the last axis of vector
    is different from the number of aphas an error is raised.
    """
    vector = np.random.rand(1000, 1, 1)
    alphas = [0.1, 0.2, 0.3]

    with pytest.raises(ValueError, match=r".*In case of the vector .*"):
        compute_quantiles(vector, alphas)


@pytest.mark.parametrize("alphas", ALPHAS)
def test_compute_quantiles_2D_shape(alphas: NDArray):
    """Test that the number of quantiles is equal to
    the number of alphas for a 2D input vector

    Parameters
    ----------
    alphas : NDArray
        Levels of confidence.
    """
    vector = np.random.rand(1000, 1)
    quantiles = compute_quantiles(vector, alphas)

    assert len(quantiles) == len(alphas)


@pytest.mark.parametrize("alphas", ALPHAS)
def test_compute_quantiles_3D_shape(alphas: NDArray):
    """Test that the number of quantiles is equal to
    the number of alphas for a 3D input vector
    """
    vector = np.random.rand(1000, 1)
    vector = np.repeat(vector, len(alphas), axis=1)
    quantiles = compute_quantiles(vector, alphas)

    assert len(quantiles) == len(alphas)


@pytest.mark.parametrize("alphas", ALPHAS)
def test_compute_quantiles_2D_and_3D(alphas: NDArray):
    """Test that if to matrices are equal (modulo one dimension)
    then there quantiles are the same.
    """
    vector1 = np.random.rand(1000, 1)
    vector2 = np.repeat(vector1, len(alphas), axis=1)

    quantiles1 = compute_quantiles(vector1, alphas)
    quantiles2 = compute_quantiles(vector2, alphas)

    assert (quantiles1 == quantiles2).all()


@pytest.mark.parametrize("estimator", [-1, 3, 0.2])
def test_quantile_prefit_non_iterable(estimator: Any) -> None:
    """
    Test that there is a list of estimators provided when cv='prefit'
    is called for MapieQuantileRegressor.
    """
    with pytest.raises(
        ValueError,
        match=r".*Estimator for prefit must be an iterable object.*",
    ):
        mapie_reg = MapieQuantileRegressor(estimator=estimator, cv="prefit")
        mapie_reg.fit([1, 2, 3], [4, 5, 6])


@pytest.mark.parametrize("strategy", ["quantile", "uniform", "array split"])
def test_binning_group_strategies(strategy: str) -> None:
    """Test that different strategies have the correct outputs."""
    bins_ = get_binning_groups(
        y_score, num_bins=10, strategy=strategy
    )
    np.testing.assert_allclose(
        results_binning[strategy],
        bins_,
        rtol=1e-05
    )


def test_wrong_split_strategy() -> None:
    """Test for wrong split strategies."""
    with pytest.raises(ValueError, match=r"Please provide a valid*"):
        check_split_strategy(strategy="not_valid")


def test_split_strategy_None() -> None:
    """Test what occurs if None is provided as split strategy."""
    strategy = check_split_strategy(None)
    assert strategy == "uniform"


@pytest.mark.parametrize("bins", ["random", LinearRegression(), 0.5])
def test_num_bins_not_int(bins: int) -> None:
    """Test input for bins is an integer."""
    with pytest.raises(
        ValueError,
        match=r"Please provide a bin number as an int*"
    ):
        check_number_bins(num_bins=bins)


def test_num_bins_below_zero() -> None:
    """Test input for bins is positive integer."""
    with pytest.raises(
        ValueError,
        match=r"Please provide a bin number greater*"
    ):
        check_number_bins(num_bins=-1)


def test_binary_target() -> None:
    """
    Test that input of binary will provide an error message for non binary.
    """
    with pytest.raises(
        ValueError,
        match=r"Please provide y_true as a bina*"
    ):
        check_binary_zero_one(np.array([0, 5, 4]))


def test_nan_values() -> None:
    """
    Test if array has only non-numerical values like NaNs
    """
    with pytest.raises(
        ValueError,
        match=r"Array contains only NaN*"
    ):
        check_array_nan(np.array([np.nan, np.nan, np.nan, np.nan]))


def test_inf_values() -> None:
    """
    Test if array has infinite values like +inf or -inf
    """
    with pytest.raises(
        ValueError,
        match=r"Array contains infinite va*"
    ):
        check_array_inf(np.array([1, 2, -np.inf, 4]))


def test_length() -> None:
    """
    Test if the arrays have the same size (length)
    """
    with pytest.raises(
        ValueError,
        match=r"There are arrays with different len*"
    ):
        check_arrays_length(np.array([1, 2, 3]), np.array([4, 5, 6, 7]))


def test_change_values_zero_one() -> None:
    """Test that binary output are changed to zero one outputs."""
    array_ = check_binary_zero_one(np.array([0, 4, 4]))
    assert (np.unique(array_) == np.array([0, 1])).all()


@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.9])
def test_valid_gamma(gamma: float) -> None:
    """Test a valid gamma parameter."""
    check_gamma(gamma)


@pytest.mark.parametrize("gamma", [1.5, -0.1])
def test_invalid_large_gamma(gamma: float) -> None:
    """Test a non-valid gamma parameter."""
    with pytest.raises(
        ValueError,
        match="Invalid gamma. Allowed values are between 0 and 1."
    ):
        check_gamma(gamma)


@pytest.mark.parametrize("cv", [5, "split"])
def test_check_cv_same_split_with_random_state(cv: BaseCrossValidator) -> None:
    """Test that cv generate same split with fixed random_state."""
    cv = check_cv(cv, random_state=random_state)

    train_indices_1, train_indices_2 = [], []
    for train_index, _ in cv.split(X):
        train_indices_1.append(train_index)
    for train_index, _ in cv.split(X):
        train_indices_2.append(train_index)

    for i in range(cv.get_n_splits()):
        np.testing.assert_allclose(train_indices_1[i], train_indices_2[i])


@pytest.mark.parametrize("cv", [5, "split"])
def test_check_cv_same_split_no_random_state(cv: BaseCrossValidator) -> None:
    """Test that cv generate same split with no random_state."""
    cv = check_cv(cv, random_state=None)

    train_indices_1, train_indices_2 = [], []
    for train_index, _ in cv.split(X):
        train_indices_1.append(train_index)
    for train_index, _ in cv.split(X):
        train_indices_2.append(train_index)

    for i in range(cv.get_n_splits()):
        np.testing.assert_allclose(train_indices_1[i], train_indices_2[i])


@pytest.mark.parametrize(
    "cv_result", [
        (1, True), (2, False),
        ("split", True), (KFold(5), False),
        (ShuffleSplit(1), True),
        (ShuffleSplit(2), False),
        (LeaveOneOut(), False),
    ]
)
def test_check_no_agg_cv(cv_result: Tuple) -> None:
    """Test that if `check_no_agg_cv` function returns the expected result."""
    array = ["prefit", "split"]
    cv, result = cv_result
    np.testing.assert_almost_equal(check_no_agg_cv(X_toy, cv, array), result)


@pytest.mark.parametrize("cv", [object()])
def test_check_no_agg_cv_value_error(cv: Any) -> None:
    """Test that if `check_no_agg_cv` function raises value error."""
    array = ["prefit", "split"]
    with pytest.raises(
        ValueError,
        match=r"Allowed values must have the `get_n_splits` method"
    ):
        check_no_agg_cv(X_toy, cv, array)


@pytest.mark.parametrize("n_samples", [-4, -2, -1])
def test_invalid_n_samples_int_negative(n_samples: int) -> None:
    """Test that invalid n_samples raise errors."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indices = X.copy()
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Invalid n_samples. Allowed values "
            r"are float in the range (0.0, 1.0) or"
            r" int in the range [1, inf)"
        )
    ):
        check_n_samples(X=X, n_samples=n_samples, indices=indices)


@pytest.mark.parametrize("n_samples", [0.002, 0.003, 0.04])
def test_invalid_n_samples_int_zero(n_samples: int) -> None:
    """Test that invalid n_samples raise errors."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indices = X.copy()
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"The value of n_samples is too small. "
            r"You need to increase it so that n_samples*X.shape[0] > 1"
            r"otherwise n_samples should be an int"
        )
    ):
        check_n_samples(X=X, n_samples=n_samples, indices=indices)


@pytest.mark.parametrize("n_samples", [-5.5, -4.3, -0.2, 1.2, 2.5, 3.4])
def test_invalid_n_samples_float(n_samples: float) -> None:
    """Test that invalid n_samples raise errors."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indices = X.copy()
    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Invalid n_samples. Allowed values "
            r"are float in the range (0.0, 1.0) or"
            r" int in the range [1, inf)"
        )
    ):
        check_n_samples(X=X, n_samples=n_samples, indices=indices)
