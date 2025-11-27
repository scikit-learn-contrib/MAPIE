from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut, ShuffleSplit
from sklearn.pipeline import Pipeline

from mapie.regression.quantile_regression import _MapieQuantileRegressor
from mapie.utils import (
    NotFittedError,
    _cast_point_predictions_to_ndarray,
    _cast_predictions_to_ndarray_tuple,
    _check_alpha,
    _check_alpha_and_n_samples,
    _check_array_inf,
    _check_array_nan,
    _check_arrays_length,
    _check_binary_zero_one,
    _check_cv,
    _check_cv_not_string,
    _check_gamma,
    _check_if_param_in_allowed_values,
    _check_lower_upper_bounds,
    _check_n_features_in,
    _check_n_jobs,
    _check_n_samples,
    _check_no_agg_cv,
    _check_null_weight,
    _check_number_bins,
    _check_split_strategy,
    _check_verbose,
    _compute_quantiles,
    _fit_estimator,
    _get_binning_groups,
    _prepare_fit_params_and_sample_weight,
    _prepare_params,
    _raise_error_if_fit_called_in_prefit_mode,
    _raise_error_if_method_already_called,
    _raise_error_if_previous_method_not_called,
    _transform_confidence_level_to_alpha,
    _transform_confidence_level_to_alpha_list,
    check_is_fitted,
    check_sklearn_user_model_is_fitted,
    train_conformalize_test_split,
)


@pytest.fixture(scope="module")
def dataset():
    X, y = make_regression(
        n_samples=100, n_features=2, noise=1.0, random_state=random_state
    )
    return X, y


class TestTrainConformalizeTestSplit:
    def test_error_sum_int_is_not_dataset_size(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError):
            train_conformalize_test_split(
                X,
                y,
                train_size=1,
                conformalize_size=1,
                test_size=1,
                random_state=random_state,
            )

    def test_error_sum_float_is_not_1(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError):
            train_conformalize_test_split(
                X,
                y,
                train_size=0.5,
                conformalize_size=0.5,
                test_size=0.5,
                random_state=random_state,
            )

    def test_error_sizes_are_int_and_float(self, dataset):
        X, y = dataset
        with pytest.raises(TypeError):
            train_conformalize_test_split(
                X,
                y,
                train_size=5,
                conformalize_size=0.5,
                test_size=0.5,
                random_state=random_state,
            )

    def test_3_floats(self, dataset):
        X, y = dataset
        (X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
            train_conformalize_test_split(
                X,
                y,
                train_size=0.6,
                conformalize_size=0.2,
                test_size=0.2,
                random_state=random_state,
            )
        )
        assert len(X_train) == 60
        assert len(X_conformalize) == 20
        assert len(X_test) == 20

    def test_3_ints(self, dataset):
        X, y = dataset
        (X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
            train_conformalize_test_split(
                X,
                y,
                train_size=60,
                conformalize_size=20,
                test_size=20,
                random_state=random_state,
            )
        )
        assert len(X_train) == 60
        assert len(X_conformalize) == 20
        assert len(X_test) == 20

    def test_random_state(self, dataset):
        X, y = dataset
        (
            X_train_1,
            X_conformalize_1,
            X_test_1,
            y_train_1,
            y_conformalize_1,
            y_test_1,
        ) = train_conformalize_test_split(
            X,
            y,
            train_size=60,
            conformalize_size=20,
            test_size=20,
            random_state=random_state,
        )
        (
            X_train_2,
            X_conformalize_2,
            X_test_2,
            y_train_2,
            y_conformalize_2,
            y_test_2,
        ) = train_conformalize_test_split(
            X,
            y,
            train_size=60,
            conformalize_size=20,
            test_size=20,
            random_state=random_state,
        )
        assert np.array_equal(X_train_1, X_train_2)
        assert np.array_equal(X_conformalize_1, X_conformalize_2)
        assert np.array_equal(X_test_1, X_test_2)
        assert np.array_equal(y_train_1, y_train_2)
        assert np.array_equal(y_conformalize_1, y_conformalize_2)
        assert np.array_equal(y_test_1, y_test_2)

    def test_different_random_state(self, dataset):
        X, y = dataset
        (
            X_train_1,
            X_conformalize_1,
            X_test_1,
            y_train_1,
            y_conformalize_1,
            y_test_1,
        ) = train_conformalize_test_split(
            X,
            y,
            train_size=60,
            conformalize_size=20,
            test_size=20,
            random_state=random_state,
        )
        (
            X_train_2,
            X_conformalize_2,
            X_test_2,
            y_train_2,
            y_conformalize_2,
            y_test_2,
        ) = train_conformalize_test_split(
            X,
            y,
            train_size=60,
            conformalize_size=20,
            test_size=20,
            random_state=random_state + 1,
        )
        assert not np.array_equal(X_train_1, X_train_2)
        assert not np.array_equal(X_conformalize_1, X_conformalize_2)
        assert not np.array_equal(X_test_1, X_test_2)
        assert not np.array_equal(y_train_1, y_train_2)
        assert not np.array_equal(y_conformalize_1, y_conformalize_2)
        assert not np.array_equal(y_test_1, y_test_2)

    def test_shuffle_false(self, dataset):
        X, y = dataset
        (X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
            train_conformalize_test_split(
                X,
                y,
                train_size=60,
                conformalize_size=20,
                test_size=20,
                random_state=random_state,
                shuffle=False,
            )
        )
        assert np.array_equal(np.concatenate((y_train, y_conformalize, y_test)), y)


@pytest.fixture
def point_predictions():
    return np.array([1, 2, 3])


@pytest.fixture
def point_and_interval_predictions():
    return np.array([1, 2]), np.array([3, 4])


@pytest.mark.parametrize(
    "confidence_level, expected",
    [
        (0.9, 0.1),
        (0.7, 0.3),
        (0.999, 0.001),
    ],
)
def test_transform_confidence_level_to_alpha(confidence_level, expected):
    result = _transform_confidence_level_to_alpha(confidence_level)
    assert result == expected
    assert str(result) == str(expected)  # Ensure clean representation


class TestTransformConfidenceLevelToAlphaList:
    def test_non_list_iterable(self):
        confidence_level = (0.8, 0.7)  # Testing a non-list iterable
        assert _transform_confidence_level_to_alpha_list(confidence_level) == [0.2, 0.3]

    def test_transform_confidence_level_to_alpha_is_called(self):
        with patch(
            "mapie.utils._transform_confidence_level_to_alpha"
        ) as mock_transform_confidence_level_to_alpha:
            _transform_confidence_level_to_alpha_list([0.2, 0.3])
            mock_transform_confidence_level_to_alpha.assert_called()


class TestCheckIfParamInAllowedValues:
    def test_error(self):
        with pytest.raises(ValueError):
            _check_if_param_in_allowed_values("invalid_option", "", ["valid_option"])

    def test_ok(self):
        assert _check_if_param_in_allowed_values("valid", "", ["valid"]) is None


def test_check_cv_not_string():
    with pytest.raises(ValueError):
        _check_cv_not_string("string")


class TestCastPointPredictionsToNdarray:
    def test_error(self, point_and_interval_predictions):
        with pytest.raises(TypeError):
            _cast_point_predictions_to_ndarray(point_and_interval_predictions)

    def test_valid_ndarray(self, point_predictions):
        point_predictions = np.array([1, 2, 3])
        result = _cast_point_predictions_to_ndarray(point_predictions)
        assert result is point_predictions
        assert isinstance(result, np.ndarray)


class TestCastPredictionsToNdarrayTuple:
    def test_error(self, point_predictions):
        with pytest.raises(TypeError):
            _cast_predictions_to_ndarray_tuple(point_predictions)

    def test_valid_ndarray(self, point_and_interval_predictions):
        result = _cast_predictions_to_ndarray_tuple(point_and_interval_predictions)
        assert result is point_and_interval_predictions
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)


@pytest.mark.parametrize(
    "params, expected", [(None, {}), ({"a": 1, "b": 2}, {"a": 1, "b": 2})]
)
def test_prepare_params(params, expected):
    assert _prepare_params(params) == expected
    assert _prepare_params(params) is not params


class TestPrepareFitParamsAndSampleWeight:
    def test_uses_prepare_params(self):
        with patch("mapie.utils._prepare_params") as mock_prepare_params:
            _prepare_fit_params_and_sample_weight({"param1": 1})
            mock_prepare_params.assert_called()

    def test_with_sample_weight(self):
        fit_params = {"sample_weight": [0.1, 0.2, 0.3]}
        assert _prepare_fit_params_and_sample_weight(fit_params) == (
            {},
            [0.1, 0.2, 0.3],
        )

    def test_without_sample_weight(self):
        params = {"param1": 1}
        assert _prepare_fit_params_and_sample_weight(params) == (params, None)


class TestRaiseErrorIfPreviousMethodNotCalled:
    def test_raises_error_when_previous_method_not_called(self):
        with pytest.raises(ValueError):
            _raise_error_if_previous_method_not_called(
                "current_method", "previous_method", False
            )

    def test_does_nothing_when_previous_method_called(self):
        assert (
            _raise_error_if_previous_method_not_called(
                "current_method", "previous_method", True
            )
            is None
        )


class TestRaiseErrorIfMethodAlreadyCalled:
    def test_raises_error_when_method_already_called(self):
        with pytest.raises(ValueError):
            _raise_error_if_method_already_called("method", True)

    def test_does_nothing_when_method_not_called(self):
        assert _raise_error_if_method_already_called("method", False) is None


class TestRaiseErrorIfFitCalledInPrefitMode:
    def test_raises_error_in_prefit_mode(self):
        with pytest.raises(ValueError):
            _raise_error_if_fit_called_in_prefit_mode(True)

    def test_does_nothing_when_not_in_prefit_mode(self):
        assert _raise_error_if_fit_called_in_prefit_mode(False) is None


X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])

n_features = 10

X, y = make_regression(n_samples=500, n_features=n_features, noise=1.0, random_state=1)
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
    "quantile": [
        0.03075388,
        0.17261836,
        0.33281326,
        0.43939618,
        0.54867626,
        0.64881987,
        0.73440899,
        0.77793816,
        0.89000413,
        0.99610621,
    ],
    "uniform": [
        0,
        0.11111111,
        0.22222222,
        0.33333333,
        0.44444444,
        0.55555556,
        0.66666667,
        0.77777778,
        0.88888889,
        1,
    ],
    "array split": [
        0.62689056,
        0.74743526,
        0.87642114,
        0.88321124,
        0.8916548,
        0.94083846,
        0.94999075,
        0.98759822,
        0.99610621,
        np.inf,
    ],
}


class DumbEstimator:
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> DumbEstimator:
        self.fitted_ = True
        return self


def test_check_null_weight_with_none() -> None:
    """Test that the function has no effect if sample weight is None."""
    sw_out, X_out, y_out = _check_null_weight(None, X_toy, y_toy)
    assert sw_out is None
    np.testing.assert_almost_equal(np.array(X_out), X_toy)
    np.testing.assert_almost_equal(np.array(y_out), y_toy)


def test_check_null_weight_with_nonzeros() -> None:
    """Test that the function has no effect if sample weight is never zero."""
    sample_weight = np.ones_like(y_toy)
    sw_out, X_out, y_out = _check_null_weight(sample_weight, X_toy, y_toy)
    np.testing.assert_almost_equal(np.array(sw_out), sample_weight)
    np.testing.assert_almost_equal(np.array(X_out), X_toy)
    np.testing.assert_almost_equal(np.array(y_out), y_toy)


def test_check_null_weight_with_zeros() -> None:
    """Test that the function reduces the shape if there are zeros."""
    sample_weight = np.ones_like(y_toy)
    sample_weight[:1] = 0.0
    sw_out, X_out, y_out = _check_null_weight(sample_weight, X_toy, y_toy)
    np.testing.assert_almost_equal(np.array(sw_out), np.array([1, 1, 1, 1, 1]))
    np.testing.assert_almost_equal(np.array(X_out), np.array([[1], [2], [3], [4], [5]]))
    np.testing.assert_almost_equal(np.array(y_out), np.array([7, 9, 11, 13, 15]))


@pytest.mark.filterwarnings(
    "ignore:Estimator exposes fitted-like attributes.*:UserWarning"
)
@pytest.mark.parametrize("estimator", [LinearRegression(), DumbEstimator()])
@pytest.mark.parametrize("sample_weight", [None, np.ones_like(y_toy)])
def test_fit_estimator(estimator: Any, sample_weight: Optional[NDArray]) -> None:
    """Test that the returned estimator is always fitted."""
    estimator = _fit_estimator(estimator, X_toy, y_toy, sample_weight)
    check_sklearn_user_model_is_fitted(estimator)


def test_fit_estimator_sample_weight() -> None:
    """Test that sample weight is taken into account if possible."""
    X = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([21, 7, 9, 11, 13, 15])
    sample_weight = np.array([5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    estimator_1 = _fit_estimator(LinearRegression(), X, y)
    estimator_2 = _fit_estimator(LinearRegression(), X, y, sample_weight)
    y_pred_1 = estimator_1.predict(X)
    y_pred_2 = estimator_2.predict(X)
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(y_pred_1, y_pred_2)


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2, 2.5, "a", ["a", "b"]])
def test_invalid_alpha(alpha: Any) -> None:
    """Test that invalid alphas raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid confidence_level.*"):
        _check_alpha(alpha=alpha)


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
    _check_alpha(alpha=alpha)


@pytest.mark.parametrize("cv", ["prefit", None])
def test_check_n_features_in_estimator_without_attribute(cv: Any) -> None:
    """
    Test that estimators fitted with a right number of features
    but missing an n_features_in_ attribute raise no errors.
    """
    estimator = DumbEstimator()
    n_features_in = _check_n_features_in(X=X, cv=cv, estimator=estimator)
    assert n_features_in == n_features


def test_check_n_features_in_X_1d_array() -> None:
    """
    Test that _check_n_features_in returns 1 for a one-dimensional array.
    """
    X_1d = np.array([1, 2, 3, 4, 5])
    n_features_in = _check_n_features_in(X=X_1d)
    assert n_features_in == 1


def test_check_n_features_in_X_without_shape() -> None:
    """
    Test that _check_n_features_in handles objects without shape attribute.
    """
    n_features_in = _check_n_features_in(X=[[1, 2], [3, 4], [5, 6]])
    assert n_features_in == 2


def test_check_n_features_in_invalid_prefit_estimator() -> None:
    """
    Test that estimators fitted with a wrong number of features raise errors.
    """
    estimator = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match=r".*mismatch between.*"):
        _check_n_features_in(X_toy, cv="prefit", estimator=estimator)


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
    _check_alpha_and_n_samples(alpha, n)


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
        _check_alpha_and_n_samples(alpha, n)


@pytest.mark.parametrize("n_jobs", ["dummy", 0, 1.5, [1, 2]])
def test_invalid_n_jobs(n_jobs: Any) -> None:
    """Test that invalid n_jobs raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid n_jobs argument.*"):
        _check_n_jobs(n_jobs)


@pytest.mark.parametrize("n_jobs", [-5, -1, 1, 4])
def test_valid_n_jobs(n_jobs: Any) -> None:
    """Test that valid n_jobs raise no errors."""
    _check_n_jobs(n_jobs)


@pytest.mark.parametrize("verbose", ["dummy", -1, 1.5, [1, 2]])
def test_invalid_verbose(verbose: Any) -> None:
    """Test that invalid verboses raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid verbose argument.*"):
        _check_verbose(verbose)


@pytest.mark.parametrize("verbose", [0, 10, 50])
def test_valid_verbose(verbose: Any) -> None:
    """Test that valid verboses raise no errors."""
    _check_verbose(verbose)


def test_initial_low_high_pred(caplog) -> None:
    """Test lower/upper predictions of the quantiles regression crossing"""
    y_preds = np.array([[4, 5, 2], [4, 4, 4], [2, 3, 4]])
    with caplog.at_level(logging.INFO):
        _check_lower_upper_bounds(y_preds[0], y_preds[1], y_preds[2])
    assert "The predictions are ill-sorted" in caplog.text


def test_final_low_high_pred(caplog) -> None:
    """Test lower/upper predictions crossing"""
    y_preds = np.array([[4, 3, 2], [3, 3, 3], [2, 3, 4]])
    y_pred_low = np.array([4, 7, 2])
    y_pred_up = np.array([3, 3, 3])
    with caplog.at_level(logging.INFO):
        _check_lower_upper_bounds(y_pred_low, y_pred_up, y_preds[2])
    assert "The predictions are ill-sorted" in caplog.text


def test_ensemble_in_predict() -> None:
    """Checking for ensemble defined in predict of CQR"""
    mapie_reg = _MapieQuantileRegressor()
    mapie_reg.fit(X, y)
    with pytest.warns(UserWarning, match=r"WARNING: Alpha should not be spec.*"):
        mapie_reg.predict(X, alpha=0.2)


def test_alpha_in_predict() -> None:
    """Checking for alpha defined in predict of CQR"""
    mapie_reg = _MapieQuantileRegressor()
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
        _compute_quantiles(vector, alphas)


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
    quantiles = _compute_quantiles(vector, alphas)

    assert len(quantiles) == len(alphas)


@pytest.mark.parametrize("alphas", ALPHAS)
def test_compute_quantiles_3D_shape(alphas: NDArray):
    """Test that the number of quantiles is equal to
    the number of alphas for a 3D input vector
    """
    vector = np.random.rand(1000, 1)
    vector = np.repeat(vector, len(alphas), axis=1)
    quantiles = _compute_quantiles(vector, alphas)

    assert len(quantiles) == len(alphas)


@pytest.mark.parametrize("alphas", ALPHAS)
def test_compute_quantiles_2D_and_3D(alphas: NDArray):
    """Test that if to matrices are equal (modulo one dimension)
    then there quantiles are the same.
    """
    vector1 = np.random.rand(1000, 1)
    vector2 = np.repeat(vector1, len(alphas), axis=1)

    quantiles1 = _compute_quantiles(vector1, alphas)
    quantiles2 = _compute_quantiles(vector2, alphas)

    assert (quantiles1 == quantiles2).all()


@pytest.mark.parametrize("estimator", [-1, 3, 0.2])
def test_quantile_prefit_non_iterable(estimator: Any) -> None:
    """
    Test that there is a list of estimators provided when cv='prefit'
    is called for _MapieQuantileRegressor.
    """
    with pytest.raises(
        ValueError,
        match=r".*Estimator for prefit must be an iterable object.*",
    ):
        mapie_reg = _MapieQuantileRegressor(estimator=estimator, cv="prefit")
        mapie_reg.fit([1, 2, 3], [4, 5, 6])


@pytest.mark.parametrize("strategy", ["quantile", "uniform", "array split"])
def test_binning_group_strategies(strategy: str) -> None:
    """Test that different strategies have the correct outputs."""
    bins_ = _get_binning_groups(y_score, num_bins=10, strategy=strategy)
    np.testing.assert_allclose(results_binning[strategy], bins_, rtol=1e-05)


def test_wrong_split_strategy() -> None:
    """Test for wrong split strategies."""
    with pytest.raises(ValueError, match=r"Please provide a valid*"):
        _check_split_strategy(strategy="not_valid")


def test_split_strategy_None() -> None:
    """Test what occurs if None is provided as split strategy."""
    strategy = _check_split_strategy(None)
    assert strategy == "uniform"


@pytest.mark.parametrize("bins", ["random", LinearRegression(), 0.5])
def test_num_bins_not_int(bins: int) -> None:
    """Test input for bins is an integer."""
    with pytest.raises(ValueError, match=r"Please provide a bin number as an int*"):
        _check_number_bins(num_bins=bins)


def test_num_bins_below_zero() -> None:
    """Test input for bins is positive integer."""
    with pytest.raises(ValueError, match=r"Please provide a bin number greater*"):
        _check_number_bins(num_bins=-1)


def test_binary_target() -> None:
    """
    Test that input of binary will provide an error message for non binary.
    """
    with pytest.raises(ValueError, match=r"Please provide y_true as a bina*"):
        _check_binary_zero_one(np.array([0, 5, 4]))


def test_nan_values() -> None:
    """
    Test if array has only non-numerical values like NaNs
    """
    with pytest.raises(ValueError, match=r"Array contains only NaN*"):
        _check_array_nan(np.array([np.nan, np.nan, np.nan, np.nan]))


def test_inf_values() -> None:
    """
    Test if array has infinite values like +inf or -inf
    """
    with pytest.raises(ValueError, match=r"Array contains infinite va*"):
        _check_array_inf(np.array([1, 2, -np.inf, 4]))


def test_length() -> None:
    """
    Test if the arrays have the same size (length)
    """
    with pytest.raises(ValueError, match=r"There are arrays with different len*"):
        _check_arrays_length(np.array([1, 2, 3]), np.array([4, 5, 6, 7]))


def test_change_values_zero_one() -> None:
    """Test that binary output are changed to zero one outputs."""
    array_ = _check_binary_zero_one(np.array([0, 4, 4]))
    assert (np.unique(array_) == np.array([0, 1])).all()


@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.9])
def test_valid_gamma(gamma: float) -> None:
    """Test a valid gamma parameter."""
    _check_gamma(gamma)


@pytest.mark.parametrize("gamma", [1.5, -0.1])
def test_invalid_large_gamma(gamma: float) -> None:
    """Test a non-valid gamma parameter."""
    with pytest.raises(
        ValueError, match="Invalid gamma. Allowed values are between 0 and 1."
    ):
        _check_gamma(gamma)


@pytest.mark.parametrize("cv", [5, "split"])
def test_check_cv_same_split_with_random_state(cv: BaseCrossValidator) -> None:
    """Test that cv generate same split with fixed random_state."""
    cv = _check_cv(cv, random_state=random_state)

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
    cv = _check_cv(cv, random_state=None)

    train_indices_1, train_indices_2 = [], []
    for train_index, _ in cv.split(X):
        train_indices_1.append(train_index)
    for train_index, _ in cv.split(X):
        train_indices_2.append(train_index)

    for i in range(cv.get_n_splits()):
        np.testing.assert_allclose(train_indices_1[i], train_indices_2[i])


@pytest.mark.parametrize(
    "cv_result",
    [
        (1, True),
        (2, False),
        ("split", True),
        (KFold(5), False),
        (ShuffleSplit(1), True),
        (ShuffleSplit(2), False),
        (LeaveOneOut(), False),
    ],
)
def test_check_no_agg_cv(cv_result: Tuple) -> None:
    """Test that if `_check_no_agg_cv` function returns the expected result."""
    array = ["prefit", "split"]
    cv, result = cv_result
    np.testing.assert_almost_equal(_check_no_agg_cv(X_toy, cv, array), result)


@pytest.mark.parametrize("cv", [object()])
def test_check_no_agg_cv_value_error(cv: Any) -> None:
    """Test that if `_check_no_agg_cv` function raises value error."""
    array = ["prefit", "split"]
    with pytest.raises(
        ValueError, match=r"Allowed values must have the `get_n_splits` method"
    ):
        _check_no_agg_cv(X_toy, cv, array)


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
        ),
    ):
        _check_n_samples(X=X, n_samples=n_samples, indices=indices)


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
        ),
    ):
        _check_n_samples(X=X, n_samples=n_samples, indices=indices)


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
        ),
    ):
        _check_n_samples(X=X, n_samples=n_samples, indices=indices)


class DummyModel:
    pass


def test_check_is_fitted_raises_before_fit():
    model = DummyModel()
    with pytest.raises(NotFittedError) as excinfo:
        check_is_fitted(model)
    assert "DummyModel is not fitted yet" in str(excinfo.value)


def test_check_is_fitted_passes_after_fit():
    model = DummyModel()
    model.is_fitted = True
    check_is_fitted(model)


def test_check_user_model_is_fitted_unfitted():
    model = DummyModel()
    with pytest.warns(UserWarning, match=r".*Estimator does not appear fitted.*"):
        check_sklearn_user_model_is_fitted(model)


def test_check_user_model_is_fitted_raises_for_unfitted_model():
    model = LinearRegression()
    with pytest.warns(UserWarning, match=r".*Estimator does not appear fitted.*"):
        check_sklearn_user_model_is_fitted(model)


@pytest.mark.parametrize(
    "Model",
    [
        LinearRegression(),
        LogisticRegression(),
        Pipeline([("LinearRegression", LinearRegression())]),
    ],
)
def test_check_user_model_is_fitted_sklearn_models(Model):
    """Check that sklearn classifiers and regressors pass."""
    X = np.random.randn(20, 4)
    y = (
        (np.random.randn(20) > 0).astype(int)
        if isinstance(Model, LogisticRegression)
        else np.random.randn(20)
    )
    model = Model.fit(X, y)
    assert check_sklearn_user_model_is_fitted(model) is True


class BrokenPredictModel:
    """Model with n_features_in_ but predict always fails"""

    n_features_in_ = 3

    def predict(self, X):
        raise RuntimeError("Predict failure")


def test_check_user_model_is_fitted_predict_fails():
    model = BrokenPredictModel()
    with pytest.raises(
        NotFittedError, match=r"Estimator has `n_features_in_` but failed"
    ):
        check_sklearn_user_model_is_fitted(model)
