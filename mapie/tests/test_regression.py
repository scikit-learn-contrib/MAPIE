from __future__ import annotations

from itertools import combinations
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import TypedDict

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.utils.validation import check_is_fitted

from mapie.aggregation_functions import aggregate_all
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample
from mapie._typing import ArrayLike

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X, y = make_regression(n_samples=500, n_features=10, noise=1.0, random_state=1)
k = np.ones(shape=(5, X.shape[1]))
METHODS = ["naive", "base", "plus", "minmax"]

Params = TypedDict(
    "Params",
    {
        "method": str,
        "agg_function": str,
        "cv": Optional[Union[int, KFold, Subsample]],
    },
)
STRATEGIES = {
    "naive": Params(method="naive", agg_function="median", cv=None),
    "jackknife": Params(method="base", agg_function="mean", cv=-1),
    "jackknife_plus": Params(method="plus", agg_function="mean", cv=-1),
    "jackknife_minmax": Params(method="minmax", agg_function="mean", cv=-1),
    "cv": Params(
        method="base",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=1),
    ),
    "cv_plus": Params(
        method="plus",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=1),
    ),
    "cv_minmax": Params(
        method="minmax",
        agg_function="mean",
        cv=KFold(n_splits=3, shuffle=True, random_state=1),
    ),
    "jackknife_plus_ab": Params(
        method="plus",
        agg_function="mean",
        cv=Subsample(n_resamplings=30, random_state=1),
    ),
    "jackknife_minmax_ab": Params(
        method="minmax",
        agg_function="mean",
        cv=Subsample(n_resamplings=30, random_state=1),
    ),
    "jackknife_plus_median_ab": Params(
        method="plus",
        agg_function="median",
        cv=Subsample(
            n_resamplings=30,
            random_state=1,
        ),
    ),
}

WIDTHS = {
    "naive": 3.81,
    "jackknife": 3.89,
    "jackknife_plus": 3.90,
    "jackknife_minmax": 3.96,
    "cv": 3.85,
    "cv_plus": 3.90,
    "cv_minmax": 4.04,
    "prefit": 4.81,
    "cv_plus_median": 3.90,
    "jackknife_plus_ab": 3.90,
    "jackknife_minmax_ab": 4.13,
    "jackknife_plus_median_ab": 3.87,
}

COVERAGES = {
    "naive": 0.952,
    "jackknife": 0.952,
    "jackknife_plus": 0.952,
    "jackknife_minmax": 0.952,
    "cv": 0.958,
    "cv_plus": 0.956,
    "cv_minmax": 0.966,
    "prefit": 0.980,
    "cv_plus_median": 0.954,
    "jackknife_plus_ab": 0.952,
    "jackknife_minmax_ab": 0.970,
    "jackknife_plus_median_ab": 0.960,
}


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = MapieRegressor()
    assert mapie_reg.agg_function is None
    assert mapie_reg.method == "plus"
    assert mapie_reg.cv is None


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    mapie_reg = MapieRegressor(
        estimator=DummyRegressor(), **STRATEGIES[strategy]
    )
    mapie_reg.fit(X_toy, y_toy)
    assert isinstance(mapie_reg.single_estimator_, DummyRegressor)
    for estimator in mapie_reg.estimators_:
        assert isinstance(estimator, DummyRegressor)


@pytest.mark.parametrize("method", METHODS)
def test_valid_method(method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_reg = MapieRegressor(method=method)
    mapie_reg.fit(X_toy, y_toy)
    check_is_fitted(mapie_reg, mapie_reg.fit_attributes)


@pytest.mark.parametrize("agg_function", ["dummy", 0, 1, 2.5, [1, 2]])
def test_invalid_agg_function(agg_function: Any) -> None:
    """Test that invalid agg_functions raise errors."""
    mapie_reg = MapieRegressor(agg_function=agg_function)
    with pytest.raises(ValueError, match=r".*Invalid aggregation function.*"):
        mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("agg_function", [None, "mean", "median"])
def test_valid_agg_function(agg_function: str) -> None:
    """Test that valid agg_functions raise no errors."""
    mapie_reg = MapieRegressor(agg_function=agg_function)
    mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [None, -1, 2, KFold(), LeaveOneOut()])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = MapieRegressor(cv=cv)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [100, 200, 300])
def test_too_large_cv(cv: Any) -> None:
    """Test that too large cv raise sklearn errors."""
    mapie_reg = MapieRegressor(cv=cv)
    with pytest.raises(
        ValueError,
        match=rf".*Cannot have number of splits n_splits={cv} greater.*",
    ):
        mapie_reg.fit(X_toy, y_toy)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.4], (0.2, 0.4)])
def test_predict_output_shape(
    strategy: str, alpha: Any, dataset: Tuple[ArrayLike, ArrayLike]
) -> None:
    """Test predict output shape."""
    mapie_reg = MapieRegressor(**STRATEGIES[strategy])
    (X, y) = dataset
    mapie_reg.fit(X, y)
    y_pred, y_pis = mapie_reg.predict(X, alpha=alpha)
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], 2, n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_reg = MapieRegressor(**STRATEGIES[strategy])
    mapie_reg.fit(X, y)
    _, y_pis = mapie_reg.predict(X, alpha=[0.1, 0.1])
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 0, 1])
    np.testing.assert_allclose(y_pis[:, 1, 0], y_pis[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    mapie_reg = MapieRegressor(**STRATEGIES[strategy])
    mapie_reg.fit(X, y)
    y_pred_float1, y_pis_float1 = mapie_reg.predict(X, alpha=alpha[0])
    y_pred_float2, y_pis_float2 = mapie_reg.predict(X, alpha=alpha[1])
    y_pred_array, y_pis_array = mapie_reg.predict(X, alpha=alpha)
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_pis_float1[:, :, 0], y_pis_array[:, :, 0])
    np.testing.assert_allclose(y_pis_float2[:, :, 0], y_pis_array[:, :, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_ordered_alpha(strategy: str) -> None:
    """
    Test that prediction intervals lower (upper) bounds give
    consistent results for ordered alphas.
    """
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=[0.05, 0.1])
    assert (y_pis[:, 0, 0] <= y_pis[:, 0, 1]).all()
    assert (y_pis[:, 1, 0] >= y_pis[:, 1, 1]).all()


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    mapie_single = MapieRegressor(n_jobs=1, **STRATEGIES[strategy])
    mapie_multi = MapieRegressor(n_jobs=-1, **STRATEGIES[strategy])
    mapie_single.fit(X_toy, y_toy)
    mapie_multi.fit(X_toy, y_toy)
    y_pred_single, y_pis_single = mapie_single.predict(X_toy, alpha=0.2)
    y_pred_multi, y_pis_multi = mapie_multi.predict(X_toy, alpha=0.2)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_pis_single, y_pis_multi)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(strategy: str) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = MapieRegressor(**STRATEGIES[strategy])
    mapie1 = MapieRegressor(**STRATEGIES[strategy])
    mapie2 = MapieRegressor(**STRATEGIES[strategy])
    mapie0.fit(X, y, sample_weight=None)
    mapie1.fit(X, y, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 5)
    y_pred0, y_pis0 = mapie0.predict(X, alpha=0.05)
    y_pred1, y_pis1 = mapie1.predict(X, alpha=0.05)
    y_pred2, y_pis2 = mapie2.predict(X, alpha=0.05)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_prediction_between_low_up(strategy: str) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=0.1)
    assert (y_pred >= y_pis[:, 0, 0]).all()
    assert (y_pred <= y_pis[:, 1, 0]).all()


@pytest.mark.parametrize("method", ["plus", "minmax"])
@pytest.mark.parametrize("cv", [-1, 2, 3, 5])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_prediction_agg_function(
    method: str, cv: Union[LeaveOneOut, KFold], alpha: int
) -> None:
    """
    Test that predictions differ when agg_function is None/"mean",
    but not prediction intervals.
    """
    mapie = MapieRegressor(method=method, cv=cv, agg_function="median")
    mapie.fit(X, y)
    y_pred_1, y_pis_1 = mapie.predict(X, alpha=alpha)
    mapie.agg_function = None
    y_pred_2, y_pis_2 = mapie.predict(X, alpha=alpha)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pred_1, y_pred_2)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_data_confidence_interval(strategy: str) -> None:
    """
    Test that MapieRegressor applied on a linear regression model
    fitted on a linear curve results in null uncertainty.
    """
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X_toy, y_toy)
    y_pred, y_pis = mapie.predict(X_toy, alpha=0.2)
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 1, 0])
    np.testing.assert_allclose(y_pred, y_pis[:, 0, 0])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a multivariate linear regression problem
    with fixed random state.
    """
    mapie = MapieRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    _, y_pis = mapie.predict(X, alpha=0.05)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)


def test_results_prefit_ignore_method() -> None:
    """Test that method is ignored when ``cv="prefit"``."""
    estimator = LinearRegression().fit(X, y)
    all_y_pis: List[ArrayLike] = []
    for method in METHODS:
        mapie_reg = MapieRegressor(
            estimator=estimator, cv="prefit", method=method
        )
        mapie_reg.fit(X, y)
        _, y_pis = mapie_reg.predict(X, alpha=0.1)
        all_y_pis.append(y_pis)
    for y_pis1, y_pis2 in combinations(all_y_pis, 2):
        np.testing.assert_allclose(y_pis1, y_pis2)


def test_results_prefit_naive() -> None:
    """
    Test that prefit, fit and predict on the same dataset
    is equivalent to the "naive" method.
    """
    estimator = LinearRegression().fit(X, y)
    mapie_reg = MapieRegressor(estimator=estimator, cv="prefit")
    mapie_reg.fit(X, y)
    _, y_pis = mapie_reg.predict(X, alpha=0.05)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(y, y_pis[:, 0, 0], y_pis[:, 1, 0])
    np.testing.assert_allclose(width_mean, WIDTHS["naive"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["naive"], rtol=1e-2)


def test_results_prefit() -> None:
    """Test prefit results on a standard train/validation/test split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=1 / 10, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=1 / 9, random_state=1
    )
    estimator = LinearRegression().fit(X_train, y_train)
    mapie_reg = MapieRegressor(estimator=estimator, cv="prefit")
    mapie_reg.fit(X_val, y_val)
    _, y_pis = mapie_reg.predict(X_test, alpha=0.05)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(
        y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]
    )
    np.testing.assert_allclose(width_mean, WIDTHS["prefit"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["prefit"], rtol=1e-2)


def test_not_enough_resamplings() -> None:
    """Test that a warning is raised if at least one residual is nan."""
    with pytest.warns(UserWarning, match=r"WARNING: at least one point of*"):
        mapie_reg = MapieRegressor(
            cv=Subsample(n_resamplings=1), agg_function="mean"
        )
        mapie_reg.fit(X, y)


def test_no_agg_fx_specified_with_subsample() -> None:
    """Test that a warning is raised if at least one residual is nan."""
    with pytest.warns(Warning, match=r"WARNING: you need to specify*"):
        mapie_reg = MapieRegressor(
            cv=Subsample(n_resamplings=1), agg_function=None
        )
        mapie_reg.fit(X, y)


def test_invalid_aggregate_all() -> None:
    """
    Test that wrong aggregation in MAPIE raise errors.
    """
    with pytest.raises(
        ValueError,
        match=r".*Aggregation function called but not defined.*",
    ):
        aggregate_all(None, X)


def test_invalid_aggregate_mask() -> None:
    """
    Test that wrong aggregation in MAPIE raise errors.
    """
    with pytest.raises(
        ValueError,
        match=r".*Aggregation function called but not defined.*",
    ):
        mapie_reg = MapieRegressor()
        mapie_reg.aggregate_with_mask(X, k)


def test_pred_loof_isnan() -> None:
    """Test that if validation set is empty then prediction is empty."""
    mapie_reg = MapieRegressor()
    _, y_pred, _, _ = mapie_reg._fit_and_predict_oof_model(
        estimator=LinearRegression(),
        X=X_toy,
        y=y_toy,
        train_index=[0, 1, 2, 3, 4],
        val_index=[],
        k=0,
    )
    assert len(y_pred) == 0
