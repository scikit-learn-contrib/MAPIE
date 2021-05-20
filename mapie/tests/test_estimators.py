from typing import Any, Union, Optional
from typing_extensions import TypedDict

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from mapie.estimators import MapieRegressor
from mapie.metrics import coverage_score


X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=1.0, random_state=1)

METHODS = ["naive", "base", "plus", "minmax"]

Params = TypedDict("Params", {"method": str, "cv": Optional[Union[int, KFold]]})
STRATEGIES = {
    "naive": Params(method="naive", cv=None),
    "jackknife": Params(method="base", cv=-1),
    "jackknife_plus": Params(method="plus", cv=-1),
    "jackknife_minmax": Params(method="minmax", cv=-1),
    "cv": Params(method="base", cv=KFold(n_splits=3, shuffle=True, random_state=1)),
    "cv_plus": Params(method="plus", cv=KFold(n_splits=3, shuffle=True, random_state=1)),
    "cv_minmax": Params(method="minmax", cv=KFold(n_splits=3, shuffle=True, random_state=1)),
}

EXPECTED_WIDTHS = {
    "naive": 3.81,
    "jackknife": 3.89,
    "jackknife_plus": 3.90,
    "jackknife_minmax": 3.96,
    "cv": 3.85,
    "cv_plus": 3.90,
    "cv_minmax": 4.04
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
    "check_fit_score_takes_y"
}


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieRegressor()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie = MapieRegressor()
    assert mapie.estimator is None
    assert mapie.alpha == 0.1
    assert mapie.method == "plus"
    assert mapie.cv is None
    assert not mapie.ensemble
    assert mapie.verbose == 0
    assert mapie.n_jobs is None


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


@pytest.mark.parametrize("estimator", [0, "estimator", KFold(), ["a", "b"]])
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


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    mapie = MapieRegressor(estimator=DummyRegressor(), **STRATEGIES[strategy])
    mapie.fit(X_toy, y_toy)
    assert isinstance(mapie.single_estimator_, DummyRegressor)
    for estimator in mapie.estimators_:
        assert isinstance(estimator, DummyRegressor)


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2, 2.5, "a", [[0.5]], ["a", "b"]])
def test_invalid_alpha(alpha: int) -> None:
    """Test that invalid alphas raise errors."""
    mapie = MapieRegressor(alpha=alpha)
    mapie.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        mapie.predict(X_toy)


@pytest.mark.parametrize(
    "alpha",
    [np.linspace(0.05, 0.95, 5), [0.05, 0.95], (0.05, 0.95), np.array([0.05, 0.95])]
)
def test_valid_alpha(alpha: int) -> None:
    """Test that valid alphas raise no errors."""
    mapie = MapieRegressor(alpha=alpha)
    mapie.fit(X_toy, y_toy)
    mapie.predict(X_toy)


@pytest.mark.parametrize("method", [0, 1, "jackknife", "cv", ["base", "plus"]])
def test_invalid_method(method: str) -> None:
    """Test that invalid methods raise errors."""
    mapie = MapieRegressor(method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", METHODS)
def test_valid_method(method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie = MapieRegressor(method=method)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("n_jobs", ["dummy", 0, 1.5, [1, 2]])
def test_invalid_n_jobs(n_jobs: Any) -> None:
    """Test that invalid n_jobs raise errors."""
    mapie = MapieRegressor(n_jobs=n_jobs)
    with pytest.raises(ValueError, match=r".*Invalid n_jobs argument.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("n_jobs", [-5, -1, 1, 4])
def test_valid_n_jobs(n_jobs: Any) -> None:
    """Test that valid n_jobs raise no errors."""
    mapie = MapieRegressor(n_jobs=n_jobs)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("verbose", ["dummy", -1, 1.5, [1, 2]])
def test_invalid_verbose(verbose: Any) -> None:
    """Test that invalid verboses raise errors."""
    mapie = MapieRegressor(verbose=verbose)
    with pytest.raises(ValueError, match=r".*Invalid verbose argument.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("verbose", [0, 10, 50])
def test_valid_verbose(verbose: Any) -> None:
    """Test that valid verboses raise no errors."""
    mapie = MapieRegressor(verbose=verbose)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("ensemble", ["dummy", 0, 1, 2.5, [1, 2]])
def test_invalid_ensemble(ensemble: Any) -> None:
    """Test that invalid ensembles raise errors."""
    mapie = MapieRegressor(ensemble=ensemble)
    with pytest.raises(ValueError, match=r".*Invalid ensemble.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("ensemble", [True, False])
def test_valid_ensemble(ensemble: bool) -> None:
    """Test that valid ensembles raise no errors."""
    mapie = MapieRegressor(ensemble=ensemble)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [-3.14, -2, 0, 1, "cv", DummyRegressor(), [1, 2]])
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie = MapieRegressor(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [100, 200, 300])
def test_too_large_cv(cv: Any) -> None:
    """Test that too large cv raise sklearn errors."""
    mapie = MapieRegressor(cv=cv)
    with pytest.raises(ValueError, match=rf".*Cannot have number of splits n_splits={cv} greater.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [None, -1, 2, KFold(), LeaveOneOut()])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = MapieRegressor(cv=cv)
    mapie.fit(X_toy, y_toy)


@parametrize_with_checks([MapieRegressor()])  # type: ignore
def test_sklearn_compatible_estimator(estimator: Any, check: Any) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    if check.func.__name__ not in SKLEARN_EXCLUDED_CHECKS:
        check(estimator)


@pytest.mark.parametrize("method", METHODS)
def test_fit_attributes(method: str) -> None:
    """Test fit attributes shared by all PI methods."""
    mapie = MapieRegressor(method=method)
    mapie.fit(X_toy, y_toy)
    assert hasattr(mapie, 'n_features_in_')
    assert hasattr(mapie, 'single_estimator_')
    assert hasattr(mapie, 'estimators_')
    assert hasattr(mapie, 'residuals_')
    assert hasattr(mapie, 'k_')


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X_reg, y_reg), (X_toy, y_toy)])
@pytest.mark.parametrize("alpha", [0.1, [0.1, 0.2], (0.1, 0.2)])
def test_predict_output_shape(strategy: str, alpha: Any, dataset: Any) -> None:
    """Test predict output shape."""
    mapie = MapieRegressor(alpha=alpha, **STRATEGIES[strategy])
    X, y = dataset
    mapie.fit(X, y)
    y_preds = mapie.predict(X)
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_preds.shape == (X.shape[0], 3, n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("ensemble", [True, False])
def test_prediction_between_low_up(strategy: str, ensemble: bool) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    mapie = MapieRegressor(ensemble=ensemble, **STRATEGIES[strategy])
    mapie.fit(X_reg, y_reg)
    y_pred, y_pred_low, y_pred_up = mapie.predict(X_reg)[:, :, 0].T
    assert (y_pred >= y_pred_low).all()
    assert (y_pred <= y_pred_up).all()


@pytest.mark.parametrize("method", ["plus", "minmax"])
@pytest.mark.parametrize("cv", [-1, 2, 3, 5])
def test_prediction_ensemble(method: str, cv: Union[LeaveOneOut, KFold]) -> None:
    """Test that predictions differs when ensemble is True/False, but not prediction intervals."""
    mapie = MapieRegressor(method=method, cv=cv, ensemble=True)
    mapie.fit(X_reg, y_reg)
    y_preds_ensemble = mapie.predict(X_reg)[:, :, 0]
    mapie.ensemble = False
    y_preds_no_ensemble = mapie.predict(X_reg)[:, :, 0]
    np.testing.assert_almost_equal(y_preds_ensemble[:, 1], y_preds_no_ensemble[:, 1])
    np.testing.assert_almost_equal(y_preds_ensemble[:, 2], y_preds_no_ensemble[:, 2])
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(y_preds_ensemble[:, 0], y_preds_no_ensemble[:, 0])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("ensemble", [True, False])
def test_linear_data_confidence_interval(strategy: str, ensemble: bool) -> None:
    """
    Test that MapieRegressor applied on a linear regression model
    fitted on a linear curve results in null uncertainty.
    """
    mapie = MapieRegressor(ensemble=ensemble, **STRATEGIES[strategy])
    mapie.fit(X_toy, y_toy)
    y_pred, y_pred_low, y_pred_up = mapie.predict(X_toy)[:, :, 0].T
    np.testing.assert_almost_equal(y_pred_up, y_pred_low)
    np.testing.assert_almost_equal(y_pred, y_pred_low)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """Test expected PIs for a multivariate linear regression problem with fixed random state."""
    mapie = MapieRegressor(alpha=0.05, **STRATEGIES[strategy])
    mapie.fit(X_reg, y_reg)
    _, y_pred_low, y_pred_up = mapie.predict(X_reg)[:, :, 0].T
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = coverage_score(y_reg, y_pred_low, y_pred_up)
    np.testing.assert_almost_equal(width_mean, EXPECTED_WIDTHS[strategy], 2)
    np.testing.assert_almost_equal(coverage, EXPECTED_COVERAGES[strategy], 2)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """Test that predictions and intervals are similar with two equal values of alpha."""
    mapie = MapieRegressor(alpha=[0.1, 0.1], **STRATEGIES[strategy])
    mapie.fit(X_reg, y_reg)
    y_preds = mapie.predict(X_reg)
    np.testing.assert_almost_equal(y_preds[:, 0, 0], y_preds[:, 0, 1])
    np.testing.assert_almost_equal(y_preds[:, 1, 0], y_preds[:, 1, 1])
    np.testing.assert_almost_equal(y_preds[:, 2, 0], y_preds[:, 2, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_ordered_alpha(strategy: str) -> None:
    """Test that prediction intervals lower (upper) bounds give consistent results for ordered alphas."""
    mapie = MapieRegressor(alpha=[0.05, 0.1], **STRATEGIES[strategy])
    mapie.fit(X_reg, y_reg)
    y_preds = mapie.predict(X_reg)
    assert (y_preds[:, 1, 0] <= y_preds[:, 1, 1]).all()
    assert (y_preds[:, 2, 0] >= y_preds[:, 2, 1]).all()


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)])
def test_results_for_alpha_as_float_and_arraylike(strategy: str, alpha: Any) -> None:
    """Test that output values do not depend on type of alpha."""
    mapie_float1 = MapieRegressor(alpha=alpha[0], **STRATEGIES[strategy])
    y_preds_float1 = mapie_float1.fit(X_reg, y_reg).predict(X_reg)
    mapie_float2 = MapieRegressor(alpha=alpha[1], **STRATEGIES[strategy])
    y_preds_float2 = mapie_float2.fit(X_reg, y_reg).predict(X_reg)
    mapie_array = MapieRegressor(alpha=alpha, **STRATEGIES[strategy])
    y_preds_array = mapie_array.fit(X_reg, y_reg).predict(X_reg)
    np.testing.assert_almost_equal(y_preds_float1[:, :, 0], y_preds_array[:, :, 0])
    np.testing.assert_almost_equal(y_preds_float2[:, :, 0], y_preds_array[:, :, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieRegressor gives equal predictions regardless of number of parallel jobs.
    """
    mapie_single = MapieRegressor(n_jobs=1, **STRATEGIES[strategy])
    mapie_single.fit(X_toy, y_toy)
    mapie_multi = MapieRegressor(n_jobs=-1, **STRATEGIES[strategy])
    mapie_multi.fit(X_toy, y_toy)
    y_preds_single = mapie_single.predict(X_toy)
    y_preds_multi = mapie_multi.predict(X_toy)
    np.testing.assert_almost_equal(y_preds_single, y_preds_multi)
