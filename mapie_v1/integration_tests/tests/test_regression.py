from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mapie_v1.regression import (
    SplitConformalRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    ConformalizedQuantileRegressor
)
from mapiev0.regression import MapieRegressor as MapieRegressorV0 # noqa
from mapiev0.regression import MapieQuantileRegressor as MapieQuantileRegressorV0 # noqa
from mapie_v1.conformity_scores.utils import select_conformity_score
from mapie_v1.utils import (filter_params,
                            calculate_coverage,
                            train_test_split_shuffle)
from sklearn.model_selection import KFold

RANDOM_STATE = 1
K_FOLDS = 3
N_BOOTSTRAPS = 30

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
X, y = make_regression(n_samples=500,
                       n_features=10,
                       noise=1.0,
                       random_state=RANDOM_STATE)


@pytest.mark.parametrize("strategy_key", ["split"])
@pytest.mark.parametrize("method", ["base", "plus", "minmax"])
@pytest.mark.parametrize("conformity_score", ["absolute"])
@pytest.mark.parametrize("confidence_level", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
@pytest.mark.parametrize("minimize_interval_width", [True, False])
@pytest.mark.parametrize("allow_infinite_bounds", [True, False])
@pytest.mark.parametrize(
    "estimator", [
        LinearRegression(),
        RandomForestRegressor(random_state=RANDOM_STATE, max_depth=2)])
@pytest.mark.parametrize("test_size", [0.2, 0.5])
def test_exact_interval_equality(
    strategy_key,
    method,
    conformity_score,
    confidence_level,
    agg_function,
    minimize_interval_width,
    allow_infinite_bounds,
    estimator,
    test_size
):
    """
    Test that the prediction intervals are exactly the same
    between v0 and v1 models when using the same settings.
    """

    v0_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": select_conformity_score(conformity_score),
        "alpha": 1 - confidence_level,
        "agg_function": agg_function,
        "random_state": RANDOM_STATE,
        "test_size": test_size,
        "optimize_beta": minimize_interval_width,
        "allow_infinite_bounds": allow_infinite_bounds
    }
    v1_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": conformity_score,
        "confidence_level": confidence_level,
        "aggregate_function": agg_function,
        "random_state": RANDOM_STATE,
        "n_bootstraps": N_BOOTSTRAPS,
        "minimize_interval_width": minimize_interval_width,
        "allow_infinite_bounds": allow_infinite_bounds
    }

    v0, v1 = initialize_models(
        strategy_key=strategy_key,
        v0_params=v0_params,
        v1_params=v1_params,
        k_folds=K_FOLDS,
        random_state=RANDOM_STATE
    )

    X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    v0.fit(X, y)
    v1.fit(X_train, y_train)

    v1.conformalize(X_conf, y_conf)

    v0_predict_params = filter_params(v0.predict, v0_params)
    v1_predict_params = filter_params(v1.predict, v1_params)
    _, v0_pred_intervals = v0.predict(X_conf, **v0_predict_params)
    v1_pred_intervals = v1.predict_set(X_conf, **v1_predict_params)
    v0_pred_intervals = v0_pred_intervals[:, :, 0]

    np.testing.assert_array_equal(
        v1_pred_intervals,
        v0_pred_intervals,
        err_msg="Prediction intervals differ between v0 and v1 models"
    )


@pytest.mark.parametrize("strategy_key", ["split"])
@pytest.mark.parametrize("method", ["base", "plus", "minmax"])
@pytest.mark.parametrize("conformity_score", ["absolute"])
@pytest.mark.parametrize("confidence_level", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
@pytest.mark.parametrize("estimator", [LinearRegression(),
                                       RandomForestRegressor(max_depth=2)])
@pytest.mark.parametrize("test_size", [0.2, 0.5])
def test_consistent_coverage(
    strategy_key,
    method,
    conformity_score,
    confidence_level,
    agg_function,
    estimator,
    test_size,
):

    v0_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": select_conformity_score(conformity_score),
        "alpha": 1-confidence_level,
        "agg_function": agg_function,
        "random_state": RANDOM_STATE,
        "test_size": test_size,
    }
    v1_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": conformity_score,
        "confidence_level": confidence_level,
        "aggregate_function": agg_function,
        "random_state": RANDOM_STATE,
        "n_bootstraps": N_BOOTSTRAPS,
    }

    v0, v1 = initialize_models(
        strategy_key=strategy_key,
        v0_params=v0_params,
        v1_params=v0_params,
        random_state=RANDOM_STATE,
        k_folds=K_FOLDS
    )

    X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
        X, y, test_size=test_size, random_state=RANDOM_STATE)

    v0.fit(X, y)
    v1.fit(X_train, y_train)

    v1.conformalize(X_conf, y_conf)

    v0_predict_params = filter_params(v0.predict, v0_params)
    v1_predict_params = filter_params(v1.predict, v1_params)
    _, v0_pred_intervals = v0.predict(X_conf, **v0_predict_params)
    v1_pred_intervals = v1.predict_set(X_conf, **v1_predict_params)
    v0_pred_intervals = v0_pred_intervals[:, :, 0]

    v0_coverage = calculate_coverage(y_conf, v0_pred_intervals)
    v1_coverage = calculate_coverage(y_conf, v1_pred_intervals)

    err_msg = f"Coverage mismatch: v0 {v0_coverage}, v1 {v1_coverage}"
    assert_almost_equal(v0_coverage,
                        v1_coverage,
                        decimal=1,
                        err_msg=err_msg)


@pytest.mark.parametrize("conformity_score", ["absolute"])
@pytest.mark.parametrize("confidence_level", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("estimator", [LinearRegression(),
                                       RandomForestRegressor(max_depth=2)])
@pytest.mark.parametrize("test_size", [0.2, 0.5])
def test_consistent_coverage_for_prefit_model(
    conformity_score,
    confidence_level,
    estimator,
    test_size,
):
    """
    Test consistent coverage calculation when the model is already fit.
    This ensures that conformal prediction methods provide intervals
    without re-fitting the model, yielding the expected coverage.
    """

    v0_params = {
        "estimator": estimator,
        "conformity_score": select_conformity_score(conformity_score),
        "alpha": 1 - confidence_level,
        "random_state": RANDOM_STATE,
        "test_size": test_size,
    }
    v1_params = {
        "estimator": estimator,
        "conformity_score": conformity_score,
        "confidence_level": confidence_level,
        "random_state": RANDOM_STATE,
        "n_bootstraps": N_BOOTSTRAPS,
    }

    v0, v1 = initialize_models(
        strategy_key="prefit",
        v0_params=v0_params,
        v1_params=v1_params,
        k_folds=K_FOLDS,
        random_state=RANDOM_STATE
    )

    X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
        X, y, test_size=test_size, random_state=RANDOM_STATE)

    estimator.fit(X_train, y_train)

    v0.fit(X_conf, y_conf)

    # Calibration
    v1.conformalize(X_conf, y_conf)

    v0_predict_params = filter_params(v0.predict, v0_params)
    v1_predict_params = filter_params(v1.predict, v1_params)
    _, v0_pred_intervals = v0.predict(X_conf, **v0_predict_params)
    v1_pred_intervals = v1.predict_set(X_conf, **v1_predict_params)
    v0_pred_intervals = v0_pred_intervals[:, :, 0]

    v0_coverage = calculate_coverage(y_conf, v0_pred_intervals)
    v1_coverage = calculate_coverage(y_conf, v1_pred_intervals)

    err_msg = f"Coverage mismatch: v0 {v0_coverage}, v1 {v1_coverage}"
    assert_almost_equal(v0_coverage,
                        v1_coverage, decimal=2,
                        err_msg=err_msg)


@pytest.mark.parametrize("strategy_key", ["split"])
@pytest.mark.parametrize("method", ["base", "plus", "minmax"])
@pytest.mark.parametrize("conformity_score", ["absolute"])
@pytest.mark.parametrize("confidence_level", [0.9, 0.95, 0.99])
@pytest.mark.parametrize("agg_function", ["mean", "median"])
@pytest.mark.parametrize("estimator", [
        LinearRegression(),
        RandomForestRegressor(random_state=RANDOM_STATE, max_depth=2)])
@pytest.mark.parametrize("test_size", [0.2, 0.5])
def test_consistent_interval_width(
    strategy_key,
    method,
    conformity_score,
    confidence_level,
    agg_function,
    estimator,
    test_size,
):
    """
    Test that the interval width for v0 and v1 models are consistent.
    Ensures that the prediction intervals generated by both models
    have similar widths.
    """

    v0_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": select_conformity_score(conformity_score),
        "alpha": 1 - confidence_level,
        "agg_function": agg_function,
        "random_state": RANDOM_STATE,
        "test_size": test_size,
    }
    v1_params = {
        "estimator": estimator,
        "method": method,
        "conformity_score": conformity_score,
        "confidence_level": confidence_level,
        "aggregate_function": agg_function,
        "random_state": RANDOM_STATE,
        "n_bootstraps": N_BOOTSTRAPS,
    }

    v0, v1 = initialize_models(
        strategy_key=strategy_key,
        v0_params=v0_params,
        v1_params=v1_params,
        k_folds=K_FOLDS,
        random_state=RANDOM_STATE
    )

    X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    v0.fit(X, y)
    v1.fit(X_train, y_train)

    v1.conformalize(X_conf, y_conf)

    v0_predict_params = filter_params(v0.predict, v0_params)
    v1_predict_params = filter_params(v1.predict, v1_params)
    _, v0_pred_intervals = v0.predict(X_conf, **v0_predict_params)
    v1_pred_intervals = v1.predict_set(X_conf, **v1_predict_params)
    v0_pred_intervals = v0_pred_intervals[:, :, 0]

    v0_interval_widths = v0_pred_intervals[:, 1] - v0_pred_intervals[:, 0]
    v1_interval_widths = v1_pred_intervals[:, 1] - v1_pred_intervals[:, 0]

    v0_avg_width = np.mean(v0_interval_widths)
    v1_avg_width = np.mean(v1_interval_widths)
    mean_interval = (v0_avg_width + v1_avg_width) / 2
    normalized_difference = abs(v0_avg_width - v1_avg_width) / mean_interval

    tolerance = 0.06
    err_msg = (
        f"Normalized interval width difference too high: "
        f"v0 avg width {v0_avg_width}, v1 avg width {v1_avg_width}, "
        f"normalized difference {normalized_difference}"
    )
    assert normalized_difference < tolerance, err_msg


def test_dummy():
    test_size = 0.5
    alpha = 0.5
    confidence_level = 1 - alpha
    random_state = 42

    v0 = MapieRegressorV0(
        cv="split", test_size=test_size, random_state=random_state
    )
    v0.fit(X_toy, y_toy)
    v0_preds = v0.predict(X_toy)
    _, v0_pred_intervals = v0.predict(X_toy, alpha=alpha)
    v0_pred_intervals = v0_pred_intervals[:, :, 0]

    X_train, X_conf, y_train, y_conf = train_test_split_shuffle(
        X_toy, y_toy, test_size=test_size, random_state=random_state
    )
    v1 = SplitConformalRegressor(
        confidence_level=confidence_level, random_state=random_state
    )
    v1.fit(X_train, y_train)
    v1.conformalize(X_conf, y_conf)
    v1_preds = v1.predict(X_toy)
    v1_pred_intervals = v1.predict_set(X_toy)

    np.testing.assert_array_equal(v1_preds, v0_preds)
    np.testing.assert_array_equal(v1_pred_intervals, v0_pred_intervals)


def initialize_models(
    strategy_key,
    v0_params: dict,
    v1_params: dict,
    k_folds=5,
    random_state=42
):

    if strategy_key == "prefit":
        v0_params.update({"cv": "prefit"})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(SplitConformalRegressor.__init__, v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = SplitConformalRegressor(prefit=True, **v1_params)

    elif strategy_key == "split":
        v0_params.update({"cv": "split"})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(SplitConformalRegressor.__init__, v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = SplitConformalRegressor(**v1_params)

    elif strategy_key == "cv":
        v0_params.update({"cv": KFold(n_splits=k_folds,
                                      shuffle=True,
                                      random_state=random_state)})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(CrossConformalRegressor.__init__, v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = CrossConformalRegressor(cv=k_folds, **v1_params)

    elif strategy_key == "jackknife":
        v0_params.update({"cv": -1})
        v0_params = filter_params(MapieRegressorV0.__init__, v0_params)
        v1_params = filter_params(JackknifeAfterBootstrapRegressor.__init__,
                                  v1_params)
        v0 = MapieRegressorV0(**v0_params)
        v1 = JackknifeAfterBootstrapRegressor(**v1_params)

    elif strategy_key == "CQR":
        v0_params = filter_params(MapieQuantileRegressorV0.__init__, v0_params)
        v1_params = filter_params(SplitConformalRegressor.__init__, v1_params)
        v0 = MapieQuantileRegressorV0(**v0_params)
        v1 = ConformalizedQuantileRegressor(**v1_params)

    else:
        raise ValueError(f"Unknown strategy key: {strategy_key}")

    return v0, v1
