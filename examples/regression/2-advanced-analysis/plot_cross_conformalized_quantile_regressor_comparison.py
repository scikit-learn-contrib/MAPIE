"""
Comparative cross-conformalized quantile regression strategies
===============================================================


This example compares the different possibilities exposed by
``CrossConformalizedQuantileRegressor``:

- the interval construction methods: ``base``, ``plus`` and ``minmax``
- the point prediction aggregation strategies: ``mean``, ``median`` and
  ``pinball_weighted_mean``

Note that the valid aggregation strategies depend on the method: ``base``
predicts points with an estimator fitted on the entire training data, and so
only accepts ``aggregate_point_predictions=None``, while ``plus`` and ``minmax``
aggregate the cross-validation folds and require one of the strategies above.

The goal is to show how the same estimator can produce different point
predictions and prediction intervals depending on the chosen strategy.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)
from mapie.regression import CrossConformalizedQuantileRegressor

RANDOM_STATE = 1
CONFIDENCE_LEVEL = 0.8


def make_heteroscedastic_data(
    n_samples: int = 1000,
    random_state: int = RANDOM_STATE,
):
    """Generate a 1D regression problem with varying noise levels."""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-3.0, 3.0, size=(n_samples, 1))
    signal = 8.0 * np.sin(X[:, 0]) + 2.0 * X[:, 0]
    noise_scale = 1.5 + 1.5 * np.abs(X[:, 0])
    y = signal + rng.normal(loc=0.0, scale=noise_scale, size=n_samples)
    return X, y


def summarize_interval(y_true: np.ndarray, y_pis: np.ndarray) -> tuple[float, float]:
    """Compute coverage and mean width for a single confidence level."""
    coverage = regression_coverage_score(y_true, y_pis)[0]
    width = regression_mean_width_score(y_pis)[0]
    return float(coverage), float(width)


def plot_prediction_intervals(
    ax: plt.Axes,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pis: np.ndarray,
    title: str,
    subtitle: str,
) -> None:
    """Plot point predictions and prediction intervals on a sorted grid."""
    order = np.argsort(X_test[:, 0])
    x_plot = X_test[order, 0]
    y_test_plot = y_test[order]
    y_pred_plot = y_pred[order]
    y_low = y_pis[order, 0, 0]
    y_up = y_pis[order, 1, 0]

    ax.scatter(x_plot, y_test_plot, s=10, alpha=0.25, color="0.4", label="Test data")
    ax.plot(x_plot, y_pred_plot, color="C1", lw=2, label="Point prediction")
    ax.fill_between(
        x_plot,
        y_low,
        y_up,
        color="C1",
        alpha=0.2,
        label="Prediction interval",
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


##############################################################################
# 1. Data
# --------------------------------------------------------------------------
# We generate a 1D non-linear regression problem with heteroscedastic noise.

X, y = make_heteroscedastic_data()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)


##############################################################################
# 2. Base quantile estimator
# --------------------------------------------------------------------------
# The same base estimator is reused for all strategies so that the effect of
# the cross-conformalization method is isolated.

base_estimator = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.5,
    random_state=RANDOM_STATE,
)


##############################################################################
# 3. Comparison of the interval construction methods
# --------------------------------------------------------------------------
# We first compare the three interval construction methods supported by the
# cross-conformalized quantile regressor.

METHODS = {
    "base": "base",
    "plus": "plus",
    "minmax": "minmax",
}

method_results = {}
method_metrics = []

for method_name, method in METHODS.items():
    regressor = CrossConformalizedQuantileRegressor(
        estimator=base_estimator,
        confidence_level=CONFIDENCE_LEVEL,
        cv=5,
        method=method,
        random_state=RANDOM_STATE,
    )
    regressor.fit_conformalize(X_train, y_train)
    # `aggregate_point_predictions` is left to its "auto" default: the valid options
    # depend on the method, since ``base`` predicts points with an estimator fitted
    # on the whole training data while ``plus`` and ``minmax`` aggregate the folds.
    y_pred, y_pis = regressor.predict_interval(X_test)

    coverage, width = summarize_interval(y_test, y_pis)
    method_results[method_name] = (y_pred, y_pis)
    method_metrics.append(
        {
            "strategy": method_name,
            "aggregation": "auto",
            "coverage": coverage,
            "mean_width": width,
        }
    )


##############################################################################
# 4. Comparison of the point aggregation strategies
# --------------------------------------------------------------------------
# The aggregation strategy is used to combine the predictions of the regressors
# trained on each cross-validation fold, so it impacts the interval bounds as
# well as the point predictions. We keep the interval construction method fixed
# to ``plus`` and vary the point aggregation strategy.
#
# ``None`` is not part of the comparison: it predicts points with an estimator
# fitted on the entire data, which only the ``base`` method fits.

AGGREGATIONS = {
    "mean": "mean",
    "median": "median",
    "pinball_weighted_mean": "pinball_weighted_mean",
}

aggregation_regressor = CrossConformalizedQuantileRegressor(
    estimator=base_estimator,
    confidence_level=CONFIDENCE_LEVEL,
    cv=5,
    method="plus",
    random_state=RANDOM_STATE,
)
aggregation_regressor.fit_conformalize(X_train, y_train)

aggregation_results = {}
aggregation_metrics = []

for aggregation_name, aggregation in AGGREGATIONS.items():
    y_pred, y_pis = aggregation_regressor.predict_interval(
        X_test,
        aggregate_point_predictions=aggregation,
    )
    coverage, width = summarize_interval(y_test, y_pis)
    aggregation_results[aggregation_name] = (y_pred, y_pis)
    aggregation_metrics.append(
        {
            "strategy": "plus",
            "aggregation": aggregation_name,
            "coverage": coverage,
            "mean_width": width,
        }
    )


##############################################################################
# 5. Summary table
# --------------------------------------------------------------------------

results = pd.DataFrame(method_metrics + aggregation_metrics)
print(results.to_string(index=False))


##############################################################################
# 6. Plot the method comparison
# --------------------------------------------------------------------------

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
for ax, (method_name, (y_pred, y_pis)) in zip(axs, method_results.items()):
    coverage, width = summarize_interval(y_test, y_pis)
    plot_prediction_intervals(
        ax=ax,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_pis=y_pis,
        title=f"Method: {method_name}",
        subtitle=f"coverage={coverage:.3f} | width={width:.3f}",
    )

axs[0].legend(loc="upper left")
fig.suptitle(
    "CrossConformalizedQuantileRegressor: interval construction methods",
    y=1.02,
    fontsize=14,
)
fig.tight_layout()


##############################################################################
# 7. Plot the aggregation comparison
# --------------------------------------------------------------------------

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
for ax, (aggregation_name, (y_pred, y_pis)) in zip(axs, aggregation_results.items()):
    coverage, width = summarize_interval(y_test, y_pis)
    plot_prediction_intervals(
        ax=ax,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_pis=y_pis,
        title=f"Aggregation: {aggregation_name}",
        subtitle=f"coverage={coverage:.3f} | width={width:.3f}",
    )

axs[0].legend(loc="upper left")
fig.suptitle(
    "CrossConformalizedQuantileRegressor: point aggregation strategies",
    y=1.02,
    fontsize=14,
)
fig.tight_layout()
plt.show()
