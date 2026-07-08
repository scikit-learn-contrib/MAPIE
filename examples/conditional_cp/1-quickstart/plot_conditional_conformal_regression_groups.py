"""
Group-conditional prediction intervals
======================================


This example shows how to use
``ConditionalSplitConformalRegressor``
to build prediction intervals with conditional guarantees on pre-defined
groups.

It is a simple companion to the Gibbs, Cherian and Candès (2023) reproduction
example in the scientific-articles gallery. Here, the goal is not to reproduce a
paper figure, but to isolate the main idea on a small synthetic regression
problem: define ``feature_map`` as group indicators, then compare marginal and
group-conditional calibration.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor
from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split

##############################################################################
# 1. Generate grouped regression data
# --------------------------------------------------------------------------
#
# This one-dimensional regression problem follows the same shape as the advanced
# tutorial: the baseline is ``x * sin(x)``, and the noise gets harder as ``x``
# increases. We use bins of ``x`` as the conditional groups, so the same covariate
# drives both the prediction task and the difficulty level.

x_bins = np.array([-1.0, 0.0, 1.5, 3.0, 5.0])


def mean_function(x):
    return x * np.sin(x)


def generate_grouped_regression_data(n_samples=1600, random_state=42):
    rng = np.random.default_rng(random_state)
    x = rng.uniform(x_bins[0], x_bins[-1], size=n_samples)

    normal_scale = 0.8 * (np.maximum(x, 0) / x_bins[-1]) ** 2 * x_bins[-1]
    y = mean_function(x) + rng.normal(0, normal_scale)
    y += rng.uniform(-2.4, 2.4, size=n_samples) * (x < 0)

    X = x.reshape(-1, 1)
    return X, y


X, y = generate_grouped_regression_data()

(
    X_train,
    X_conformalize,
    X_test,
    y_train,
    y_conformalize,
    y_test,
) = train_conformalize_test_split(
    X,
    y,
    train_size=0.35,
    conformalize_size=0.45,
    test_size=0.20,
    random_state=42,
)


##############################################################################
# 2. Plot the data
# --------------------------------------------------------------------------
#
# The scatter plot shows that the vertical spread increases across ``x`` regions,
# which makes the right-hand side of the problem harder than the left-hand side.

group_indexes = np.digitize(X[:, 0], x_bins[1:-1], right=False)
bin_labels = [
    f"[{left:.2f}, {right:.2f})"
    for left, right in zip(x_bins[:-1], x_bins[1:])
]
bin_labels[-1] = f"[{x_bins[-2]:.2f}, {x_bins[-1]:.2f}]"

fig, ax = plt.subplots(figsize=(7, 4))
for group_index, label in enumerate(bin_labels):
    mask = group_indexes == group_index
    ax.scatter(
        X[mask, 0],
        y[mask],
        s=18,
        alpha=0.45,
        label=f"x in {label}",
    )

grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
ax.plot(grid, mean_function(grid), color="black", linewidth=2)
ax.set_xlabel("x")
ax.set_ylabel("Target")
ax.set_title("One-dimensional heteroscedastic regression data")
ax.legend()
plt.tight_layout()
plt.show()


##############################################################################
# 3. Define the conditional groups
# --------------------------------------------------------------------------
#
# ``feature_map`` returns one indicator column per ``x`` region. The conditional
# regressor uses these columns to calibrate score cutoffs that are valid on each
# group, not only on average over the full distribution.

bin_centers = (x_bins[:-1] + x_bins[1:]) / 2


def indicator_matrix(values, bin_edges):
    values = np.asarray(values).reshape(-1)
    bin_indexes = np.digitize(values, bin_edges[1:-1], right=False)
    matrix = np.zeros((len(values), len(bin_edges) - 1))
    matrix[np.arange(len(values)), bin_indexes] = 1
    return matrix


def phi_fn(X):
    return indicator_matrix(np.asarray(X).reshape(-1), x_bins)


##############################################################################
# 4. Fit marginal and conditional conformal regressors
# --------------------------------------------------------------------------
#
# Both methods use the same fitted polynomial regressor and the same
# conformalization data. The marginal regressor uses one residual cutoff for all
# samples, while ``ConditionalSplitConformalRegressor`` receives ``feature_map`` and
# calibrates the cutoff by ``x`` group.

confidence_level = 0.90
estimator = make_pipeline(PolynomialFeatures(degree=4), LinearRegression()).fit(
    X_train, y_train
)

mapie_marginal = SplitConformalRegressor(
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_marginal.conformalize(X_conformalize, y_conformalize)
_, y_interval_marginal = mapie_marginal.predict_interval(X_test)

mapie_conditional = ConditionalSplitConformalRegressor(
    phi_fn,
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_conditional.conformalize(X_conformalize, y_conformalize)
_, y_interval_conditional = mapie_conditional.predict_interval(X_test)


##############################################################################
# 5. Evaluate and visualize the correction
# --------------------------------------------------------------------------
#
# Marginal split conformal prediction has constant interval width with the
# absolute residual score. This overcovers the low-noise group and undercovers
# the high-noise groups. Conditional conformal prediction makes an additional
# group-level correction, narrowing intervals where the problem is easy and
# widening them where the problem is hard.


def group_mask(X, bin_index):
    if bin_index == len(x_bins) - 2:
        return (X[:, 0] >= x_bins[bin_index]) & (
            X[:, 0] <= x_bins[-1]
        )
    return (X[:, 0] >= x_bins[bin_index]) & (
        X[:, 0] < x_bins[bin_index + 1]
    )


def scores_by_group(y_true, intervals, X):
    coverages = []
    widths = []
    for bin_index in range(len(x_bins) - 1):
        mask = group_mask(X, bin_index)
        coverages.append(regression_coverage_score(y_true[mask], intervals[mask]))
        widths.append(regression_mean_width_score(intervals[mask]))
    return np.asarray(coverages).ravel(), np.asarray(widths).ravel()


coverage_marginal_by_group, width_marginal_by_group = scores_by_group(
    y_test, y_interval_marginal, X_test
)
coverage_conditional_by_group, width_conditional_by_group = scores_by_group(
    y_test, y_interval_conditional, X_test
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
group_positions = np.arange(len(bin_labels))
bar_width = 0.35

axes[0].bar(
    group_positions - bar_width / 2,
    coverage_marginal_by_group,
    width=bar_width,
    label="Marginal",
)
axes[0].bar(
    group_positions + bar_width / 2,
    coverage_conditional_by_group,
    width=bar_width,
    label="Conditional",
)
axes[0].axhline(confidence_level, color="black", linestyle="--", linewidth=1)
axes[0].set_ylim(0.60, 1.02)
axes[0].set_ylabel("Coverage")
axes[0].set_title("Coverage by x group")
axes[0].legend()

axes[1].bar(
    group_positions - bar_width / 2,
    width_marginal_by_group,
    width=bar_width,
    label="Marginal",
)
axes[1].bar(
    group_positions + bar_width / 2,
    width_conditional_by_group,
    width=bar_width,
    label="Conditional",
)
axes[1].set_ylim(0.0, 8.5)
axes[1].set_ylabel("Mean interval width")
axes[1].set_title("Interval width by x group")

for ax in axes:
    ax.set_xlabel("x group")
    ax.set_xticks(group_positions)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")

plt.tight_layout()
plt.show()
