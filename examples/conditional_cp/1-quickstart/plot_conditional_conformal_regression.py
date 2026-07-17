"""
Group-conditional prediction intervals (regression)
===================================================

This example shows how to use ``ConditionalSplitConformalRegressor``
to build prediction intervals with conditional guarantees on pre-defined groups.

The key idea is to provide a basis function ``feature_map`` that
identifies the covariate groups on which coverage should be controlled.
"""

# mkdocs_gallery_thumbnail_number = 2

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
# 1. Generate grouped regression data and fit the model
# --------------------------------------------------------------------------
#
# This one-dimensional regression problem follows the same shape as the advanced
# tutorial: the baseline is ``x * sin(x)``, and the noise gets harder as ``x``
# increases. We use bins of ``x`` as the conditional groups, so the same covariate
# drives both the prediction task and the difficulty level. The data is split into
# train / conformalize / test sets, and a polynomial regressor is fitted on the
# training set.

x_bins = np.array([-1.0, 0.0, 1.5, 3.0, 5.0])


def mean_function(x):
    return x * np.sin(x)


def generate_grouped_regression_data(n_samples=2500, random_state=42):
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
    train_size=0.4,
    conformalize_size=0.3,
    test_size=0.3,
    random_state=42,
)

estimator = make_pipeline(PolynomialFeatures(degree=4), LinearRegression()).fit(
    X_train, y_train
)


##############################################################################
# 2. Visualize the fitted model
# --------------------------------------------------------------------------
#
# Overlaying the predictions of the fitted polynomial regressor on the test data
# gives the curve that the conformal regressors will wrap with intervals.

bin_labels = [
    f"[{left:.2f}, {right:.2f})" for left, right in zip(x_bins[:-1], x_bins[1:])
]
bin_labels[-1] = f"[{x_bins[-2]:.2f}, {x_bins[-1]:.2f}]"

grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
X_grid = grid.reshape(-1, 1)
y_pred_grid = estimator.predict(X_grid)

group_indexes_test = np.digitize(X_test[:, 0], x_bins[1:-1], right=False)

fig, ax = plt.subplots(figsize=(7, 4))
for group_index, label in enumerate(bin_labels):
    mask = group_indexes_test == group_index
    ax.scatter(
        X_test[mask, 0],
        y_test[mask],
        s=18,
        alpha=0.45,
        label=f"y for x in {label}",
    )
ax.plot(grid, y_pred_grid, color="black", linewidth=2, label="Model prediction")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Fitted polynomial regressor and test data")
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


def feature_map(X):
    x = np.asarray(X).reshape(-1)
    bin_indexes = np.digitize(x, x_bins[1:-1], right=False)
    matrix = np.zeros((len(x), len(x_bins) - 1))
    matrix[np.arange(len(x)), bin_indexes] = 1
    return matrix


##############################################################################
# 4. Fit standard and conditional conformal regressors
# --------------------------------------------------------------------------
#
# Both methods use the same fitted polynomial regressor and the same
# conformalization data. The standard regressor uses one residual cutoff for all
# samples, while ``ConditionalSplitConformalRegressor`` receives ``feature_map`` and
# calibrates the cutoff by ``x`` group.

confidence_level = 0.90

mapie_standard = SplitConformalRegressor(
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_standard.conformalize(X_conformalize, y_conformalize)
_, y_interval_standard = mapie_standard.predict_interval(X_test)

mapie_conditional = ConditionalSplitConformalRegressor(
    feature_map,
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_conditional.conformalize(X_conformalize, y_conformalize)
_, y_interval_conditional = mapie_conditional.predict_interval(X_test)


##############################################################################
# 5. Visualize the prediction intervals
# --------------------------------------------------------------------------
#
# Predicting intervals on a fine grid of ``x`` makes the correction visible.
# The standard regressor produces a single constant-width band everywhere,
# which is too wide in the low-noise region and too narrow in the high-noise
# region. The conditional regressor adapts the band to each ``x`` group:
# narrower where the data is tight and wider where the noise grows.

_, y_interval_standard_grid = mapie_standard.predict_interval(X_grid)
_, y_interval_conditional_grid = mapie_conditional.predict_interval(X_grid)

fig, ax = plt.subplots(figsize=(7, 4))
for group_index, label in enumerate(bin_labels):
    mask = group_indexes_test == group_index
    ax.scatter(
        X_test[mask, 0],
        y_test[mask],
        s=18,
        alpha=0.2,
        label=f"y for x in {label}",
    )
ax.plot(grid, y_pred_grid, color="black", linewidth=2, label="Model prediction")
for intervals, color, label in (
    (y_interval_standard_grid, "tab:blue", "Standard interval"),
    (y_interval_conditional_grid, "tab:orange", "Conditional interval"),
):
    ax.fill_between(
        grid,
        intervals[:, 0, 0],
        intervals[:, 1, 0],
        color=color,
        alpha=0.25,
    )
    ax.plot(grid, intervals[:, 0, 0], color=color, linewidth=1.5, label=label)
    ax.plot(grid, intervals[:, 1, 0], color=color, linewidth=1.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Standard vs conditional prediction intervals")
ax.legend()
plt.tight_layout()
plt.show()


##############################################################################
# 6. Evaluate the correction by group
# --------------------------------------------------------------------------
#
# The bar charts below summarize the effect on coverage and interval width.
# The first bar group reports the overall score over the full test set, then
# the remaining bar groups show the same metrics inside each ``x`` group.


def group_mask(X, bin_index):
    if bin_index == len(x_bins) - 2:
        return (X[:, 0] >= x_bins[bin_index]) & (X[:, 0] <= x_bins[-1])
    return (X[:, 0] >= x_bins[bin_index]) & (X[:, 0] < x_bins[bin_index + 1])


def scores_by_group(y_true, intervals, X):
    coverages = []
    widths = []
    for bin_index in range(len(x_bins) - 1):
        mask = group_mask(X, bin_index)
        coverages.append(regression_coverage_score(y_true[mask], intervals[mask]))
        widths.append(regression_mean_width_score(intervals[mask]))
    return np.asarray(coverages).ravel(), np.asarray(widths).ravel()


coverage_standard_by_group, width_standard_by_group = scores_by_group(
    y_test, y_interval_standard, X_test
)
coverage_conditional_by_group, width_conditional_by_group = scores_by_group(
    y_test, y_interval_conditional, X_test
)

coverage_standard = regression_coverage_score(y_test, y_interval_standard)
coverage_conditional = regression_coverage_score(y_test, y_interval_conditional)
width_standard = regression_mean_width_score(y_interval_standard)
width_conditional = regression_mean_width_score(y_interval_conditional)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
score_labels = ["All", *bin_labels]
group_positions = np.arange(len(score_labels))
bar_width = 0.35

axes[0].bar(
    group_positions - bar_width / 2,
    np.r_[coverage_standard, coverage_standard_by_group],
    width=bar_width,
    label="Standard",
)
axes[0].bar(
    group_positions + bar_width / 2,
    np.r_[coverage_conditional, coverage_conditional_by_group],
    width=bar_width,
    label="Conditional",
)
axes[0].axhline(confidence_level, color="black", linestyle="--", linewidth=1)
axes[0].set_ylim(0.60, 1.02)
axes[0].set_ylabel("Coverage")
axes[0].set_title("Coverage overall and by x group")
axes[0].legend()

axes[1].bar(
    group_positions - bar_width / 2,
    np.r_[width_standard, width_standard_by_group],
    width=bar_width,
    label="Standard",
)
axes[1].bar(
    group_positions + bar_width / 2,
    np.r_[width_conditional, width_conditional_by_group],
    width=bar_width,
    label="Conditional",
)
axes[1].set_ylim(0.0, 8.5)
axes[1].set_ylabel("Mean interval width")
axes[1].set_title("Interval width overall and by x group")

for ax in axes:
    ax.set_xlabel("x group")
    ax.set_xticks(group_positions)
    ax.set_xticklabels(score_labels, rotation=30, ha="right")

plt.tight_layout()
plt.show()


##############################################################################
# 7. Go further
# --------------------------------------------------------------------------
#
# Explore the advanced examples to learn how to build richer feature maps.
