"""
Tutorial: conditional conformal prediction for regression
=========================================================


This tutorial shows how to use
:class:`~mapie.conditional_conformal_prediction.ConditionalSplitConformalRegressor`.
The method follows Gibbs, Cherian and Candès (2023) [1] and builds prediction
intervals with guarantees over a user-chosen finite class of covariate shifts.

In MAPIE, this finite class is passed as ``feature_map``. The examples below
compare standard split conformal prediction with two conditional feature maps:

- polynomial features of ``X``;
- smooth radial basis functions over the one-dimensional input.

[1] Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès,
"Conformal Prediction With Conditional Guarantees",
`arXiv <https://arxiv.org/abs/2305.12616>`_, 2023.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
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
# 1. Generate heteroscedastic regression data
# --------------------------------------------------------------------------
#
# The noise distribution changes with ``X``: it is almost uniform on the left
# side and increasingly Gaussian on the right side. A single marginal cutoff
# cannot adapt to this local variation.

rng = np.random.default_rng(42)
confidence_level = 0.9


def mean_function(x):
    return x * np.sin(x)


def generate_data(n_samples=1800):
    X = rng.uniform(-1, 5, size=n_samples)
    gaussian_scale = 0.8 * np.maximum(X, 0) ** 2 / 5
    noise = rng.normal(scale=gaussian_scale)
    noise += rng.uniform(-2.4, 2.4, size=n_samples) * (X < 0)
    y = mean_function(X) + noise
    return X.reshape(-1, 1), y


def true_interval_bounds(X):
    x = np.asarray(X).reshape(-1)
    gaussian_scale = 0.8 * np.maximum(x, 0) ** 2 / 5
    true_interval = np.column_stack([mean_function(x), mean_function(x)])
    true_interval[x < 0, 0] -= 2.4 * confidence_level
    true_interval[x < 0, 1] += 2.4 * confidence_level
    normal_half_width = norm.ppf((1 + confidence_level) / 2) * gaussian_scale
    true_interval[x >= 0, 0] -= normal_half_width[x >= 0]
    true_interval[x >= 0, 1] += normal_half_width[x >= 0]
    return true_interval


X, y = generate_data()
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
true_interval_test = true_interval_bounds(X_test)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(X_train[:, 0], y_train, s=8, alpha=0.35, label="Training data")
sort_order = np.argsort(X_train[:, 0])
ax.plot(X_train[sort_order, 0], mean_function(X_train[sort_order, 0]), color="black")
ax.set_title("Heteroscedastic regression data")
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
ax.legend()
plt.tight_layout()
plt.show()


##############################################################################
# 2. Define conditional feature maps
# --------------------------------------------------------------------------
#
# ``polynomial_feature_map`` encodes a simple prior: uncertainty is expected to
# change smoothly with ``X``. ``rbf_feature_map`` is less parametric and uses
# Gaussian bumps spread across the input range. Both include an intercept-like
# component through their first column.


def polynomial_feature_map(X):
    x = np.asarray(X).reshape(-1)
    return np.column_stack([np.ones(len(x)), x, x**2])


rbf_centers = np.linspace(-1, 5, 7)


def rbf_feature_map(X):
    x = np.asarray(X).reshape(-1)
    squared_distances = (x[:, np.newaxis] - rbf_centers[np.newaxis, :]) ** 2
    return np.column_stack([np.ones(len(x)), np.exp(-squared_distances / 0.75)])


##############################################################################
# 3. Fit split and conditional conformal regressors
# --------------------------------------------------------------------------
#
# The estimator is fitted once. The conformal methods share the same
# conformalization samples and differ only in how the residual cutoff is chosen.

estimator = make_pipeline(PolynomialFeatures(4), LinearRegression()).fit(
    X_train, y_train
)

mapie_split = SplitConformalRegressor(
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_split.conformalize(X_conformalize, y_conformalize)
y_pred_split, y_interval_split = mapie_split.predict_interval(X_test)

mapie_poly = ConditionalSplitConformalRegressor(
    polynomial_feature_map,
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_poly.conformalize(X_conformalize, y_conformalize)
y_pred_poly, y_interval_poly = mapie_poly.predict_interval(X_test)

mapie_rbf = ConditionalSplitConformalRegressor(
    rbf_feature_map,
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_rbf.conformalize(X_conformalize, y_conformalize)
y_pred_rbf, y_interval_rbf = mapie_rbf.predict_interval(X_test)


##############################################################################
# 4. Visualize prediction intervals
# --------------------------------------------------------------------------

intervals = [y_interval_split, y_interval_poly, y_interval_rbf]
predictions = [y_pred_split, y_pred_poly, y_pred_rbf]
titles = [
    "Marginal split conformal",
    "Conditional: polynomial feature map",
    "Conditional: RBF feature map",
]
colors = ["C0", "C1", "C2"]
sort_order = np.argsort(X_test[:, 0])

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
for ax, y_pred, y_interval, title, color in zip(
    axes, predictions, intervals, titles, colors
):
    ax.scatter(X_test[:, 0], y_test, s=8, alpha=0.25, color="black")
    ax.plot(X_test[sort_order, 0], y_pred[sort_order], color="black", linewidth=1.5)
    ax.plot(
        X_test[sort_order, 0],
        true_interval_test[sort_order, 0],
        "--",
        color="gray",
        linewidth=1,
    )
    ax.plot(
        X_test[sort_order, 0],
        true_interval_test[sort_order, 1],
        "--",
        color="gray",
        linewidth=1,
    )
    ax.fill_between(
        X_test[sort_order, 0],
        y_interval[sort_order, 0, 0],
        y_interval[sort_order, 1, 0],
        color=color,
        alpha=0.3,
    )
    ax.set_title(title)
    ax.set_xlabel("$X$")
axes[0].set_ylabel("$Y$")
plt.tight_layout()
plt.show()


##############################################################################
# 5. Evaluate local coverage and interval width
# --------------------------------------------------------------------------
#
# We bin the test samples by ``X``. A conditional method is useful when it gets
# closer to the target coverage in each region while assigning width where the
# data are genuinely noisy.

bin_edges = np.linspace(-1, 5, 9)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


def scores_by_bin(y_true, y_interval, X):
    coverages = []
    widths = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (X[:, 0] >= left) & (X[:, 0] < right)
        coverages.append(regression_coverage_score(y_true[mask], y_interval[mask]))
        widths.append(regression_mean_width_score(y_interval[mask]))
    return np.asarray(coverages).ravel(), np.asarray(widths).ravel()


fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for y_interval, title, color in zip(intervals, titles, colors):
    coverage, width = scores_by_bin(y_test, y_interval, X_test)
    axes[0].plot(bin_centers, coverage, marker="o", color=color, label=title)
    axes[1].plot(bin_centers, width, marker="o", color=color, label=title)

axes[0].axhline(confidence_level, color="black", linestyle="--", linewidth=1)
axes[0].set_ylim(0.55, 1.02)
axes[0].set_ylabel("Coverage")
axes[0].set_title("Coverage by region")
axes[0].legend(fontsize=8)

axes[1].set_ylabel("Mean interval width")
axes[1].set_title("Interval width by region")

for ax in axes:
    ax.set_xlabel("$X$ bin center")

plt.tight_layout()
plt.show()
