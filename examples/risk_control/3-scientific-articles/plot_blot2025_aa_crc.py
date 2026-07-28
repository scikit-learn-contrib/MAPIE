"""
Automatically Adaptive Conformal Risk Control, Blot et al. (2025)
=================================================================

This example gives a lightweight illustration inspired by Figure 2 of Blot
et al. (2025) [1].

A polynomial regression supplies point predictions, while indicators of fixed
intervals of the input space form the embedding used by
``ConditionalRiskController``. The controller learns input-dependent interval
widths using the differentiable PyTorch ``miscoverage_loss``.

The top panel compares automatically adaptive conformal risk control (AA-CRC)
with ``SplitConformalRegressor``. The bottom panel shows coverage within each
group of the feature map.

[1] Vincent Blot, Anastasios N. Angelopoulos, Michael I. Jordan, and
Nicolas J-B. Brunel. "Automatically Adaptive Conformal Risk Control."
AISTATS, 2025.
"""

# mkdocs_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.regression import SplitConformalRegressor
from mapie.risk_control import ConditionalRiskController
from mapie.utils import train_conformalize_test_split

RANDOM_STATE = 42
ALPHA = 0.1


##############################################################################
# Generate grouped regression data
# --------------------------------

x_bins = np.array([-1.0, 0.0, 1.5, 3.0, 5.0])


def mean_function(x):
    return x * np.sin(x)


def generate_grouped_regression_data(n_samples=1000, random_state=RANDOM_STATE):
    """Generate heteroscedastic data over four groups."""
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
    X_calib,
    X_test,
    y_train,
    y_calib,
    y_test,
) = train_conformalize_test_split(
    X,
    y,
    train_size=0.5,
    conformalize_size=0.25,
    test_size=0.25,
    random_state=RANDOM_STATE,
)


##############################################################################
# Fit the point predictor and define the four groups
# --------------------------------------------------

point_predictor = make_pipeline(
    PolynomialFeatures(degree=4),
    LinearRegression(),
).fit(X_train, y_train)


def feature_map(X):
    """Return one indicator column for each x group."""
    x = np.asarray(X).reshape(-1)
    bin_indexes = np.digitize(x, x_bins[1:-1], right=False)
    matrix = np.zeros((len(x), len(x_bins) - 1))
    matrix[np.arange(len(x)), bin_indexes] = 1
    return matrix


##############################################################################
# Define the interval prediction function
# ---------------------------------------
# Called with only ``X``, it returns the raw point predictions used by the
# differentiable loss. Called with widths, it returns the final intervals.


def interval_prediction(X, widths=None):
    y_pred = point_predictor.predict(X)
    if widths is None:
        return y_pred
    return np.column_stack([y_pred - widths, y_pred + widths])


##############################################################################
# Fit AA-CRC and split conformal intervals
# -----------------------------------------

controller_params = dict(
    predict_function=interval_prediction,
    feature_map=feature_map,
    confidence_level=1 - ALPHA,
    risk="miscoverage",
    predict_param_range=(0.0, 5.0),
    learning_rate=1e-1,
    weight_decay=0.0,
)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
aa_controller = ConditionalRiskController(**controller_params)
aa_controller.conformalize(X_calib, y_calib, n_epochs=200)

y_interval_aa = aa_controller.predict(X_test, n_epochs=1)
y_lower = y_interval_aa[:, 0]
y_upper = y_interval_aa[:, 1]

split_controller = SplitConformalRegressor(
    estimator=point_predictor,
    confidence_level=1 - ALPHA,
    prefit=True,
)
split_controller.conformalize(X_calib, y_calib)
y_pred, y_interval = split_controller.predict_interval(X_test)
y_split_lower = y_interval[:, 0, 0]
y_split_upper = y_interval[:, 1, 0]


##############################################################################
# Plot the intervals and group coverage
# -------------------------------------

test_groups = feature_map(X_test).astype(bool)
aa_covered = (y_test >= y_lower) & (y_test <= y_upper)
split_covered = (y_test >= y_split_lower) & (y_test <= y_split_upper)
n_groups = len(x_bins) - 1
aa_group_coverage = np.array(
    [aa_covered[test_groups[:, group]].mean() for group in range(n_groups)]
)
split_group_coverage = np.array(
    [split_covered[test_groups[:, group]].mean() for group in range(n_groups)]
)

order = np.argsort(X_test[:, 0])
fig, (ax_interval, ax_coverage) = plt.subplots(
    2,
    1,
    figsize=(9, 8),
    gridspec_kw={"height_ratios": [2, 1]},
)

ax_interval.scatter(
    X_test[order, 0],
    y_test[order],
    s=8,
    alpha=0.45,
    color="tab:blue",
    label="Test data",
)
ax_interval.plot(
    X_test[order, 0],
    y_pred[order],
    color="tab:blue",
    linewidth=2,
    label="Model prediction",
)
ax_interval.fill_between(
    X_test[order, 0],
    y_lower[order],
    y_upper[order],
    color="tab:orange",
    alpha=0.45,
    label="AA-CRC interval",
)
ax_interval.plot(
    X_test[order, 0],
    y_split_lower[order],
    "k--",
    linewidth=1.2,
    label="Split conformal interval",
)
ax_interval.plot(
    X_test[order, 0],
    y_split_upper[order],
    "k--",
    linewidth=1.2,
)
ax_interval.set(xlabel="$x$", ylabel="$y$")
ax_interval.set_title("AA-CRC and split conformal prediction intervals")
ax_interval.legend(loc="upper left", ncols=2)

groups = np.arange(n_groups)
bar_width = 0.4
ax_coverage.bar(
    groups - bar_width / 2,
    aa_group_coverage,
    width=bar_width,
    label="AA-CRC",
)
ax_coverage.bar(
    groups + bar_width / 2,
    split_group_coverage,
    width=bar_width,
    label="Split conformal",
)
ax_coverage.axhline(
    1 - ALPHA,
    color="tab:red",
    linestyle="--",
    label=r"$1-\alpha=0.9$",
)
ax_coverage.set(
    xlabel="Feature-map group",
    ylabel="Coverage",
    ylim=(0, 1),
    xticks=groups,
    xticklabels=[
        f"[{left:g}, {right:g}{']' if group == n_groups - 1 else ')'}"
        for group, (left, right) in enumerate(zip(x_bins[:-1], x_bins[1:]))
    ],
)
ax_coverage.set_title("Coverage within each feature-map group")
ax_coverage.legend(loc="lower right")

plt.tight_layout()
plt.show()

print(f"AA-CRC marginal coverage: {aa_covered.mean():.3f}")
print(
    "AA-CRC group coverage range: "
    f"[{aa_group_coverage.min():.3f}, {aa_group_coverage.max():.3f}]"
)
print(f"Split conformal marginal coverage: {split_covered.mean():.3f}")
print(
    "Split conformal group coverage range: "
    f"[{split_group_coverage.min():.3f}, {split_group_coverage.max():.3f}]"
)
