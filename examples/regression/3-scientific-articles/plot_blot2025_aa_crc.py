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

RANDOM_STATE = 42
ALPHA = 0.1


##############################################################################
# Generate the synthetic regression data
# --------------------------------------


def generate_data(seed=RANDOM_STATE, n_train=1000, n_calib=100, n_test=120):
    """Generate heteroscedastic data."""
    rng = np.random.RandomState(seed)

    def response(x):
        y = np.zeros(len(x), dtype=float)
        for i, value in enumerate(x):
            y[i] = rng.poisson(np.sin(value) ** 2 + 0.1)
            y[i] += 0.03 * value * rng.randn()
            y[i] += 25 * (rng.uniform() < 0.01) * rng.randn()
        return y

    X = rng.uniform(0, 5, size=n_train + n_calib).astype(np.float32)
    X_test = rng.uniform(0, 5, size=n_test).astype(np.float32)
    y = response(X)
    y_test = response(X_test)
    return (
        X[:n_train, None],
        y[:n_train],
        X[n_train:, None],
        y[n_train:],
        X_test[:, None],
        y_test,
    )


X_train, y_train, X_calib, y_calib, X_test, y_test = generate_data()


##############################################################################
# Fit the point predictor and define simple groups
# ------------------------------------------------

point_predictor = make_pipeline(
    PolynomialFeatures(degree=4),
    LinearRegression(),
).fit(X_train, y_train)

GROUP_WIDTH = 5 / 3
GROUP_STARTS = np.arange(0, 5.0, GROUP_WIDTH)


def group_feature_map(X):
    """Return indicators of fixed-width intervals of the input space."""
    x = np.asarray(X)
    return np.column_stack(
        [
            ((x >= start) & (x < start + GROUP_WIDTH)).astype(np.float32).ravel()
            for start in GROUP_STARTS
        ]
    )


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
    feature_map=group_feature_map,
    confidence_level=1 - ALPHA,
    risk="miscoverage",
    predict_param_range=(0.0, 5.0),
    learning_rate=1e-1,
    weight_decay=0.0,
)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
aa_controller = ConditionalRiskController(**controller_params)
aa_controller.conformalize(X_calib, y_calib, n_epochs=10, batch_size=100)

y_interval_aa = aa_controller.predict(X_test, n_epochs=1, batch_size=100)
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

test_groups = group_feature_map(X_test).astype(bool)
aa_covered = (y_test >= y_lower) & (y_test <= y_upper)
split_covered = (y_test >= y_split_lower) & (y_test <= y_split_upper)
n_groups = len(GROUP_STARTS)
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
ax_interval.set(xlabel="$x$", ylabel="$y$", ylim=(-5, 6))
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
