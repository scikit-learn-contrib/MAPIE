"""
Conditional coverage metrics
============================


This example uses a one-dimensional heteroscedastic regression problem
to compare MAPIE's conditional coverage metrics on marginal and
conditionally-calibrated conformal regressors.

The two regressors use the same fitted polynomial model and the same
conformalization data. The marginal ``SplitConformalRegressor`` uses
one global residual cutoff, while ``ConditionalSplitConformalRegressor``
uses interval indicators in ``X`` as its conditional feature map. The goal is to
show which metrics detect the local undercoverage of the marginal intervals.

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.metrics.conditional import (
    coverage_gap,
    excess_risk_target_coverage,
    worst_slab_coverage,
)
from mapie.metrics.regression import (
    hsic,
    regression_coverage_score,
    regression_mean_width_score,
    regression_ssc,
    regression_ssc_score,
)
from mapie.regression import SplitConformalRegressor

warnings.filterwarnings("ignore")


##############################################################################
# 1. Generate heteroscedastic data
# --------------------------------------------------------------------------
#
# The conditional spread of ``Y`` increases sharply for large values of ``X``.
# A marginal conformal interval can still achieve the requested coverage on
# average, but it tends to overcover low-noise regions and undercover high-noise
# regions.


def mean_function(x):
    return np.sin(1.5 * x) + 0.25 * x


def noise_scale(x):
    return 0.15 + 0.15 * x + 0.9 * (x > 3.0)


def generate_heteroscedastic_data(
    seed,
    n_train=600,
    n_conformalize=700,
    n_test=500,
):
    rng = np.random.default_rng(seed)
    n_train_conformalize = n_train + n_conformalize

    x_train_conformalize = rng.uniform(0, 5, size=n_train_conformalize)
    x_test = rng.uniform(0, 5, size=n_test)

    y_train_conformalize = mean_function(x_train_conformalize) + noise_scale(
        x_train_conformalize
    ) * rng.normal(size=n_train_conformalize)
    y_test = mean_function(x_test) + noise_scale(x_test) * rng.normal(size=n_test)

    X_train_conformalize = x_train_conformalize.reshape(-1, 1)
    X_test = x_test.reshape(-1, 1)

    return (
        X_train_conformalize[:n_train],
        y_train_conformalize[:n_train],
        X_train_conformalize[n_train:],
        y_train_conformalize[n_train:],
        X_test,
        y_test,
    )


(
    X_train,
    y_train,
    X_conformalize,
    y_conformalize,
    X_test,
    y_test,
) = generate_heteroscedastic_data(seed=1)


##############################################################################
# 2. Define conditional groups
# --------------------------------------------------------------------------
#
# The conditional regressor is asked to guarantee coverage on five intervals of
# the input space. The same groups are later used by ``coverage_gap`` to check
# whether the local coverage errors are large.

confidence_level = 0.90
x_bins = np.linspace(0, 5, 6)
x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
x_bin_labels = [
    f"[{left:.0f}, {right:.0f})" for left, right in zip(x_bins[:-1], x_bins[1:])
]
x_bin_labels[-1] = "[4, 5]"


def group_index(X):
    return np.digitize(np.asarray(X).reshape(-1), x_bins[1:-1], right=False)


def indicator_matrix(X):
    groups = group_index(X)
    matrix = np.zeros((len(groups), len(x_bins) - 1))
    matrix[np.arange(len(groups)), groups] = 1
    return matrix


##############################################################################
# 3. Fit marginal and conditional regressors
# --------------------------------------------------------------------------
#
# Both conformal regressors share the same fitted polynomial regressor. The
# conditional method uses a non-symmetric absolute conformity score so the lower
# and upper residual cutoffs can adapt separately by group.

estimator = make_pipeline(PolynomialFeatures(4), LinearRegression()).fit(
    X_train, y_train
)

mapie_marginal = SplitConformalRegressor(
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
)
mapie_marginal.conformalize(X_conformalize, y_conformalize)
y_pred_marginal, y_interval_marginal = mapie_marginal.predict_interval(X_test)

mapie_conditional = ConditionalSplitConformalRegressor(
    indicator_matrix,
    estimator=estimator,
    confidence_level=confidence_level,
    conformity_score=AbsoluteConformityScore(sym=False),
    prefit=True,
)
mapie_conditional.conformalize(X_conformalize, y_conformalize)
y_pred_conditional, y_interval_conditional = mapie_conditional.predict_interval(X_test)


##############################################################################
# 4. Compare conditional coverage metrics
# --------------------------------------------------------------------------
#
# ``coverage_gap`` and ``worst_slab_coverage`` use the covariates directly, so
# they capture the local undercoverage of the marginal interval. ``hsic`` and
# ``regression_ssc_score`` condition on interval width instead. They are useful
# for adaptive intervals, but a constant-width marginal interval can make them
# look harmless even when coverage depends strongly on ``X``.


def interval_width(y_intervals):
    return y_intervals[:, 1, 0] - y_intervals[:, 0, 0]


def safe_regression_ssc_score(y_true, y_intervals, num_bins):
    try:
        return regression_ssc_score(y_true, y_intervals, num_bins=num_bins)[0]
    except ValueError:
        return np.nan


def compute_metrics(name, y_intervals):
    groups = group_index(X_test)
    return {
        "Regressor": name,
        "Marginal coverage": regression_coverage_score(y_test, y_intervals)[0],
        "Mean width": regression_mean_width_score(y_intervals)[0],
        "CovGap": coverage_gap(
            y_test,
            groups,
            confidence_level,
            y_intervals=y_intervals,
        ),
        "WCovGap": coverage_gap(
            y_test,
            groups,
            confidence_level,
            y_intervals=y_intervals,
            weighted=True,
        ),
        "WSC": worst_slab_coverage(
            X_test,
            y_test,
            y_intervals=y_intervals,
            delta=0.15,
            n_directions=100,
            random_state=1,
        ),
        "ERT loss": excess_risk_target_coverage(
            X_test,
            y_test,
            confidence_level,
            y_intervals=y_intervals,
            n_splits=3,
            random_state=1,
        ),
        "SSC min coverage": safe_regression_ssc_score(
            y_test,
            y_intervals,
            num_bins=4,
        ),
        "HSIC": hsic(y_test, y_intervals)[0],
    }


metrics = pd.DataFrame(
    [
        compute_metrics("Marginal", y_interval_marginal),
        compute_metrics("Conditional", y_interval_conditional),
    ]
).set_index("Regressor")

metrics.round(3).style.format("{:.3f}")

##############################################################################
# The two regressors have similar marginal coverage, close to the target 0.90.
# The group-aware metrics tell a different story:
#
# - ``CovGap`` and ``WCovGap`` (lower is better) are much larger for the marginal regressor,
#   because some ``X`` groups are overcovered while the high-noise groups are
#   undercovered.
# - ``WSC`` (higher is better) is lower for the marginal regressor; it searches geometric
#   slabs in feature space and finds a poorly-covered region.
# - ``ERT`` (lower is better) is lower for the conditional regressor: the
#   conditional coverage is closer to the target across ``X``.
# - ``SSC`` and ``HSIC`` only look at coverage as a function of interval width.
#   They are not enough to diagnose a constant-width marginal interval (returning NaN or 0).


##############################################################################
# 5. Visualize coverage by covariate group
# --------------------------------------------------------------------------
#
# The plot below shows why the group-aware metrics separate the two regressors:
# the marginal regressor misses the target in the high-noise bins, whereas the
# conditional regressor keeps each group closer to 90% coverage.


def coverage_by_group(y_true, y_intervals):
    coverages = []
    for group in range(len(x_bins) - 1):
        mask = group_index(X_test) == group
        coverages.append(regression_coverage_score(y_true[mask], y_intervals[mask])[0])
    return np.asarray(coverages)


coverage_marginal_by_group = coverage_by_group(y_test, y_interval_marginal)
coverage_conditional_by_group = coverage_by_group(y_test, y_interval_conditional)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
bar_width = 0.35
bar_positions = np.arange(len(x_bin_labels))

axes[0].bar(
    bar_positions - bar_width / 2,
    coverage_marginal_by_group,
    width=bar_width,
    label="Marginal",
)
axes[0].bar(
    bar_positions + bar_width / 2,
    coverage_conditional_by_group,
    width=bar_width,
    label="Conditional",
)
axes[0].axhline(confidence_level, color="black", linestyle="--", linewidth=1)
axes[0].set_ylim(0.60, 1.02)
axes[0].set_ylabel("Coverage")
axes[0].set_title("Coverage by X group")
axes[0].set_xticks(bar_positions)
axes[0].set_xticklabels(x_bin_labels, rotation=30, ha="right")
axes[0].legend()

sort_order = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sort_order, 0]
axes[1].scatter(X_test[:, 0], y_test, alpha=0.20, s=16, label="Test data")
axes[1].plot(
    X_test_sorted,
    y_pred_marginal[sort_order],
    color="black",
    linewidth=1.5,
    label="Prediction",
)
axes[1].fill_between(
    X_test_sorted,
    y_interval_marginal[sort_order, 0, 0],
    y_interval_marginal[sort_order, 1, 0],
    alpha=0.25,
    label="Marginal interval",
)
axes[1].fill_between(
    X_test_sorted,
    y_interval_conditional[sort_order, 0, 0],
    y_interval_conditional[sort_order, 1, 0],
    alpha=0.25,
    label="Conditional interval",
)
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].set_title("Prediction intervals")
axes[1].legend()

plt.tight_layout()
plt.show()


##############################################################################
# 6. Check the width-conditioned diagnostics
# --------------------------------------------------------------------------
#
# The conditional interval has several distinct widths, so its
# size-stratified coverage can be plotted. The marginal interval is almost
# constant-width, and MAPIE correctly refuses to split it into width bins.

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

width_marginal = interval_width(y_interval_marginal)
width_conditional = interval_width(y_interval_conditional)
width_bins = np.linspace(
    min(width_marginal.min(), width_conditional.min()),
    max(width_marginal.max(), width_conditional.max()),
    9,
)

axes[0].hist(width_marginal, bins=width_bins, alpha=0.7, label="Marginal")
axes[0].hist(
    width_conditional,
    bins=width_bins,
    alpha=0.7,
    label="Conditional",
)
axes[0].set_xlabel("Interval width")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of interval widths")
axes[0].legend()

ssc_conditional = regression_ssc(y_test, y_interval_conditional, num_bins=4)[0]
axes[1].bar(np.arange(len(ssc_conditional)), ssc_conditional)
axes[1].axhline(confidence_level, color="black", linestyle="--", linewidth=1)
axes[1].set_ylim(0.60, 1.02)
axes[1].set_xlabel("Width bin")
axes[1].set_ylabel("Coverage")
axes[1].set_title("Conditional regressor: SSC by width")

plt.tight_layout()
plt.show()
