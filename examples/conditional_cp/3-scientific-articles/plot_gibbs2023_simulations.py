"""
Reproduction of Gibbs et al. (2023) conditional simulations
===========================================================


This example reproduces the spirit of the one-dimensional simulations from
Gibbs, Cherian and Candès (2023) [1] with MAPIE's
:class:`~mapie.conditional_conformal_prediction.ConditionalSplitConformalRegressor`.

The original paper studies conditional coverage under user-chosen covariate
shifts. We compare standard split conformal prediction with conditional split
conformal prediction for two feature maps:

- interval indicators over ``X`` ("Groups");
- smooth Gaussian bumps around two evaluation locations ("Shifts").

[1] Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès,
"Conformal Prediction With Conditional Guarantees",
`arXiv <https://arxiv.org/abs/2305.12616>`_, 2023.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.regression import SplitConformalRegressor

##############################################################################
# 1. Data-generating process from Gibbs et al. (2023)
# --------------------------------------------------------------------------


confidence_level = 0.9
alpha = 1 - confidence_level


def init_model():
    return make_pipeline(PolynomialFeatures(4), LinearRegression())


def generate_data(seed, n_train=1200, n_calib=700, n_test=250):
    rng = np.random.default_rng(seed)

    def f(x):
        y = np.zeros_like(x)
        for i, value in enumerate(x):
            y[i] = rng.poisson(np.sin(value) ** 2 + 0.1)
            y[i] += 0.03 * value * rng.normal()
            y[i] += 25 * (rng.uniform() < 0.01) * rng.normal()
        return y.astype(np.float64)

    X_train = rng.uniform(0, 5.0, size=n_train)
    X_calib = rng.uniform(0, 5.0, size=n_calib)
    X_test = rng.uniform(0, 5.0, size=n_test)
    y_train = f(X_train)
    y_calib = f(X_calib)
    y_test = f(X_test)
    return (
        X_train.reshape(-1, 1),
        y_train,
        X_calib.reshape(-1, 1),
        y_calib,
        X_test.reshape(-1, 1),
        y_test,
    )


X_train, y_train, X_calib, y_calib, X_test, y_test = generate_data(seed=1)
model = init_model().fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
axes[0].scatter(X_train[:, 0], y_train, s=8, alpha=0.35)
axes[0].set_title("Training data")
axes[0].set_xlabel("$X$")
axes[0].set_ylabel("$Y$")
axes[1].scatter(X_train[:, 0], y_train, s=8, alpha=0.35)
axes[1].set_ylim(-2, 6)
axes[1].set_title("Training data, zoomed")
axes[1].set_xlabel("$X$")
plt.tight_layout()
plt.show()


##############################################################################
# 2. Conditional feature maps
# --------------------------------------------------------------------------

group_edges = np.arange(0, 5.5, 0.5)


def group_feature_map(X):
    x = np.asarray(X).reshape(-1)
    indexes = np.digitize(x, group_edges[1:-1], right=False)
    features = np.zeros((len(x), len(group_edges) - 1))
    features[np.arange(len(x)), indexes] = 1
    return features


eval_locs = np.array([1.5, 3.5])
other_locs = np.array([0.5, 2.5, 4.5])
shift_locs = np.concatenate([eval_locs, other_locs])
shift_scales = np.array([0.2, 0.2, 1.0, 1.0, 1.0])


def shift_feature_map(X):
    x = np.asarray(X).reshape(-1, 1)
    bumps = norm.pdf(x, loc=shift_locs.reshape(1, -1), scale=shift_scales)
    return np.column_stack([np.ones(len(x)), bumps])


##############################################################################
# 3. Helpers for fitting, plotting, and repeated evaluation
# --------------------------------------------------------------------------


def predict_split_interval(X_calib, y_calib, X_test):
    mapie_split = SplitConformalRegressor(
        estimator=model,
        confidence_level=confidence_level,
        conformity_score=AbsoluteConformityScore(sym=False),
        prefit=True,
    )
    mapie_split.conformalize(X_calib, y_calib)
    return mapie_split.predict_interval(X_test)[1]


def predict_conditional_interval(X_calib, y_calib, X_test, feature_map, seed=0):
    mapie_conditional = ConditionalSplitConformalRegressor(
        feature_map,
        estimator=model,
        confidence_level=confidence_level,
        conformity_score=AbsoluteConformityScore(sym=False),
        prefit=True,
        randomize=True,
        seed=seed,
    )
    mapie_conditional.conformalize(X_calib, y_calib)
    return mapie_conditional.predict_interval(X_test)[1]


def miscoverage(y_true, interval):
    return (y_true < interval[:, 0, 0]) | (y_true > interval[:, 1, 0])


def weighted_miscoverage(miscover, weights):
    return np.sum(miscover * weights) / np.sum(weights)


def evaluation_functions(experiment):
    if experiment == "Groups":
        return {
            "Marginal": lambda X: np.ones(len(X), dtype=float),
            "[1, 2]": lambda X: ((X[:, 0] >= 1) & (X[:, 0] <= 2)).astype(float),
            "[3, 4]": lambda X: ((X[:, 0] >= 3) & (X[:, 0] <= 4)).astype(float),
        }
    return {
        "Marginal": lambda X: np.ones(len(X), dtype=float),
        "Shift 1.5": lambda X: norm.pdf(X[:, 0], loc=1.5, scale=0.2),
        "Shift 3.5": lambda X: norm.pdf(X[:, 0], loc=3.5, scale=0.2),
    }


def summarize_miscoverage(feature_map, experiment, n_trials=8):
    rows = []
    functions = evaluation_functions(experiment)
    for seed in range(n_trials):
        _, _, X_calib_t, y_calib_t, X_test_t, y_test_t = generate_data(seed=seed)
        interval_split = predict_split_interval(X_calib_t, y_calib_t, X_test_t)
        interval_conditional = predict_conditional_interval(
            X_calib_t, y_calib_t, X_test_t, feature_map, seed=seed
        )

        for method, interval in [
            ("Split", interval_split),
            ("Conditional", interval_conditional),
        ]:
            errors = miscoverage(y_test_t, interval)
            for name, function in functions.items():
                rows.append(
                    {
                        "Experiment": experiment,
                        "Method": method,
                        "Region": name,
                        "Miscoverage": weighted_miscoverage(errors, function(X_test_t)),
                    }
                )
    return pd.DataFrame(rows)


def plot_intervals(feature_map, experiment):
    interval_split = predict_split_interval(X_calib, y_calib, X_test)
    interval_conditional = predict_conditional_interval(
        X_calib, y_calib, X_test, feature_map
    )

    sort_order = np.argsort(X_test[:, 0])
    x_sorted = X_test[sort_order, 0]
    y_sorted = y_test[sort_order]
    y_pred_sorted = model.predict(X_test[sort_order])
    intervals = [interval_split, interval_conditional]
    titles = ["Split calibration", "Conditional calibration"]
    colors = ["C0", "C1"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    for ax, interval, title, color in zip(axes, intervals, titles, colors):
        ax.plot(x_sorted, y_sorted, ".", alpha=0.2)
        ax.plot(x_sorted, y_pred_sorted, color="black", linewidth=1)
        ax.fill_between(
            x_sorted,
            interval[sort_order, 0, 0],
            interval[sort_order, 1, 0],
            color=color,
            alpha=0.35,
        )
        ax.set_ylim(-2, 6.5)
        ax.set_title(title)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")

        if experiment == "Groups":
            ax.axvspan(1, 2, facecolor="gray", alpha=0.25)
            ax.axvspan(3, 4, facecolor="gray", alpha=0.25)
        else:
            for loc in eval_locs:
                ax.plot(
                    x_sorted,
                    norm.pdf(x_sorted, loc=loc, scale=0.2),
                    color="gray",
                    linestyle="--",
                    linewidth=2,
                )
    plt.tight_layout()
    plt.show()


def plot_miscoverage(summary, experiment):
    means = (
        summary.groupby(["Region", "Method"], sort=False)["Miscoverage"]
        .mean()
        .unstack()
    )
    x = np.arange(len(means.index))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - bar_width / 2, means["Split"], width=bar_width, label="Split")
    ax.bar(
        x + bar_width / 2,
        means["Conditional"],
        width=bar_width,
        label="Conditional",
    )
    ax.axhline(alpha, color="red", linewidth=1, label=r"Target $\alpha$")
    ax.set_xticks(x)
    ax.set_xticklabels(means.index)
    ax.set_ylim(0, 0.22)
    ax.set_ylabel("Miscoverage")
    ax.set_xlabel(experiment)
    ax.legend()
    plt.tight_layout()
    plt.show()


##############################################################################
# 4. Reproduce grouped and shifted conditional guarantees
# --------------------------------------------------------------------------
#
# The bars report average miscoverage over repeated calibration/test draws.
# The conditional method is calibrated against the selected feature map, so it
# better balances miscoverage on the corresponding groups or shifts.

plot_intervals(group_feature_map, "Groups")
group_summary = summarize_miscoverage(group_feature_map, "Groups")
plot_miscoverage(group_summary, "Groups")

plot_intervals(shift_feature_map, "Shifts")
shift_summary = summarize_miscoverage(shift_feature_map, "Shifts")
plot_miscoverage(shift_summary, "Shifts")
