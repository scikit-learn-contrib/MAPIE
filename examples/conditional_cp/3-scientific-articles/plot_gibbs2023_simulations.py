"""
Reproduction of experiments of Gibbs et al. (2023)
==================================================

``ConditionalSplitConformalRegressor`` is used to reproduce part of the
experiments from Gibbs et al. (2023) [1]. Their method produces adaptive
prediction intervals with coverage guarantees for all subgroups of interest.

For a given model, the simulation uses
``ConditionalSplitConformalRegressor`` on a synthetic dataset first considered
by Romano et al. (2019) [2], and compares the interval bounds with those from
the standard ``SplitConformalRegressor``.

This simulation is carried out to check that the conditional method implemented in
MAPIE gives the same results as [1], and that the bounds of the intervals are
obtained.

[1] Isaac Gibbs, John J. Cherian, Emmanuel J. Candès (2023).
[Conformal Prediction With Conditional Guarantees](https://arxiv.org/abs/2305.12616).

[2] Yaniv Romano, Evan Patterson, Emmanuel J. Candès (2019).
Conformalized Quantile Regression.
33rd Conference on Neural Information Processing Systems (NeurIPS 2019).
"""

# mkdocs_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.regression import SplitConformalRegressor

random_state = 1
ALPHA = 0.1
confidence_level = 1 - ALPHA


###############################################################################
# 1. Global model parameters
# -----------------------------------------------------------------------------


def init_model():
    # the degree of the polynomial regression
    degree = 4

    model = Pipeline(
        [("poly", PolynomialFeatures(degree=degree)), ("linear", LinearRegression())]
    )
    return model


###############################################################################
# 2. Generate and show data
# -----------------------------------------------------------------------------


def generate_data(seed=random_state, n_train=2000, n_calib=1000, n_test=500):
    np.random.seed(seed)
    n_train = n_train + n_calib

    def f(x):
        ax = 0 * x
        for i in range(len(x)):
            ax[i] = (
                np.random.poisson(np.sin(x[i]) ** 2 + 0.1)
                + 0.03 * x[i] * np.random.randn(1)
            ).item()
            ax[i] += (
                25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
            ).item()
        return ax.astype(np.float32)

    X_train = np.random.uniform(0, 5.0, size=n_train).astype(np.float32)
    X_test = np.random.uniform(0, 5.0, size=n_test).astype(np.float32)
    y_train = f(X_train)
    y_test = f(X_test)

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    train_set_size = len(y_train) - n_calib
    X_train_final = X_train[:train_set_size]
    X_calib = X_train[train_set_size:]
    y_train_final = y_train[:train_set_size]
    y_calib = y_train[train_set_size:]

    return X_train_final, y_train_final, X_calib, y_calib, X_test, y_test


X_train, y_train, X_calib, y_calib, X_test, y_test = generate_data(n_calib=2000)
X_calib = np.asarray(X_calib, dtype=np.float64)
y_calib = np.asarray(y_calib, dtype=np.float64)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(X_train[:, 0], y_train, s=1.5, alpha=0.6, label="Train Data")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_title("Train Data")
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(X_train[:, 0], y_train, s=1.5, alpha=0.6, label="Train Data")
ax2.set_ylim([-2, 6])
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Zoom")
ax2.legend()

plt.show()

##############################################################################
# 3. Prepare model and show predictions
# -----------------------------------------------------------------------------

model = init_model()
model.fit(X_train, y_train)

sort_order = np.argsort(X_test[:, 0])
x_test_s = X_test[sort_order]
y_pred_s = model.predict(x_test_s)

plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], y_test, s=1.5, alpha=0.6, label="Test Data")
plt.plot(x_test_s, y_pred_s, "-k", label="Prediction")
plt.ylim([-2, 6])
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Test Data (Zoom)")
plt.legend()
plt.show()


##############################################################################
# 4. Prepare Experiments
# -----------------------------------------------------------------------------
# In this experiment, we will use the
# ``SplitConformalRegressor`` and
# ``ConditionalSplitConformalRegressor``
# to compute prediction intervals.

eval_locs = np.array([1.5, 3.5])
eval_scale = 0.2
other_locs = np.array([0.5, 2.5, 4.5])
other_scale = 1.0
shift_locs = np.concatenate([eval_locs, other_locs])
shift_scales = np.array([eval_scale] * len(eval_locs) + [other_scale] * len(other_locs))


def group_feature_map(X):
    x = np.asarray(X)
    return np.column_stack(
        [
            ((x >= t) & (x < t + 0.5)).astype(float).ravel()
            for t in np.arange(0, 5.0, 0.5)
        ]
    )


def shift_feature_map(X):
    x = np.asarray(X).reshape(-1, 1)
    gaussian_features = norm.pdf(
        x,
        loc=shift_locs.reshape(1, -1),
        scale=shift_scales.reshape(1, -1),
    )
    return np.column_stack([np.ones(len(x)), gaussian_features])


def fit_split_interval(model, X_calib, y_calib, X_test):
    mapie_split = SplitConformalRegressor(
        model,
        confidence_level=confidence_level,
        conformity_score=AbsoluteConformityScore(sym=True),
        prefit=True,
    )
    mapie_split.conformalize(X_calib, y_calib)
    return mapie_split.predict_interval(X_test)


def fit_ccp_interval(
    model,
    X_calib,
    y_calib,
    X_test,
    feature_map,
    seed=0,
):
    mapie_ccp = ConditionalSplitConformalRegressor(
        feature_map,
        estimator=model,
        confidence_level=confidence_level,
        conformity_score=AbsoluteConformityScore(sym=False),
        prefit=True,
        seed=seed,
    )
    mapie_ccp.conformalize(X_calib, y_calib)
    return mapie_ccp.predict_interval(X_test)


def estimate_coverage(feature_map, group_functs=None, seed=0):
    if group_functs is None:
        group_functs = []

    _, _, X_calib, y_calib, X_test, y_test = generate_data(seed=seed, n_calib=2000)
    X_calib = np.asarray(X_calib, dtype=np.float64)
    y_calib = np.asarray(y_calib, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    _, y_pi_split = fit_split_interval(model, X_calib, y_calib, X_test)
    _, y_pi_ccp = fit_ccp_interval(
        model,
        X_calib,
        y_calib,
        X_test,
        feature_map,
        seed=seed,
    )

    cover_split = np.logical_or(
        y_test < y_pi_split[:, 0, 0], y_test > y_pi_split[:, 1, 0]
    )
    cover_ccp = np.logical_or(y_test < y_pi_ccp[:, 0, 0], y_test > y_pi_ccp[:, 1, 0])
    group_covers = []
    marginal_cover = np.asarray((cover_split.mean(), cover_ccp.mean()))
    for funct in group_functs:
        weights = funct(X_test).ravel()
        group_cover = np.zeros((2,))
        group_cover[0] = np.sum(weights * cover_split) / np.sum(weights)
        group_cover[1] = np.sum(weights * cover_ccp) / np.sum(weights)
        group_covers.append(group_cover)
    return marginal_cover, np.array(group_covers)


def plot_results(X_test, y_test, n_trials=20, experiment="Groups"):
    _, y_pi_split = fit_split_interval(model, X_calib, y_calib, X_test)

    if experiment == "Groups":
        feature_map = group_feature_map
        eval_functions = [
            lambda X, a=a, b=b: ((X > a) & (X < b)).astype(float)
            for a, b in zip([1, 3], [2, 4])
        ]
        eval_names = ["[1,2]", "[3,4]"]
    elif experiment == "Shifts":
        feature_map = shift_feature_map
        eval_functions = [
            lambda x: norm.pdf(x, loc=1.5, scale=0.2).reshape(-1, 1),
            lambda x: norm.pdf(x, loc=3.5, scale=0.2).reshape(-1, 1),
        ]
        eval_names = ["f1", "f2"]
    else:
        raise ValueError("Wrong experiment name")

    _, y_pi_ccp = fit_ccp_interval(model, X_calib, y_calib, X_test, feature_map)

    marginal_cov = np.zeros((n_trials, 2))
    group_cov = np.zeros((len(eval_functions), n_trials, 2))
    for j in range(n_trials):
        marginal_cov[j], group_cov[:, j, :] = estimate_coverage(
            feature_map, eval_functions, seed=j
        )

    coverage_data = pd.DataFrame()

    for group, cov in zip(["Marginal"] + eval_names, [marginal_cov] + list(group_cov)):
        for i, name in enumerate(["Split", "Conditional"]):
            coverage_data = pd.concat(
                [
                    coverage_data,
                    pd.DataFrame(
                        {
                            "Method": [name] * len(cov),
                            "Range": [group] * len(cov),
                            "Miscoverage": np.asarray(cov)[:, i],
                        }
                    ),
                ],
                axis=0,
            )

    cp = plt.get_cmap("tab10").colors

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.grid"] = False

    fig = plt.figure()
    fig.set_size_inches(17, 6)

    sort_order = np.argsort(X_test[:, 0])
    x_test_s = X_test[sort_order]
    y_test_s = y_test[sort_order]
    y_pred_s = model.predict(x_test_s)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x_test_s, y_test_s, ".", alpha=0.2)
    ax1.plot(x_test_s, y_pred_s, lw=1, color="k")
    ax1.plot(x_test_s, y_pi_split[sort_order, 0, 0], color=cp[0], lw=2)
    ax1.plot(x_test_s, y_pi_split[sort_order, 1, 0], color=cp[0], lw=2)
    ax1.fill_between(
        x_test_s.flatten(),
        y_pi_split[sort_order, 0, 0],
        y_pi_split[sort_order, 1, 0],
        color=cp[0],
        alpha=0.4,
        label="split prediction interval",
    )
    ax1.set_ylim(-2, 6.5)
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax1.set_xlabel("$X$", fontsize=16, labelpad=10)
    ax1.set_ylabel("$Y$", fontsize=16, labelpad=10)
    ax1.set_title("Split calibration", fontsize=18, pad=12)

    if experiment == "Groups":
        ax1.axvspan(1, 2, facecolor="grey", alpha=0.25)
        ax1.axvspan(3, 4, facecolor="grey", alpha=0.25)
    else:
        for loc in eval_locs:
            ax1.plot(
                x_test_s,
                norm.pdf(x_test_s, loc=loc, scale=eval_scale),
                color="grey",
                ls="--",
                lw=3,
            )

    ax2 = fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax2.plot(x_test_s, y_test_s, ".", alpha=0.2)
    ax2.plot(x_test_s, y_pred_s, color="k", lw=1)
    ax2.plot(x_test_s, y_pi_ccp[sort_order, 0, 0], color=cp[1], lw=2)
    ax2.plot(x_test_s, y_pi_ccp[sort_order, 1, 0], color=cp[1], lw=2)
    ax2.fill_between(
        x_test_s.flatten(),
        y_pi_ccp[sort_order, 0, 0],
        y_pi_ccp[sort_order, 1, 0],
        color=cp[1],
        alpha=0.4,
        label="conditional calibration",
    )
    ax2.tick_params(axis="both", which="major", direction="out", labelsize=14)
    ax2.set_xlabel("$X$", fontsize=16, labelpad=10)
    ax2.set_ylabel("$Y$", fontsize=16, labelpad=10)
    ax2.set_title("Conditional calibration", fontsize=18, pad=12)

    if experiment == "Groups":
        ax2.axvspan(1, 2, facecolor="grey", alpha=0.25)
        ax2.axvspan(3, 4, facecolor="grey", alpha=0.25)
    else:
        for loc in eval_locs:
            ax2.plot(
                x_test_s,
                norm.pdf(x_test_s, loc=loc, scale=eval_scale),
                color="grey",
                ls="--",
                lw=3,
            )

    ax3 = fig.add_subplot(1, 3, 3)

    ranges = coverage_data["Range"].unique()
    methods = coverage_data["Method"].unique()
    bar_width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        method_data = coverage_data[coverage_data["Method"] == method]
        x = np.arange(len(ranges)) + i * bar_width
        ax3.bar(
            x,
            method_data.groupby("Range")["Miscoverage"].mean(),
            width=bar_width,
            label=method,
            color=cp[i],
        )

    ax3.set_xticks(np.arange(len(ranges)) + bar_width * (len(methods) - 1) / 2)
    ax3.set_xticklabels(ranges)

    ax3.axhline(ALPHA, color="red")
    ax3.legend()
    ax3.set_ylabel("Miscoverage", fontsize=18, labelpad=10)
    ax3.set_xlabel(experiment, fontsize=18, labelpad=10)
    ax3.set_ylim(0.0, 0.2)
    ax3.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout(pad=2)
    plt.show()


##############################################################################
# 5. Reproduce experiment and results
# -----------------------------------------------------------------------------

###############################################################################
# Group-conditional experiment
# -----------------------------------------------------------------------------
# This first experiment consists of performing group-conditional conformal
# prediction. Groups are defined with intervals of $x$.

plot_results(X_test, y_test, experiment="Groups")

###############################################################################
# Covariate shift experiment
# -----------------------------------------------------------------------------
# This second experiment illustrates the case of covariate shift.

plot_results(X_test, y_test, experiment="Shifts")


##############################################################################
# We successfully reproduced the experiment of the Gibbs et al. paper [1].
