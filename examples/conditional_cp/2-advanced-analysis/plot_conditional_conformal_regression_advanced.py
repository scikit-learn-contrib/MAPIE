"""
Group-conditional prediction intervals (advanced)
=================================================

The tutorial will explain how to use the conditional conformal prediction
for regression with ``ConditionalSplitConformalRegressor``, and
will compare it with the other methods available in MAPIE. The conditional method
implements the method described in the Gibbs et al. (2023) paper [1].
It has a lot of advantages:

- It is model agnostic (it doesn't depend on the model but only on the
  predictions, unlike conformalized quantile regression (CQR)),
- It can create very adaptive intervals (with a varying width which truly
  reflects the model uncertainty),
- while providing coverage guarantee on all subgroups of interest
  (avoiding biases),
- with the possibility to inject prior knowledge about the data or the model.

However, we will also see its disadvantages:

- The adaptivity depends on the feature map used, which can be difficult
  to define,
- The inference step (``predict_interval``) takes much longer than for the
  other methods, as an optimization process is solved for each test point.


----

In this tutorial, we will use a synthetic toy dataset.
The estimator will be ``Pipeline``
with ``PolynomialFeatures`` and
``LinearRegression`` (or
``QuantileRegressor`` for CQR).

We will compare the different available feature maps of the conditional method
(using ``ConditionalSplitConformalRegressor``),
with the standard split-conformal method, the CV+ method
(``CrossConformalRegressor``), CQR (``ConformalizedQuantileRegressor``), and
cross CQR (``CrossConformalizedQuantileRegressor``)

[1] Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès,
"Conformal Prediction With Conditional Guarantees",
[arXiv](https://arxiv.org/abs/2305.12616), 2023.
"""

# mkdocs_gallery_thumbnail_number = 4

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor
from mapie.regression import (
    ConformalizedQuantileRegressor,
    CrossConformalRegressor,
    CrossConformalizedQuantileRegressor,
    SplitConformalRegressor,
)

random_state = 42
rng = np.random.default_rng(random_state)

ALPHA = 0.1
confidence_level = 1 - ALPHA

##############################################################################
# 1. Data generation
# --------------------------------------------------------------------------
# Let's start by creating some synthetic data with different domains and
# distributions to evaluate the adaptivity of the methods:
#  - baseline distribution of ``x*sin(x)``
#  - Add noise :
#   - between -1 and 0: uniform distribution of the points around the baseline
#   - between 0 and 5: normal distribution with a noise value which
#     increase with ``x``
#
# We use reduced sample sizes compared with the paper-scale experiment so that
# the documentation example remains fast enough to execute.


def x_sinx(x):
    """One-dimensional x*sin(x) function."""
    return x * np.sin(x)


def get_1d_data_with_heteroscedastic_noise(
    min_x=-1,
    max_x=5,
    n_samples=1400,
    noise=0.8,
    power=2,
    seed=42,
):
    data_rng = np.random.default_rng(seed)
    X = data_rng.uniform(min_x, max_x, size=n_samples)
    normal_scale = noise * (np.maximum(X, 0) / max_x) ** power * max_x
    y = x_sinx(X) + data_rng.normal(0, normal_scale)
    y += data_rng.uniform(-noise * 3, noise * 3, size=n_samples) * (X < 0)

    true_pi = np.column_stack([x_sinx(X), x_sinx(X)])
    true_pi[X < 0, 0] -= noise * 3 * confidence_level
    true_pi[X < 0, 1] += noise * 3 * confidence_level
    normal_half_width = norm.ppf((1 + confidence_level) / 2) * normal_scale
    true_pi[X >= 0, 0] -= normal_half_width[X >= 0]
    true_pi[X >= 0, 1] += normal_half_width[X >= 0]

    return X.reshape(-1, 1), y, true_pi


def generate_data(n_train=500, n_calib=500, n_test=350, seed=42):
    X, y, true_pi = get_1d_data_with_heteroscedastic_noise(
        n_samples=n_train + n_calib + n_test,
        seed=seed,
    )
    permutation = rng.permutation(len(X))
    train_indexes = permutation[:n_train]
    calib_indexes = permutation[n_train : n_train + n_calib]
    test_indexes = permutation[n_train + n_calib :]
    return (
        X[train_indexes],
        y[train_indexes],
        X[calib_indexes],
        y[calib_indexes],
        X[test_indexes],
        y[test_indexes],
        true_pi[test_indexes],
    )


X_train, y_train, X_calib, y_calib, X_test, y_test, test_pi = generate_data()

plt.scatter(X_train, y_train, color="C0", alpha=0.5, s=6, label="Training data")
sort_order = np.argsort(X_train[:, 0])
x_sorted = X_train[sort_order]
plt.plot(
    x_sorted[:, 0],
    x_sinx(x_sorted[:, 0]),
    "k-",
    label="Baseline",
)
plt.plot(
    x_sorted[:, 0],
    x_sinx(x_sorted[:, 0]) - 0.8 * 3 * confidence_level,
    "k--",
    label=f"Uniform-noise interval (alpha={ALPHA})",
)
plt.plot(x_sorted[:, 0], x_sinx(x_sorted[:, 0]) + 0.8 * 3 * confidence_level, "k--")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Data")
plt.legend()
plt.show()


##############################################################################
# 2. Model: polynomial regression
# --------------------------------------------------------------------------

polynomial_degree = 4
quantile_estimator = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=polynomial_degree)),
        ("linear", QuantileRegressor(solver="highs", alpha=0)),
    ]
)
estimator = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=polynomial_degree)),
        ("linear", LinearRegression()),
    ]
)


##############################################################################
# 3. Plotting and adaptivity comparison functions
# --------------------------------------------------------------------------

gaussian_centers = np.linspace(-1, 5, 8)
x_bins = np.array([-1, 0, 1.5, 3.0, 5.0])


def constant_feature_map(X):
    X = np.asarray(X)
    return np.ones((len(X), 1))


def binned_feature_map(X):
    x = np.asarray(X).reshape(-1)
    bin_indexes = np.digitize(x, x_bins[1:-1], right=False)
    features = np.zeros((len(x), len(x_bins) - 1))
    features[np.arange(len(x)), bin_indexes] = 1
    return features


def polynomial_feature_map(X):
    x = np.asarray(X).reshape(-1)
    return np.column_stack([np.ones(len(x)), x, x**2, x**3])


def gaussian_feature_map(X, sigma=0.55):
    x = np.asarray(X).reshape(-1)
    squared_distances = (x[:, np.newaxis] - gaussian_centers[np.newaxis, :]) ** 2
    return np.column_stack(
        [np.ones(len(x)), np.exp(-squared_distances / (2 * sigma**2))]
    )


def grouped_polynomial_feature_map(X):
    x = np.asarray(X).reshape(-1)
    left = (x < 0).astype(float)
    right = (x >= 0).astype(float)
    return np.column_stack(
        [
            left,
            right,
            right * x,
            right * x**2,
            right * x**3,
        ]
    )


def plot_subplot(
    ax,
    X,
    y,
    mapie,
    y_pred,
    y_pi,
    color_rgb,
    show_transform=False,
    ax_transform=None,
):
    """Plot the prediction interval and, optionally, feature-map components."""
    sort_order = np.argsort(X[:, 0])
    color = mcolors.rgb2hex(color_rgb)
    x_sorted = X[sort_order]
    y_sorted = y[sort_order]
    y_pred_sorted = y_pred[sort_order]
    lower_pi_sorted = y_pi[sort_order, 0, 0]
    upper_pi_sorted = y_pi[sort_order, 1, 0]

    ax.scatter(
        x_sorted[:, 0],
        y_sorted,
        s=3,
        alpha=0.3,
        color="darkblue",
        label="Test data",
    )
    ax.plot(x_sorted[:, 0], y_pred_sorted, lw=1, color="black", label="Prediction")
    ax.fill_between(
        x_sorted[:, 0],
        lower_pi_sorted,
        upper_pi_sorted,
        color=color,
        alpha=0.3,
        label="Prediction interval",
    )
    ax.plot(x_sorted[:, 0], lower_pi_sorted, lw=1, color=color)
    ax.plot(x_sorted[:, 0], upper_pi_sorted, lw=1, color=color)
    ax.plot(
        x_sorted[:, 0],
        test_pi[sort_order, 0],
        "--k",
        lw=1,
        label="True interval",
    )
    ax.plot(x_sorted[:, 0], test_pi[sort_order, 1], "--k", lw=1)

    if show_transform and isinstance(mapie, ConditionalSplitConformalRegressor):
        transform = mapie.feature_map(x_sorted)
        for column in range(transform.shape[1]):
            ax_transform.plot(x_sorted[:, 0], transform[:, column], lw=1, color=color)


def plot_figure(mapies, y_preds, y_pis, titles, show_components=False):
    """Plot the prediction intervals of all MAPIE instances."""
    cp = plt.get_cmap("tab10").colors
    ncols = min(3, len(titles))
    nrows = int(np.ceil(len(titles) / ncols))

    if show_components:
        fig, axes = plt.subplots(
            nrows=2 * nrows,
            ncols=ncols,
            figsize=(ncols * 4, nrows * 5.2),
            height_ratios=[3, 1] * nrows,
        )
        axes = np.asarray(axes).reshape(2 * nrows, ncols)
        main_axes = axes[::2].flatten()
        transform_axes = axes[1::2].flatten()
    else:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4)
        )
        main_axes = np.asarray(axes).reshape(nrows, ncols).flatten()
        transform_axes = np.full(main_axes.shape, None)

    for axis_index in range(len(mapies), len(main_axes)):
        fig.delaxes(main_axes[axis_index])
        if transform_axes[axis_index] is not None:
            fig.delaxes(transform_axes[axis_index])

    for index, (m_ax, t_ax, mapie, y_pred, y_pi, title) in enumerate(
        zip(main_axes, transform_axes, mapies, y_preds, y_pis, titles)
    ):
        plot_subplot(
            m_ax,
            X_test,
            y_test,
            mapie,
            y_pred,
            y_pi,
            cp[index],
            show_transform=show_components,
            ax_transform=t_ax,
        )
        m_ax.set_title(title)
        m_ax.set_xlabel("X")
        if index % ncols == 0:
            m_ax.set_ylabel("Y")
        m_ax.legend(fontsize=8)
        if t_ax is not None:
            t_ax.set_title("Feature-map components")
            t_ax.set_xlabel("X")
            if index % ncols == 0:
                t_ax.set_ylabel("Value")

    fig.tight_layout()
    plt.show()


def compute_conditional_coverage(X, y, y_pi, bins_width=0.5):
    """Compute conditional coverage on bins of X."""
    bin_edges = np.arange(np.min(X), np.max(X) + bins_width, bins_width)
    coverage = np.zeros(len(bin_edges) - 1)

    for bin_index in range(len(bin_edges) - 1):
        in_bin = (X[:, 0] >= bin_edges[bin_index]) & (
            X[:, 0] < bin_edges[bin_index + 1]
        )
        if np.any(in_bin):
            coverage[bin_index] = np.mean(
                (y[in_bin] >= y_pi[in_bin, 0, 0]) & (y[in_bin] <= y_pi[in_bin, 1, 0])
            )
        else:
            coverage[bin_index] = np.nan

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, coverage


def plot_evaluation(titles, y_pis, X, y):
    """Plot conditional coverages and interval widths."""
    sort_order = np.argsort(X[:, 0])
    cp = plt.get_cmap("tab10").colors

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    for index, pi in enumerate(y_pis):
        color = mcolors.rgb2hex(cp[index])
        bin_centers, coverage = compute_conditional_coverage(X, y, pi)
        axs[0].plot(bin_centers, coverage, lw=2, color=color, label=titles[index])
        width = pi[sort_order, 1, 0] - pi[sort_order, 0, 0]
        axs[1].plot(X[sort_order, 0], width, lw=2, color=color, label=titles[index])

    perfect_width = test_pi[sort_order, 1] - test_pi[sort_order, 0]
    axs[1].plot(
        X[sort_order, 0],
        perfect_width,
        lw=2,
        color="black",
        linestyle="--",
        label="True width",
    )
    axs[0].axhline(
        y=confidence_level,
        color="black",
        linestyle="--",
        label=f"target={confidence_level}",
    )
    axs[0].legend(fontsize=8)
    axs[0].set_title("Conditional coverage")
    axs[0].set_xlabel("X (bins of 0.5 width)")
    axs[0].set_ylabel("Coverage")
    axs[0].set_ylim([0.55, 1.05])
    axs[1].legend(fontsize=8)
    axs[1].set_title("Prediction interval width")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Width")
    plt.tight_layout()
    plt.show()


##############################################################################
# 4. Creation of MAPIE instances
# --------------------------------------------------------------------------
# We are going to test different methods: ``CV+``, ``CQR``, ``cross CQR``,
# ``Conditional`` (with default parameters), and the ``residual_normalized``
# conformity score.

estimator_split = clone(estimator).fit(X_train, y_train)
mapie_split = SplitConformalRegressor(
    estimator=estimator_split,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
).conformalize(X_calib, y_calib)
y_pred_split, y_pi_split = mapie_split.predict_interval(X_test)

mapie_cv = CrossConformalRegressor(
    estimator=clone(estimator),
    confidence_level=confidence_level,
    method="plus",
    cv=3,
)
mapie_cv.fit_conformalize(
    np.vstack([X_train, X_calib]),
    np.hstack([y_train, y_calib]),
)
y_pred_cv, y_pi_cv = mapie_cv.predict_interval(X_test)

mapie_cqr = ConformalizedQuantileRegressor(
    estimator=clone(quantile_estimator),
    confidence_level=confidence_level,
)
mapie_cqr.fit(X_train, y_train).conformalize(X_calib, y_calib)
y_pred_cqr, y_pi_cqr = mapie_cqr.predict_interval(X_test)

mapie_cross_cqr = CrossConformalizedQuantileRegressor(
    estimator=clone(quantile_estimator),
    confidence_level=confidence_level,
    method="plus",
    cv=3,
    random_state=random_state,
)
mapie_cross_cqr.fit_conformalize(
    np.vstack([X_train, X_calib]),
    np.hstack([y_train, y_calib]),
)
y_pred_cross_cqr, y_pi_cross_cqr = mapie_cross_cqr.predict_interval(X_test)

mapie_residual = SplitConformalRegressor(
    estimator=estimator_split,
    confidence_level=confidence_level,
    conformity_score="residual_normalized",
    prefit=True,
).conformalize(X_calib, y_calib)
y_pred_residual, y_pi_residual = mapie_residual.predict_interval(X_test)

mapie_conditional = ConditionalSplitConformalRegressor(
    gaussian_feature_map,
    estimator=estimator_split,
    confidence_level=confidence_level,
    conformity_score="absolute",
    prefit=True,
).conformalize(X_calib, y_calib)
y_pred_conditional, y_pi_conditional = mapie_conditional.predict_interval(X_test)

mapies = [
    mapie_split,
    mapie_cv,
    mapie_cqr,
    mapie_cross_cqr,
    mapie_residual,
    mapie_conditional,
]
y_preds = [
    y_pred_split,
    y_pred_cv,
    y_pred_cqr,
    y_pred_cross_cqr,
    y_pred_residual,
    y_pred_conditional,
]
y_pis = [
    y_pi_split,
    y_pi_cv,
    y_pi_cqr,
    y_pi_cross_cqr,
    y_pi_residual,
    y_pi_conditional,
]
titles = [
    "Basic split",
    "CV+",
    "CQR",
    "Cross CQR+",
    "Residual normalized",
    "Conditional - Gaussian feature map",
]

plot_figure(mapies, y_preds, y_pis, titles)
plot_evaluation(titles, y_pis, X_test, y_test)


##############################################################################
# The ``ConditionalSplitConformalRegressor``
# is a very adaptive method, even with default
# parameter values. If the dataset is more complex, the default parameters
# may not be enough to get the best performance. In this case, we can use
# more advanced settings, described below.


##############################################################################
# 5. How to improve the results?
# --------------------------------------------------------------------------
#
# 5.1. How does the conditional method work?
# --------------------------------------------------------------------------
# The conditional method is based on a function which creates some features (vector of
# d dimensions), based on ``X``.
#
# These features should be able to represent the distribution of the
# conformity scores, which is here (by default) the absolute residual:
# ``|y_true - y_pred|``

##############################################################################
# Examples of basic functions:
# --------------------------------------------------------------------------
#
##############################################################################

##############################################################################
#  1) ``f : X -> (1)``, will try to estimate the absolute residual with a
#  constant, and will result in a prediction interval of constant width
#  (like the basic split CP)
#
#  2) ``f : X -> (1, X)``, will result in a prediction interval of width
#  equal to: a constant + a value proportional to the value of ``X``
#  (it seems a good idea here, as the uncertainty increases with ``X``)
#
#  3) ``f : X -> (1_{X in bin_1}, ..., 1_{X in bin_k})`` defines a simple
#  group-conditional feature map. It is useful when the subgroups of interest
#  are known in advance.
#
# In the current API, these are passed as ``feature_map`` callables.

feature_maps = [
    constant_feature_map,
    polynomial_feature_map,
    gaussian_feature_map,
    binned_feature_map,
    grouped_polynomial_feature_map,
]
feature_map_titles = [
    "Conditional - constant",
    "Conditional - polynomial",
    "Conditional - Gaussian",
    "Conditional - binned groups",
    "Conditional - grouped polynomial",
]

conditional_mapies = []
conditional_y_preds = []
conditional_y_pis = []
for feature_map in feature_maps:
    mapie = ConditionalSplitConformalRegressor(
        feature_map,
        estimator=estimator_split,
        confidence_level=confidence_level,
        conformity_score="absolute",
        prefit=True,
    ).conformalize(X_calib, y_calib)
    y_pred, y_pi = mapie.predict_interval(X_test)
    conditional_mapies.append(mapie)
    conditional_y_preds.append(y_pred)
    conditional_y_pis.append(y_pi)

plot_figure(
    conditional_mapies, conditional_y_preds, conditional_y_pis, feature_map_titles, True
)
plot_evaluation(feature_map_titles, conditional_y_pis, X_test, y_test)


##############################################################################
# 6. Conclusion:
# --------------------------------------------------------------------------
# The goal is to get prediction intervals that are as adaptive as possible while
# still keeping the target coverage. Perfect adaptivity would result in a
# perfectly constant conditional coverage.
#
# This is the power of the conditional method: use prior knowledge or
# generic features (Gaussian kernels) to have a great overall adaptivity.
#
# However, it can be difficult to find the best feature map.
# Sometimes, a simpler method can be enough. Don't forget to try at first
# the simpler method, and move on with the more advanced if it is necessary.
