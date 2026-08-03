"""
Group-conditional prediction sets (advanced)
============================================

The tutorial explains how to use ``ConditionalSplitConformalClassifier`` for
classification. In particular, a Gaussian feature map is compared to groups
defined from the predicted class.

In this tutorial, the classifier will be
``LogisticRegression``.
We will use a synthetic toy dataset.

We will compare the conditional method
with the standard method, using for both, the LAC conformity score
(``LACConformityScore``).

Recall that the ``LAC`` method consists of applying a threshold to the
predicted class probabilities, to keep in the set all the classes with predicted
probabilities above the threshold.
"""

# mkdocs_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression

from mapie.classification import SplitConformalClassifier
from mapie.conditional_conformal_prediction import ConditionalSplitConformalClassifier

random_state = 1
CONFIDENCE_LEVEL = 0.8  # 1 - alpha
N_CLASSES = 5

##############################################################################
# 1. Data generation
# --------------------------------------------------------------------------
# Let's start by creating some synthetic data with 5 Gaussian distributions.

centers = np.array(
    [
        [0.0, 3.5],
        [-3.0, 0.0],
        [0.0, -2.0],
        [4.0, -1.0],
        [3.0, 1.0],
    ]
)
covariances = [
    np.diag([1.0, 1.0]),
    np.diag([2.0, 2.0]),
    np.diag([3.0, 2.0]),
    np.diag([3.0, 3.0]),
    np.diag([2.0, 2.0]),
]


def create_toy_dataset(n_samples=1000, seed=1):
    rng = np.random.default_rng(seed)
    n_per_class = np.full(N_CLASSES, n_samples // N_CLASSES)
    n_per_class[: n_samples % N_CLASSES] += 1
    X = np.vstack(
        [
            rng.multivariate_normal(center, covariance, n)
            for center, covariance, n in zip(centers, covariances, n_per_class)
        ]
    )
    y = np.hstack(
        [np.full(n, class_index) for class_index, n in enumerate(n_per_class)]
    )
    permutation = rng.permutation(len(y))
    return X[permutation], y[permutation]


def generate_data(seed=1, n_train=1000, n_calib=700, n_test=1000):
    x_train, y_train = create_toy_dataset(n_train, seed=seed)
    x_calib, y_calib = create_toy_dataset(n_calib, seed=seed + 1)
    x_test, y_test = create_toy_dataset(n_test, seed=seed + 2)
    return x_train, y_train, x_calib, y_calib, x_test, y_test


x_train, y_train, *_ = generate_data(seed=random_state)

for class_index in range(N_CLASSES):
    plt.scatter(
        x_train[y_train == class_index, 0],
        x_train[y_train == class_index, 1],
        c=f"C{class_index}",
        s=3,
        label=f"Class {class_index}",
    )
plt.legend(markerscale=4)
plt.title("Synthetic multiclass data")
plt.xlabel("$X_0$")
plt.ylabel("$X_1$")
plt.show()


##############################################################################
# 2. Plotting and adaptivity comparison functions
# --------------------------------------------------------------------------
# The current API receives the conditional class of functions through a
# ``feature_map`` callable.


def predicted_class_feature_map(estimator):
    def feature_map(X):
        predicted_class = estimator.predict(X)
        return np.eye(N_CLASSES)[predicted_class]

    return feature_map


def gaussian_feature_map(X):
    X = np.asarray(X)
    squared_distances = ((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(
        axis=2
    )
    sigma = 1.5
    gaussian_features = np.exp(-squared_distances / (2 * sigma**2))
    return np.column_stack([np.ones(len(X)), gaussian_features])


def fit_methods(x_train, y_train, x_calib, y_calib):
    estimator = LogisticRegression(max_iter=1000).fit(x_train, y_train)

    mapie_lac = SplitConformalClassifier(
        estimator=estimator,
        confidence_level=CONFIDENCE_LEVEL,
        conformity_score="lac",
        prefit=True,
    ).conformalize(x_calib, y_calib)

    mapie_conditional_y_pred = ConditionalSplitConformalClassifier(
        predicted_class_feature_map(estimator),
        estimator=estimator,
        confidence_level=CONFIDENCE_LEVEL,
        conformity_score="lac",
        prefit=True,
    ).conformalize(x_calib, y_calib)

    mapie_conditional_gauss = ConditionalSplitConformalClassifier(
        gaussian_feature_map,
        estimator=estimator,
        confidence_level=CONFIDENCE_LEVEL,
        conformity_score="lac",
        prefit=True,
    ).conformalize(x_calib, y_calib)

    return [mapie_lac, mapie_conditional_y_pred, mapie_conditional_gauss]


def evaluate_conditional_coverage(mapies, x_test, y_test):
    scores = np.zeros((len(mapies), N_CLASSES + 1))
    for method_index, mapie in enumerate(mapies):
        _, y_ps_test = mapie.predict_set(x_test)
        scores[method_index, 1:] = [
            y_ps_test[y_test == class_index, class_index, 0].mean()
            for class_index in range(N_CLASSES)
        ]
        scores[method_index, 0] = y_ps_test[np.arange(len(y_test)), y_test, 0].mean()
    return scores


def run_exp(
    names,
    n_train=1000,
    n_calib=700,
    n_test=1000,
    grid_step=26,
    plot=True,
    seed=1,
    max_display=1000,
):
    x_train, y_train, x_calib, y_calib, x_test, y_test = generate_data(
        seed=seed, n_train=n_train, n_calib=n_calib, n_test=n_test
    )
    mapies = fit_methods(x_train, y_train, x_calib, y_calib)

    if max_display:
        rng = np.random.default_rng(seed)
        display_ind = rng.choice(len(x_test), min(max_display, len(x_test)))
    else:
        display_ind = np.arange(len(x_test))

    if plot:
        fig = plt.figure(figsize=(6 * (len(mapies) + 1), 7))
        grid = plt.GridSpec(1, len(mapies) + 1)
        xx, yy = np.meshgrid(
            np.linspace(-6, 8, grid_step), np.linspace(-6, 8, grid_step)
        )
        x_mesh = np.stack([xx.ravel(), yy.ravel()], axis=1)
        color_map = plt.colormaps["Purples"].resampled(N_CLASSES + 1)

    if not plot:
        return evaluate_conditional_coverage(mapies, x_test, y_test)

    for method_index, (mapie, name) in enumerate(zip(mapies, names)):
        y_pred_mesh, y_ps_mesh = mapie.predict_set(x_mesh)

        if method_index == 0:
            ax = fig.add_subplot(grid[0, 0])
            ax.scatter(
                x_mesh[:, 0],
                x_mesh[:, 1],
                c=[f"C{x}" for x in y_pred_mesh],
                alpha=0.95,
                marker="s",
                edgecolor="none",
                s=24,
            )
            ax.scatter(
                x_test[display_ind, 0],
                x_test[display_ind, 1],
                c=[f"C{x}" for x in y_test[display_ind]],
                alpha=0.85,
                marker=".",
                edgecolor="black",
                s=60,
            )
            ax.set_title("Predictions")
            ax.set_xlim([-6, 8])
            ax.set_ylim([-6, 8])
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=".",
                    color="w",
                    markerfacecolor=f"C{i}",
                    markersize=10,
                )
                for i in range(N_CLASSES)
            ]
            ax.legend(handles, [f"Class {i}" for i in range(N_CLASSES)])

        y_ps_sums = y_ps_mesh[:, :, 0].sum(axis=1)
        ax = fig.add_subplot(grid[0, method_index + 1])
        scatter = ax.scatter(
            x_mesh[:, 0],
            x_mesh[:, 1],
            c=y_ps_sums,
            marker="s",
            edgecolor="none",
            s=24,
            cmap=color_map,
            vmin=0,
            vmax=N_CLASSES,
        )
        ax.scatter(
            x_test[display_ind, 0],
            x_test[display_ind, 1],
            c=[f"C{x}" for x in y_test[display_ind]],
            alpha=0.55,
            marker=".",
            edgecolor="gray",
            s=35,
        )
        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.ax.set_ylabel("Set size")
        ax.set_title(name)
        ax.set_xlim([-6, 8])
        ax.set_ylim([-6, 8])

    scores = evaluate_conditional_coverage(mapies, x_test, y_test)
    fig.tight_layout()
    plt.show()
    return scores


def plot_cond_coverage(scores, names):
    labels = [f"Class {i}" for i in range(N_CLASSES)]
    labels.insert(0, "marginal")
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for method_index in range(scores.shape[1]):
        ax.boxplot(
            scores[:, method_index, :],
            positions=x + width * (method_index - 1),
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor=f"C{method_index}"),
            medianprops=dict(color="black"),
            tick_labels=labels,
        )
    ax.axhline(
        y=CONFIDENCE_LEVEL,
        color="red",
        linestyle="--",
        label=f"target={CONFIDENCE_LEVEL}",
    )
    ax.axvline(x=0.5, color="black", linestyle="--")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage on each class")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0.6, 1])

    custom_handles = [
        Patch(facecolor=f"C{i}", edgecolor="black", label=names[i])
        for i in range(len(names))
    ]
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles + custom_handles, legend_labels + names, loc="lower left")
    plt.show()


##############################################################################
# 3. Creation of Mapie instances
# --------------------------------------------------------------------------
# We are going to compare the standard ``LAC`` method with:
#
# - The conditional method using the predicted classes as groups (to have a
#   homogeneous coverage on each class).
# - The conditional method with Gaussian kernels, to have adaptive prediction
#   sets, without prior knowledge or information.


##############################################################################
# 4. Generate the prediction sets
# --------------------------------------------------------------------------

names = [
    "Standard LAC",
    "Conditional with predicted class groups",
    "Conditional with Gaussian kernel",
]

scores = run_exp(names)

##############################################################################
# We can see that the conditional method seems to create better
# prediction sets than the standard method. Indeed, where the
# classes distributions overlap (especially for class 3 and 4),
# the size of the sets should increase, to correctly represent the model
# uncertainty on those samples.
#
# The middle of all the classes distributions, where points could
# belong to any class, should have the biggest prediction sets (with almost
# all the classes in the sets, as we are very uncertain). The feature map
# with Gaussian kernels represented this uncertainty, with big sets
# for the middle points.
#
# Thus, between the two conditional methods, the one using Gaussian kernels
# seems the most adaptive.


##############################################################################
# 5. Evaluate the adaptivity
# --------------------------------------------------------------------------
# While we can get a first sense of the adaptivity of the methods just by
# looking at the prediction sets, the most accurate way is to check whether the
# coverage is homogeneous on subparts of the data (on each class for instance).

N_TRIALS = 4
scores = np.zeros((N_TRIALS, len(names), N_CLASSES + 1))
for trial in range(N_TRIALS):
    scores[trial, :, :] = run_exp(names, plot=False, seed=trial)

plot_cond_coverage(scores, names)

##############################################################################
# A perfectly adaptive method would result in a homogeneous coverage
# for all classes. We can see that the conditional method, with the predicted
# classes as groups, is more adaptive than the standard method.
#
# To conclude, the conditional method offers adaptive prediction sets.
# We can inject prior knowledge or groups on which we want to avoid bias.
# Groups can be defined from different features from X, including the
# predicted class.
# Using Gaussian kernels, with a correct sigma parameter
# can be the easiest and best solution to have very adaptive prediction sets
# for this dataset.
