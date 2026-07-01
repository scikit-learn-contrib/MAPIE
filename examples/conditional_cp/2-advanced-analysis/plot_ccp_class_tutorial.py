"""
Tutorial: conditional conformal prediction for classification
=============================================================


This tutorial shows how to use
:class:`~mapie.conditional_conformal_prediction.ConditionalSplitConformalClassifier`
on a synthetic multiclass problem inspired by Gibbs, Cherian and Candès (2023)
[1].

The conditional classifier receives a ``feature_map`` function. This function
maps each input sample to a finite set of basis functions defining where the
conditional coverage guarantees should hold. Here we compare:

- a standard marginal split-conformal classifier;
- a conditional classifier whose ``feature_map`` groups samples by the base
  model's predicted class;
- a conditional classifier whose ``feature_map`` contains radial basis
  functions centered on the class clouds.

[1] Isaac Gibbs, John J. Cherian, and Emmanuel J. Candès,
"Conformal Prediction With Conditional Guarantees",
`arXiv <https://arxiv.org/abs/2305.12616>`_, 2023.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from mapie.classification import SplitConformalClassifier
from mapie.conditional_conformal_prediction import ConditionalSplitConformalClassifier
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
)

##############################################################################
# 1. Generate a multiclass toy dataset
# --------------------------------------------------------------------------
#
# The classes overlap near the center of the plot. In that region, prediction
# sets should become larger because the base classifier is less certain.

N_CLASSES = 5
confidence_level = 0.8

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


def make_toy_dataset(n_samples, random_state):
    rng = np.random.default_rng(random_state)
    n_per_class = np.full(N_CLASSES, n_samples // N_CLASSES)
    n_per_class[: n_samples % N_CLASSES] += 1
    X = np.vstack(
        [
            rng.multivariate_normal(center, covariance, n)
            for center, covariance, n in zip(centers, covariances, n_per_class)
        ]
    )
    y = np.concatenate(
        [np.full(n, class_index) for class_index, n in enumerate(n_per_class)]
    )
    permutation = rng.permutation(len(y))
    return X[permutation], y[permutation]


X_train, y_train = make_toy_dataset(1200, random_state=1)
X_conformalize, y_conformalize = make_toy_dataset(1200, random_state=2)
X_test, y_test = make_toy_dataset(1600, random_state=3)

fig, ax = plt.subplots(figsize=(6, 5))
for class_index in range(N_CLASSES):
    mask = y_train == class_index
    ax.scatter(
        X_train[mask, 0],
        X_train[mask, 1],
        s=8,
        alpha=0.55,
        label=f"Class {class_index}",
    )
ax.set_title("Multiclass data with overlapping class clouds")
ax.set_xlabel("$X_0$")
ax.set_ylabel("$X_1$")
ax.legend(markerscale=2)
plt.tight_layout()
plt.show()


##############################################################################
# 2. Define conditional feature maps
# --------------------------------------------------------------------------
#
# ``predicted_class_feature_map`` asks the conditional procedure to balance
# coverage across the base classifier's predicted classes. ``rbf_feature_map``
# instead gives a smooth local basis over the feature space.

base_classifier = LogisticRegression(max_iter=1000).fit(X_train, y_train)


def predicted_class_feature_map(X):
    predicted_class = base_classifier.predict(X)
    return np.eye(N_CLASSES)[predicted_class]


def rbf_feature_map(X):
    X = np.asarray(X)
    squared_distances = ((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(
        axis=2
    )
    rbf_values = np.exp(-squared_distances / (2 * 2.0**2))
    return np.column_stack([np.ones(len(X)), rbf_values])


##############################################################################
# 3. Fit marginal and conditional classifiers
# --------------------------------------------------------------------------
#
# All three methods use the same fitted logistic regression model and the
# same conformalization data. Only the conformal calibration step changes.

mapie_lac = SplitConformalClassifier(
    estimator=base_classifier,
    confidence_level=confidence_level,
    conformity_score="lac",
    prefit=True,
)
mapie_lac.conformalize(X_conformalize, y_conformalize)
y_pred_lac, y_set_lac = mapie_lac.predict_set(X_test)

mapie_predicted_class = ConditionalSplitConformalClassifier(
    predicted_class_feature_map,
    estimator=base_classifier,
    confidence_level=confidence_level,
    conformity_score="lac",
    prefit=True,
)
mapie_predicted_class.conformalize(X_conformalize, y_conformalize)
y_pred_pc, y_set_pc = mapie_predicted_class.predict_set(X_test)

mapie_rbf = ConditionalSplitConformalClassifier(
    rbf_feature_map,
    estimator=base_classifier,
    confidence_level=confidence_level,
    conformity_score="lac",
    prefit=True,
)
mapie_rbf.conformalize(X_conformalize, y_conformalize)
y_pred_rbf, y_set_rbf = mapie_rbf.predict_set(X_test)


##############################################################################
# 4. Visualize adaptivity of the prediction sets
# --------------------------------------------------------------------------
#
# The plots show the size of each prediction set on a feature-space grid. The
# conditional feature maps allocate larger sets near the overlapping regions.

xx, yy = np.meshgrid(np.linspace(-6, 8, 30), np.linspace(-6, 8, 30))
X_grid = np.column_stack([xx.ravel(), yy.ravel()])

_, y_set_grid_lac = mapie_lac.predict_set(X_grid)
_, y_set_grid_pc = mapie_predicted_class.predict_set(X_grid)
_, y_set_grid_rbf = mapie_rbf.predict_set(X_grid)

grid_sets = [y_set_grid_lac, y_set_grid_pc, y_set_grid_rbf]
titles = [
    "Marginal LAC",
    "Conditional: predicted-class groups",
    "Conditional: RBF feature map",
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
for ax, y_pred_set, title in zip(axes, grid_sets, titles):
    set_sizes = y_pred_set[:, :, 0].sum(axis=1).reshape(xx.shape)
    image = ax.pcolormesh(xx, yy, set_sizes, cmap="Purples", vmin=0, vmax=N_CLASSES)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="tab10", s=5, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("$X_0$")
axes[0].set_ylabel("$X_1$")
fig.colorbar(image, ax=axes, label="Prediction set size")
plt.show()


##############################################################################
# 5. Compare coverage by class
# --------------------------------------------------------------------------
#
# The marginal method targets coverage only on average. Conditional feature maps
# make it possible to target more local notions of coverage chosen by the user.


def scores_by_class(y_true, y_pred_set):
    coverages = []
    set_sizes = []
    for class_index in range(N_CLASSES):
        mask = y_true == class_index
        coverages.append(classification_coverage_score(y_true[mask], y_pred_set[mask]))
        set_sizes.append(classification_mean_width_score(y_pred_set[mask]))
    return np.asarray(coverages).ravel(), np.asarray(set_sizes).ravel()


coverage_lac, width_lac = scores_by_class(y_test, y_set_lac)
coverage_pc, width_pc = scores_by_class(y_test, y_set_pc)
coverage_rbf, width_rbf = scores_by_class(y_test, y_set_rbf)

x = np.arange(N_CLASSES)
bar_width = 0.25
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

for offset, coverage, label in [
    (-bar_width, coverage_lac, "Marginal LAC"),
    (0.0, coverage_pc, "Predicted-class groups"),
    (bar_width, coverage_rbf, "RBF feature map"),
]:
    axes[0].bar(x + offset, coverage, width=bar_width, label=label)
axes[0].axhline(confidence_level, color="black", linestyle="--", linewidth=1)
axes[0].set_ylim(0.55, 1.02)
axes[0].set_ylabel("Coverage")
axes[0].set_title("Coverage by true class")
axes[0].legend()

for offset, width, label in [
    (-bar_width, width_lac, "Marginal LAC"),
    (0.0, width_pc, "Predicted-class groups"),
    (bar_width, width_rbf, "RBF feature map"),
]:
    axes[1].bar(x + offset, width, width=bar_width, label=label)
axes[1].set_ylabel("Mean prediction set size")
axes[1].set_title("Set size by true class")

for ax in axes:
    ax.set_xlabel("Class")
    ax.set_xticks(x)

plt.tight_layout()
plt.show()
