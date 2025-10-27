"""
======================================================
LAC and APS methods explained
======================================================

In this tutorial, we compare the prediction sets estimated by
:class:`~mapie_v1.classification.SplitConformalClassifier` with the "lac"
and "aps" on the two-dimensional dataset presented
by Sadinle et al. (2019).
"""

##############################################################################
# We will use MAPIE to estimate a prediction set of several classes such that
# the probability that the true label of a new test point is included in the
# prediction set is always higher than the target confidence level :
# ``1 - α``.
# Throughout this tutorial, we compare two conformity scores :
# softmax score or cumulated softmax score.
# We start by using the softmax score or cumulated score output by the base
# classifier as the conformity score on a toy two-dimensional dataset.
# We estimate the prediction sets as follows :
#
# * First we generate a dataset with train, conformalization and test, the model
#   is fitted in the training set.
#
# * We set the conformal score ``Sᵢ = 𝑓̂(Xᵢ)ᵧᵢ``
#   from the softmax output of the true class or the cumulated score
#   (by decreasing order) for each sample in the conformalization set.
#
# * Then we define q̂ as being the
#   ``(n + 1)(1 - α) / n``
#   previous quantile of ``S₁, ..., Sₙ`` (this is essentially the
#   quantile α, but with a small sample correction).
#
# * Finally, for a new test data point (where ``Xₙ₊₁`` is known but
#   ``Yₙ₊₁`` is not), create a prediction set
#   ``C(Xₙ₊₁) = {y: 𝑓̂(Xₙ₊₁)ᵧ > q̂}`` which includes
#   all the classes with a sufficiently high conformity score.
#
# We use a two-dimensional dataset with three labels.
# The distribution of the data is a bivariate normal with diagonal covariance
# matrices for each label.

# Reference:
# Mauricio Sadinle, Jing Lei, and Larry Wasserman.
# "Least Ambiguous Set-Valued Classifiers With Bounded Error Levels."
# Journal of the American Statistical Association, 114:525, 223-234, 2019.

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from numpy.typing import NDArray
from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
)

centers = [(0, 3.5), (-2, 0), (2, 0)]
covs = [np.eye(2), np.eye(2) * 2, np.diag([5, 1])]
x_min, x_max, y_min, y_max, step = -6, 8, -6, 8, 0.1
n_samples = 500
n_classes = 3
np.random.seed(42)
X = np.vstack(
    [
        np.random.multivariate_normal(center, cov, n_samples)
        for center, cov in zip(centers, covs)
    ]
)
y = np.hstack([np.full(n_samples, i) for i in range(n_classes)])
X_train, X_conf, y_train, y_conf = train_test_split(X, y, test_size=0.3)

xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(x_min, x_max, step))
X_test = np.stack([xx.ravel(), yy.ravel()], axis=1)


##############################################################################
# Let's see our training data

colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
y_train_col = list(map(colors.get, y_train))
fig = plt.figure()
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    color=y_train_col,
    marker="o",
    s=10,
    edgecolor="k",
)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


##############################################################################
# We fit our training data with a Gaussian Naive Base estimator.
# Then, we initialize a :class:`~mapie_v1.classification.SplitConformalClassifier` with
# conformity scores ``"lac"`` and ``"aps"`` , using our pre-fitted estimator.
# Lastly, we compute the prediction sets with different alpha values using the
# ``conformalize`` and ``predict`` methods.

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)

conformity_scores = ["lac", "aps"]
mapie, y_pred_mapie, y_ps_mapie = {}, {}, {}
alpha = [0.2, 0.1, 0.05]
for conformity_score in conformity_scores:
    mapie[conformity_score] = SplitConformalClassifier(
        estimator=clf,
        confidence_level=1 - np.array(alpha),
        conformity_score=conformity_score,
        prefit=True,
        random_state=42,
    )
    mapie[conformity_score].conformalize(X_conf, y_conf)
    y_pred_mapie[conformity_score], y_ps_mapie[conformity_score] = mapie[
        conformity_score
    ].predict_set(X_test, conformity_score_params={"include_last_label": True})


##############################################################################
# MAPIE produces two outputs:
#
# - y_pred_mapie: the prediction in the test set given by the base estimator.
#
# - y_ps_mapie: the prediction sets estimated by MAPIE using the selected
#   method.
#
# Let's now visualize the distribution of the conformity scores with the two
# methods with the calculated quantiles for the three alpha values.


def plot_scores(
    alphas: List[float],
    scores: NDArray,
    quantiles: NDArray,
    conformity_score: str,
    ax: plt.Axes,
) -> None:
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    ax.hist(scores, bins="auto")
    i = 0
    for quantile in quantiles:
        ax.vlines(
            x=quantile,
            ymin=0,
            ymax=500,
            color=colors[i],
            linestyles="dashed",
            label=f"alpha = {alphas[i]}",
        )
        i = i + 1
    ax.set_title(f"Distribution of scores for '{conformity_score}' method")
    ax.legend()
    ax.set_xlabel("scores")
    ax.set_ylabel("count")


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, conformity_score in enumerate(conformity_scores):
    conf_scores = mapie[conformity_score]._mapie_classifier.conformity_scores_
    n = mapie[conformity_score]._mapie_classifier.n_samples_
    quantiles = mapie[
        conformity_score
    ]._mapie_classifier.conformity_score_function_.quantiles_
    plot_scores(alpha, conf_scores, quantiles, conformity_score, axs[i])
plt.show()


##############################################################################
# We will now compare the differences between the prediction sets of the
# different values ​​of alpha.


def plot_results(
    alphas: List[float], y_pred_mapie: NDArray, y_ps_mapie: NDArray
) -> None:
    tab10 = plt.cm.get_cmap("Purples", 4)
    colors = {
        0: "#1f77b4",
        1: "#ff7f0e",
        2: "#2ca02c",
        3: "#d62728",
        4: "#c896af",
        5: "#94a98a",
        6: "#8a94a9",
        7: "#a99f8a",
        8: "#1e1b16",
        9: "#4a4336",
    }
    y_pred_col = list(map(colors.get, y_pred_mapie))
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 10))
    axs = {0: ax1, 1: ax2, 2: ax3, 3: ax4}
    axs[0].scatter(
        X_test[:, 0],
        X_test[:, 1],
        color=y_pred_col,
        marker=".",
        s=10,
        alpha=0.4,
    )
    axs[0].set_title("Predicted labels")
    for i, alpha_ in enumerate(alphas):
        y_pi_sums = y_ps_mapie[:, :, i].sum(axis=1)
        num_labels = axs[i + 1].scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_pi_sums,
            marker="o",
            s=10,
            alpha=1,
            cmap=tab10,
            vmin=0,
            vmax=3,
        )
        plt.colorbar(num_labels, ax=axs[i + 1])
        axs[i + 1].set_title(f"Number of labels for alpha={alpha_}")
    plt.show()


for conformity_score in conformity_scores:
    plot_results(alpha, y_pred_mapie[conformity_score], y_ps_mapie[conformity_score])


##############################################################################
# For the "lac" method, when the class coverage is not large enough, the
# prediction sets can be empty when the model is uncertain at the border
# between two labels. These null regions disappear for larger class coverages
# but ambiguous classification regions arise with several labels included in
# the prediction sets.
# By definition, the "aps" method does not produce empty
# prediction sets. However, the prediction sets tend to be slightly bigger
# in ambiguous regions.
#
# Let's now compare the effective coverage and the average of prediction set
# widths as function of the ``1 - α`` target coverage.

alpha_ = np.arange(0.02, 0.98, 0.02)
coverage, mean_width = {}, {}
mapie, y_ps_mapie = {}, {}
for conformity_score in conformity_scores:
    mapie[conformity_score] = SplitConformalClassifier(
        estimator=clf,
        confidence_level=1 - alpha_,
        conformity_score=conformity_score,
        prefit=True,
        random_state=42,
    )
    mapie[conformity_score].conformalize(X_conf, y_conf)
    _, y_ps_mapie[conformity_score] = mapie[conformity_score].predict_set(
        X, conformity_score_params={"include_last_label": "randomized"}
    )
    coverage[conformity_score] = [
        classification_coverage_score(y, y_ps_mapie[conformity_score])
    ]
    mean_width[conformity_score] = classification_mean_width_score(
        y_ps_mapie[conformity_score]
    )

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].set_xlabel("1 - alpha")
axs[0].set_ylabel("Quantile")
for conformity_score in conformity_scores:
    quantiles = mapie[
        conformity_score
    ]._mapie_classifier.conformity_score_function_.quantiles_
    axs[0].scatter(1 - alpha_, quantiles, label=conformity_score)
axs[0].legend()
for conformity_score in conformity_scores:
    axs[1].scatter(1 - alpha_, coverage[conformity_score], label=conformity_score)
axs[1].set_xlabel("1 - alpha")
axs[1].set_ylabel("Coverage score")
axs[1].plot([0, 1], [0, 1], label="x=y", color="black")
axs[1].legend()
for conformity_score in conformity_scores:
    axs[2].scatter(1 - alpha_, mean_width[conformity_score], label=conformity_score)
axs[2].set_xlabel("1 - alpha")
axs[2].set_ylabel("Average size of prediction sets")
axs[2].legend()
plt.show()

##############################################################################
# It is seen that both methods give coverages close to the target coverages,
# regardless of the ``α`` value. However, the "aps"
# produces slightly bigger prediction sets, but without empty regions
# (if the selection of the last label is not randomized).
