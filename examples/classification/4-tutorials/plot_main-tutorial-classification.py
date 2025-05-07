"""
===========================
Tutorial for classification
===========================


In this tutorial, we compare the prediction sets estimated by the conformal
methods implemented in MAPIE on a toy two-dimensional dataset.

Throughout this tutorial, we will answer the following questions:

- How does the number of classes in the prediction sets vary according to
  the confidence level?

- Is the chosen conformal method well calibrated?

- What are the pros and cons of the conformal methods included in MAPIE?
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

from mapie.classification import SplitConformalClassifier
from mapie.utils import train_conformalize_test_split
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
)

##############################################################################
# 1. Conformal Prediction method using the softmax score of the true label
# ------------------------------------------------------------------------
#
# We will use MAPIE to estimate a prediction set of several classes such
# that the probability that the true label of a new test point is included
# in the prediction set is always higher than the target confidence level :
# ``P(Yₙ₊₁ ∈ Ĉₙ,α(Xₙ₊₁)) ≥ 1 - α``.
# We start by using the softmax score output by the base classifier as the
# conformity score on a toy two-dimensional dataset.
#
# We estimate the prediction sets as follows :
#
# * Generate a dataset with train, conformalization and test, the model is
#   fitted on the training set.
#
# * Set the conformal score ``Sᵢ = 𝑓̂(Xᵢ)ᵧᵢ``, the softmax
#   output of the true class for each sample in the conformity set.
#
# * Define ``q̂`` as being the ``(n + 1)(α) / n``
#   previous quantile of ``S₁, ..., Sₙ``
#   (this is essentially the quantile ``α``, but with a small sample
#   correction).
#
# * Finally, for a new test data point (where ``Xₙ₊₁`` is known but
#   ``Yₙ₊₁`` is not), create a prediction set
#   ``C(Xₙ₊₁) = {y: 𝑓̂(Xₙ₊₁)ᵧ > q̂}`` which includes
#   all the classes with a sufficiently high softmax output.

# We use a two-dimensional toy dataset with three labels. The distribution of
# the data is a bivariate normal with diagonal covariance matrices for each
# label.

centers = [(0, 3.5), (-2, 0), (2, 0)]
covs = [np.eye(2), np.eye(2)*2, np.diag([5, 1])]
x_min, x_max, y_min, y_max, step = -6, 8, -6, 8, 0.1
n_samples = 1000
n_classes = 3
np.random.seed(42)
X = np.vstack([
    np.random.multivariate_normal(center, cov, n_samples)
    for center, cov in zip(centers, covs)
])
y = np.hstack([np.full(n_samples, i) for i in range(n_classes)])
(X_train, X_conf, X_test,
 y_train, y_conf, y_test) = train_conformalize_test_split(
    X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2
)

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, step), np.arange(x_min, x_max, step)
)
X_test_mesh = np.stack([xx.ravel(), yy.ravel()], axis=1)

##############################################################################
# Let’s see our training data.

colors = {0: "#1f77b4", 1: "#ff7f0e", 2:  "#2ca02c", 3: "#d62728"}
y_train_col = list(map(colors.get, y_train))
fig = plt.figure()
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    color=y_train_col,
    marker='o',
    s=10,
    edgecolor='k'
)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

##############################################################################
# We fit our training data with a Gaussian Naive Base estimator. And then we
# apply MAPIE in the conformity data with the LAC conformity score to the
# estimator indicating that it has already been fitted with `prefit=True`.
# We then estimate the prediction sets with different confidence level values with a
# ``conformalize`` and ``predict`` process.

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)
confidence_level = [0.8, 0.9, 0.95]
mapie_score = SplitConformalClassifier(
    estimator=clf,
    confidence_level=confidence_level,
    prefit=True
)
mapie_score.conformalize(X_conf, y_conf)
y_pred_score, y_ps_score = mapie_score.predict_set(X_test_mesh)

##############################################################################
# * ``y_pred_score``: represents the prediction in the test set by the base
#   estimator.
# * ``y_ps_score``: represents the prediction sets estimated by MAPIE with
#   the "lac" conformity score.


def plot_scores(n, confidence_levels, scores, quantiles):
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins="auto")
    for i, quantile in enumerate(quantiles):
        plt.vlines(
            x=quantile,
            ymin=0,
            ymax=400,
            color=colors[i],
            ls="dashed",
            label=f"confidence_level = {confidence_levels[i]}"
        )
    plt.title("Distribution of scores")
    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.show()


##############################################################################
# Let’s see the distribution of the scores with the calculated quantiles.

scores = mapie_score._mapie_classifier.conformity_scores_
n = len(mapie_score._mapie_classifier.conformity_scores_)
quantiles = mapie_score._mapie_classifier.conformity_score_function_.quantiles_
plot_scores(n, confidence_level, scores, quantiles)

##############################################################################
# The estimated quantile increases with the confidence level.
# A low confidence level can potentially lead to a low quantile ``q``; the associated
# ``1 - q`` threshold would therefore not necessarily be reached by any class in
# uncertain areas, resulting in null regions.
#
# We will now visualize the differences between the prediction sets of the
# different values of confidence level.


def plot_results(confidence_levels, X, y_pred, y_ps):
    tab10 = plt.cm.get_cmap('Purples', 4)
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2:  "#2ca02c", 3: "#d62728"}
    y_pred_col = list(map(colors.get, y_pred))
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 10))
    axs = {0: ax1, 1: ax2, 2:  ax3, 3: ax4}
    axs[0].scatter(
        X[:, 0],
        X[:, 1],
        color=y_pred_col,
        marker='.',
        s=10,
        alpha=0.4
    )
    axs[0].set_title("Predicted labels")
    for i, confidence_level in enumerate(confidence_levels):
        y_pi_sums = y_ps[:, :, i].sum(axis=1)
        num_labels = axs[i+1].scatter(
            X[:, 0],
            X[:, 1],
            c=y_pi_sums,
            marker='.',
            s=10,
            alpha=1,
            cmap=tab10,
            vmin=0,
            vmax=3
        )
        plt.colorbar(num_labels, ax=axs[i+1])
        axs[i+1].set_title(f"Number of labels for confidence_level={confidence_level}")
    plt.show()


plot_results(confidence_level, X_test_mesh, y_pred_score, y_ps_score)

##############################################################################
# When the class coverage is not large enough, the prediction sets can be
# empty when the model is uncertain at the border between two classes.
# The null region disappears for larger class coverages but ambiguous
# classification regions arise with several labels included in the
# prediction sets highlighting the uncertain behaviour of the base
# classifier.
#
# Let’s now study the effective coverage and the mean prediction set widths
# as function of the ``confidence_level`` target coverage. To this aim, we use once
# again the ``predict`` method of MAPIE to estimate predictions sets on a
# large number of ``confidence_level`` values.

confidence_level2 = np.arange(0.02, 0.98, 0.02)
mapie_score2 = SplitConformalClassifier(
    estimator=clf,
    confidence_level=confidence_level2,
    prefit=True
)
mapie_score2.conformalize(X_conf, y_conf)
_, y_ps_score2 = mapie_score2.predict_set(X_test)
coverages_score = classification_coverage_score(y_test, y_ps_score2)
widths_score = classification_mean_width_score(y_ps_score2)


def plot_coverages_widths(confidence_level, coverage, width, conformity_score):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(confidence_level, coverage, label=conformity_score)
    axs[0].set_xlabel("Confidence level")
    axs[0].set_ylabel("Coverage score")
    axs[0].plot([0, 1], [0, 1], label="x=y", color="black")
    axs[0].legend()
    axs[1].scatter(confidence_level, width, label=conformity_score)
    axs[1].set_xlabel("Confidence level")
    axs[1].set_ylabel("Average size of prediction sets")
    axs[1].legend()
    plt.show()


plot_coverages_widths(
    confidence_level2, coverages_score, widths_score, "lac"
)

##############################################################################
# 2. Conformal Prediction method using the cumulative softmax score
# -----------------------------------------------------------------
#
# We saw in the previous section that the "lac" conformity score is well calibrated by
# providing accurate coverage levels. However, it tends to give null
# prediction sets for uncertain regions, especially when the ``confidence_level``
# value is low.
# MAPIE includes another method, called Adaptive Prediction Set (APS),
# whose conformity score is the cumulated score of the softmax output until
# the true label is reached (see the theoretical description for more details).
# We will see in this section that this method no longer estimates null
# prediction sets by giving slightly bigger prediction sets.
#
# Let's visualize the prediction sets obtained with the APS method on the test
# set after fitting MAPIE on the conformity set.

confidence_level = [0.8, 0.9, 0.95]
mapie_aps = SplitConformalClassifier(
    estimator=clf,
    confidence_level=confidence_level,
    conformity_score="aps",
    prefit=True
)
mapie_aps.conformalize(X_conf, y_conf)
y_pred_aps, y_ps_aps = mapie_aps.predict_set(
    X_test_mesh, conformity_score_params={"include_last_label": True}
)

plot_results(confidence_level, X_test_mesh, y_pred_aps, y_ps_aps)

##############################################################################
# One can notice that the uncertain regions are emphasized by wider
# boundaries,  but without null prediction sets with respect to the first
# "lac" method.

mapie_aps2 = SplitConformalClassifier(
    estimator=clf,
    confidence_level=confidence_level2,
    conformity_score="aps",
    prefit=True
)
mapie_aps2.conformalize(X_conf, y_conf)
_, y_ps_aps2 = mapie_aps2.predict_set(
    X_test, conformity_score_params={"include_last_label": "randomized"}
)
coverages_aps = classification_coverage_score(y_test, y_ps_aps2)
widths_aps = classification_mean_width_score(y_ps_aps2)

plot_coverages_widths(
    confidence_level2, coverages_aps, widths_aps, "aps"
)

##############################################################################
# This method also gives accurate conformalization plots, meaning that the
# effective coverage level is always very close to the target coverage,
# sometimes at the expense of slightly bigger prediction sets.
