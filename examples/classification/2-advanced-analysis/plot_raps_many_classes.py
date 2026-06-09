"""
RAPS method on a dataset with many classes
==========================================


In this tutorial, we compare the prediction sets estimated by
`SplitConformalClassifier` with the "lac", "aps" and "raps" conformity
scores on a synthetic dataset with many classes, where the benefit of
the RAPS regularization is the most visible.
"""

##############################################################################
# The Regularized Adaptive Prediction Sets (RAPS) method builds on the
# Adaptive Prediction Sets (APS) method: both include classes in the
# prediction set by decreasing order of estimated probability, until the
# cumulated probability exceeds a calibrated quantile. With a large number
# of classes, the tail of the estimated probability distribution is noisy,
# and the APS method may need to include many unlikely classes to reach the
# quantile, hence producing occasionally very large prediction sets.
#
# RAPS adds a penalty term `λ (k - k_reg)⁺` to the cumulated probability of
# the k-th class, which discourages the inclusion of classes beyond the
# optimal set size `k_reg`. The parameters `λ` and `k_reg` are chosen
# automatically by MAPIE on a fraction of the conformalization data (set by
# the `size_raps` parameter of `RAPSConformityScore`, 20% by default).
#
# Note that, in MAPIE, the "raps" conformity score is only available with
# `SplitConformalClassifier`.

# Reference:
# Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan
# and Jitendra Malik.
# "Uncertainty Sets for Image Classifiers using Conformal Prediction."
# International Conference on Learning Representations 2021.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
)
from mapie.utils import train_conformalize_test_split

RANDOM_STATE = 42

##############################################################################
# 1. Building a classification problem with many classes
# ------------------------------------------------------
#
# We generate a synthetic dataset with 50 classes and a moderate class
# separation, so that the classifier is often uncertain among several
# classes. We split the data into a training set (to fit the model),
# a conformalization set (to compute the conformity scores) and a test
# set (to evaluate coverage and prediction set sizes).

X, y = make_classification(
    n_samples=20000,
    n_features=30,
    n_informative=20,
    n_classes=50,
    class_sep=2.5,
    random_state=RANDOM_STATE,
)

(X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.5,
        conformalize_size=0.3,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
print(f"Accuracy of the base classifier: {clf.score(X_test, y_test):.3f}")

##############################################################################
# The accuracy of the base classifier is rather low: point predictions are
# not reliable on this problem, which is precisely the situation where
# prediction sets are useful.
#
# 2. Comparing the "lac", "aps" and "raps" conformity scores
# ----------------------------------------------------------
#
# We now conformalize the pre-fitted classifier with the three conformity
# scores, at two confidence levels (90% and 95%), and compute the
# prediction sets on the test set.

confidence_levels = [0.9, 0.95]
conformity_scores = ["lac", "aps", "raps"]

y_pred_sets = {}
for conformity_score in conformity_scores:
    mapie_classifier = SplitConformalClassifier(
        estimator=clf,
        confidence_level=confidence_levels,
        conformity_score=conformity_score,
        prefit=True,
        random_state=RANDOM_STATE,
    )
    mapie_classifier.conformalize(X_conformalize, y_conformalize)
    _, y_pred_sets[conformity_score] = mapie_classifier.predict_set(X_test)

coverages = {
    conformity_score: classification_coverage_score(y_test, y_pred_set)
    for conformity_score, y_pred_set in y_pred_sets.items()
}
mean_widths = {
    conformity_score: classification_mean_width_score(y_pred_set)
    for conformity_score, y_pred_set in y_pred_sets.items()
}

for i, confidence_level in enumerate(confidence_levels):
    print(f"Confidence level {confidence_level}:")
    for conformity_score in conformity_scores:
        print(
            f"  {conformity_score:>4} - effective coverage: "
            f"{coverages[conformity_score][i]:.3f}, "
            f"average set size: {mean_widths[conformity_score][i]:.2f}"
        )

##############################################################################
# Let us visualize the effective coverages and the average prediction set
# sizes obtained with each conformity score.

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(confidence_levels))
bar_width = 0.25
for j, conformity_score in enumerate(conformity_scores):
    offset = (j - 1) * bar_width
    axs[0].bar(
        x + offset,
        coverages[conformity_score],
        bar_width,
        label=conformity_score,
    )
    axs[1].bar(
        x + offset,
        mean_widths[conformity_score],
        bar_width,
        label=conformity_score,
    )
for i, confidence_level in enumerate(confidence_levels):
    axs[0].hlines(
        confidence_level,
        x[i] - 2 * bar_width,
        x[i] + 2 * bar_width,
        color="black",
        linestyles="dashed",
        label="target coverage" if i == 0 else None,
    )
axs[0].set_xticks(x, [str(cl) for cl in confidence_levels])
axs[0].set_xlabel("Confidence level")
axs[0].set_ylabel("Effective coverage")
axs[0].set_title("Effective coverage")
axs[0].legend(loc="lower right")
axs[1].set_xticks(x, [str(cl) for cl in confidence_levels])
axs[1].set_xlabel("Confidence level")
axs[1].set_ylabel("Average size of prediction sets")
axs[1].set_title("Average size of prediction sets")
axs[1].legend()
plt.show()

##############################################################################
# All three conformity scores reach the target coverage, with comparable
# average set sizes. The "lac" sets are the smallest on average (it is
# known to be optimal in this respect), and the "aps" sets are the largest,
# the price to pay for their adaptivity.
#
# 3. Distribution of the prediction set sizes
# -------------------------------------------
#
# The average size does not tell the whole story: RAPS is specifically
# designed to avoid the occasionally very large sets produced by APS.
# Let us look at the distribution of the prediction set sizes at the 90%
# confidence level.

set_sizes = {
    conformity_score: y_pred_sets[conformity_score][:, :, 0].sum(axis=1)
    for conformity_score in conformity_scores
}

max_size = max(sizes.max() for sizes in set_sizes.values())
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(0, max_size + 2) - 0.5
for conformity_score in conformity_scores:
    ax.hist(
        set_sizes[conformity_score],
        bins=bins,
        alpha=0.5,
        label=conformity_score,
    )
ax.set_xlabel("Size of the prediction set")
ax.set_ylabel("Number of test samples")
ax.set_title("Distribution of prediction set sizes (confidence level 0.9)")
ax.legend()
plt.show()

for conformity_score in conformity_scores:
    print(
        f"{conformity_score:>4} - maximum set size: "
        f"{int(set_sizes[conformity_score].max())}"
    )

##############################################################################
# The "aps" distribution has a long right tail: for the most ambiguous test
# points, the prediction sets contain more than 20 of the 50 classes. With
# "raps", the penalty term truncates this tail: the sets are never larger
# than a few classes, and their sizes are much more stable across test
# points, while preserving the target coverage.
#
# In summary, when the number of classes is large, the "raps" conformity
# score is an appealing alternative to "aps": it keeps the adaptivity of the
# prediction sets but regularizes their size, at the cost of a small fraction
# of the conformalization data being used to tune its parameters.
