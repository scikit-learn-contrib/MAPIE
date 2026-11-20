"""
RAPS: Regularized Adaptive Prediction Sets
===========================================

In this example, we show how to use the Regularized Adaptive Prediction Sets
(RAPS) method introduced by Angelopoulos et al. (2021) with
:class:`~mapie.classification.SplitConformalClassifier`.

RAPS extends the Adaptive Prediction Sets (APS) method with a regularization
penalty that discourages including low-probability classes in the prediction
sets. The penalty is controlled by two hyperparameters (lambda and k), which
MAPIE tunes automatically on a fraction of the conformalization data.

Reference:
Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan,
and Jitendra Malik.
"Uncertainty Sets for Image Classifiers using Conformal Prediction."
International Conference on Learning Representations, 2021.
"""

##############################################################################
# We start by generating a multi-class classification dataset and splitting it
# into training, conformalization, and test sets.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import RAPSConformityScore
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
)
from mapie.utils import train_conformalize_test_split

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_classes=10,
    n_clusters_per_class=1,
    random_state=42,
)

(X_train, X_conf, X_test, y_train, y_conf, y_test) = train_conformalize_test_split(
    X, y, train_size=0.4, conformalize_size=0.4, test_size=0.2, random_state=42
)

##############################################################################
# We fit a Gradient Boosting classifier on the training set.

clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_train, y_train)
print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")

##############################################################################
# We wrap the fitted classifier in a
# :class:`~mapie.classification.SplitConformalClassifier` using the RAPS
# conformity score. ``size_raps`` sets the fraction of the conformalization
# data used to tune the regularization hyperparameters; the rest is used to
# compute the conformity scores.

confidence_levels = [0.90, 0.95]

mapie_clf = SplitConformalClassifier(
    estimator=clf,
    confidence_level=confidence_levels,
    conformity_score=RAPSConformityScore(size_raps=0.2),
    prefit=True,
    random_state=42,
)
mapie_clf.conformalize(X_conf, y_conf)
y_pred, y_pred_set = mapie_clf.predict_set(
    X_test, conformity_score_params={"include_last_label": True}
)

##############################################################################
# ``y_pred_set`` is a boolean array of shape
# ``(n_samples, n_classes, n_confidence_levels)``. Let's look at the prediction
# sets of the first few test samples at the 90% confidence level.

for i in range(5):
    labels = np.where(y_pred_set[i, :, 0])[0]
    print(f"Sample {i}: true={y_test[i]}, predicted={y_pred[i]}, set={labels}")

##############################################################################
# We check that the empirical coverage reaches the target and report the
# average prediction set size.

coverage = classification_coverage_score(y_test, y_pred_set)
width = classification_mean_width_score(y_pred_set)

print(f"{'Conf. Level':<13} {'Coverage':<10} {'Avg. Size':<10}")
print("-" * 33)
for i, cl in enumerate(confidence_levels):
    print(f"{cl:<13.2f} {coverage[i]:<10.3f} {width[i]:<10.3f}")

##############################################################################
# Finally, we plot the distribution of prediction set sizes. RAPS is adaptive:
# easy samples get small sets (often a single class), while harder samples get
# larger sets.

set_sizes = y_pred_set.sum(axis=1)
bins = np.arange(1, set_sizes.max() + 2) - 0.5

fig, axs = plt.subplots(1, len(confidence_levels), figsize=(10, 4), sharey=True)
for i, (ax, cl) in enumerate(zip(axs, confidence_levels)):
    ax.hist(set_sizes[:, i], bins=bins, edgecolor="black")
    ax.set_title(f"Confidence level {cl:.0%}")
    ax.set_xlabel("Prediction set size")
axs[0].set_ylabel("Number of test samples")
plt.tight_layout()
plt.show()
