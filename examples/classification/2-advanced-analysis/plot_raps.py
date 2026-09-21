"""
RAPS: Regularized Adaptive Prediction Sets
===========================================

In this example, we demonstrate the Regularized Adaptive Prediction Sets
(RAPS) method introduced by Angelopoulos et al. (2021). RAPS extends the
Adaptive Prediction Sets (APS) method by adding a regularization penalty
that encourages smaller prediction sets without sacrificing coverage.

We compare the prediction sets obtained by APS and RAPS on a multi-class
classification task. Both methods guarantee marginal coverage, but RAPS
uses a learned penalty (lambda, k) to discourage including low-probability
classes — an advantage that is most visible with many classes or weak
classifiers.

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
from mapie.metrics.classification import (
    classification_coverage_score,
    classification_mean_width_score,
)
from mapie.utils import train_conformalize_test_split

np.random.seed(42)

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_classes=10,
    n_clusters_per_class=1,
    random_state=42,
)

(X_train, X_conf, X_test, y_train, y_conf, y_test) = train_conformalize_test_split(
    X, y, train_size=0.4, conformalize_size=0.4, test_size=0.2
)

##############################################################################
# We fit a Gradient Boosting classifier on the training set.

clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_train, y_train)
print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")

##############################################################################
# Now we compare APS and RAPS. Both produce valid prediction sets (coverage
# >= 1 - alpha), but RAPS adds a regularization penalty that shrinks the
# average set size.

confidence_levels = [0.90, 0.95]

results = {}
for method in ["aps", "raps"]:
    mapie_clf = SplitConformalClassifier(
        estimator=clf,
        confidence_level=confidence_levels,
        conformity_score=method,
        prefit=True,
        random_state=42,
    )
    mapie_clf.conformalize(X_conf, y_conf)
    y_pred, y_pred_set = mapie_clf.predict_set(
        X_test, conformity_score_params={"include_last_label": True}
    )
    results[method] = {
        "coverage": classification_coverage_score(y_test, y_pred_set),
        "width": classification_mean_width_score(y_pred_set),
    }

##############################################################################
# Let's compare the coverage and average prediction set size for both methods.

print(f"{'Method':<8} {'Conf. Level':<13} {'Coverage':<10} {'Avg. Size':<10}")
print("-" * 43)
for method in ["aps", "raps"]:
    for i, cl in enumerate(confidence_levels):
        print(
            f"{method.upper():<8} {cl:<13.2f} "
            f"{results[method]['coverage'][i]:<10.3f} "
            f"{results[method]['width'][i]:<10.3f}"
        )

##############################################################################
# We visualize the average prediction set sizes.

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(confidence_levels))
width = 0.35

bars_aps = ax.bar(
    x - width / 2,
    results["aps"]["width"],
    width,
    label="APS",
    color="#1f77b4",
)
bars_raps = ax.bar(
    x + width / 2,
    results["raps"]["width"],
    width,
    label="RAPS",
    color="#ff7f0e",
)

ax.set_xlabel("Confidence Level")
ax.set_ylabel("Average Prediction Set Size")
ax.set_title("APS vs RAPS: Average Prediction Set Size")
ax.set_xticks(x)
ax.set_xticklabels([f"{cl:.0%}" for cl in confidence_levels])
ax.legend()
ax.set_ylim(0)

for bar in bars_aps:
    ax.annotate(
        f"{bar.get_height():.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
        fontsize=9,
    )
for bar in bars_raps:
    ax.annotate(
        f"{bar.get_height():.2f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()

##############################################################################
# We also compare the coverage to confirm both methods remain valid.

fig, ax = plt.subplots(figsize=(8, 5))

bars_aps = ax.bar(
    x - width / 2,
    results["aps"]["coverage"],
    width,
    label="APS",
    color="#1f77b4",
)
bars_raps = ax.bar(
    x + width / 2,
    results["raps"]["coverage"],
    width,
    label="RAPS",
    color="#ff7f0e",
)

for i, cl in enumerate(confidence_levels):
    ax.axhline(y=cl, color="red", linestyle="--", alpha=0.5)

ax.set_xlabel("Confidence Level")
ax.set_ylabel("Empirical Coverage")
ax.set_title("APS vs RAPS: Coverage (dashed red = target)")
ax.set_xticks(x)
ax.set_xticklabels([f"{cl:.0%}" for cl in confidence_levels])
ax.legend()
ax.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.show()

##############################################################################
# Both methods achieve at least the target coverage (above the dashed red
# line), confirming their theoretical validity. The regularization penalty in
# RAPS is most effective in settings with many classes and weak classifiers,
# where APS tends to include many low-probability classes. In favorable
# scenarios, RAPS can significantly reduce average set size.
#
# The key parameter controlling RAPS behavior is `size_raps`, which determines
# the fraction of calibration data used to tune the regularization
# hyperparameters (lambda and k). By default, it is set to 0.2. Increasing
# it gives more data for tuning but less for computing conformity scores.
