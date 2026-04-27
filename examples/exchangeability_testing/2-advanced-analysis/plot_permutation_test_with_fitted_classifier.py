"""
# Permutation test with a fitted classifier

This example illustrates how to run
:class:`~mapie.exchangeability_testing.permutations.PValuePermutationTest`
on a multiclass classification problem using a user-provided fitted
:class:`~mapie.classification.SplitConformalClassifier`.

Compared to the companion permutation example, the key differences are:

- the estimator is *fitted beforehand* and passed to the test via the
  ``mapie_estimator`` argument. The permutation test then runs directly
  on held-out data without any internal refitting, which is cheaper and
  typically more powerful;
- we use the ``"top_k"`` conformity score instead of the default one;
- once exchangeability has been checked, we continue with the usual
  MAPIE conformalization and prediction pipeline.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from utils import plot_running_pvalues

from mapie.classification import SplitConformalClassifier
from mapie.exchangeability_testing.permutations import PValuePermutationTest
from mapie.utils import train_conformalize_test_split

##############################################################################
# Fit a conformal classifier with a non-default conformity score
# --------------------------------------------------------------
#
# We split the data into train, conformalization, and test subsets using
# MAPIE's built-in utility. A multiclass :class:`KNeighborsClassifier` is
# fitted on the training data, then wrapped in a
# :class:`~mapie.classification.SplitConformalClassifier` with the
# ``"top_k"`` conformity score in prefit mode. At this stage, the MAPIE
# classifier is initialized but not conformalized yet.

RANDOM_STATE = 7

X_full, y_full = make_classification(
    n_samples=360,
    n_features=10,
    n_informative=6,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=RANDOM_STATE,
)

(
    X_train,
    X_conformalize,
    X_test,
    y_train,
    y_conformalize,
    y_test,
) = train_conformalize_test_split(
    X_full,
    y_full,
    train_size=0.5,
    conformalize_size=0.25,
    test_size=0.25,
    shuffle=False,
)

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)

mapie_classifier = SplitConformalClassifier(
    estimator=classifier,
    conformity_score="top_k",
    prefit=True,
)

##############################################################################
# Test exchangeability on the held-out dataset
# --------------------------------------------
#
# To assess whether the conformalization guarantees provided by MAPIE are
# valid, we test the exchangeability of the conformalization and test
# datasets. If the datasets are exchangeable, the coverage guarantees hold;
# otherwise they may be violated.

X_eval = np.concatenate([X_conformalize, X_test], axis=0)
y_eval = np.concatenate([y_conformalize, y_test], axis=0)

test_level = 0.1
num_permutations = 200

exchangeability_test = PValuePermutationTest(
    test_level=test_level,
    random_state=RANDOM_STATE,
    num_permutations=num_permutations,
    mapie_estimator=mapie_classifier,
)

exchangeability_test.run(X_eval, y_eval)

print("\nExchangeability result")
print("----------------------")
print(f"Is the held-out dataset exchangeable? {exchangeability_test.is_exchangeable}")
print(f"Final running p-value: {exchangeability_test.p_values[-1]:.3f}")

##############################################################################
# Visualize the running p-value
# -----------------------------

plot_running_pvalues(
    [exchangeability_test],
    ["Running p-value"],
    test_level,
    "Permutation test with a fitted top-k classifier",
)

##############################################################################
# Continue with the standard MAPIE pipeline
# -----------------------------------------
#
# Once exchangeability has been checked, we continue with the standard
# MAPIE pipeline on the predefined conformalization and test subsets.

mapie_classifier.conformalize(X_conformalize, y_conformalize)
y_pred, y_pred_set = mapie_classifier.predict_set(X_test)
average_set_size = float(np.mean(np.sum(y_pred_set[:, :, 0], axis=1)))
empirical_coverage = float(np.mean(y_pred_set[np.arange(len(y_test)), y_test, 0]))

print("\nClassical MAPIE pipeline")
print("------------------------")
print(f"Average prediction-set size: {average_set_size:.2f}")
print(f"Empirical coverage:          {empirical_coverage:.2f}")
