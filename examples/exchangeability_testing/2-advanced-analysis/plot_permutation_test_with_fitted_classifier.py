"""
Permutation test with a fitted classifier
=========================================

This example illustrates how to run `PValuePermutationTest` on a multiclass
classification problem using a user-provided fitted
`SplitConformalClassifier`. We use the `"top_k"` conformity score instead of the
default one, first test exchangeability on a held-out dataset, and then
continue with the usual conformalization and prediction pipeline.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from mapie.classification import SplitConformalClassifier
from mapie.exchangeability_testing.permutation_tests import PValuePermutationTest
from mapie.utils import train_conformalize_test_split

##############################################################################
# 1. Fit a conformal classifier with a non-default conformity score
# -----------------------------------------------------------------
#
# We first split the data into train, conformalization, and test subsets using
# MAPIE's built-in utility. We fit a multiclass `KNeighborsClassifier` on the
# training data, then wrap it in a `SplitConformalClassifier` with the `"top_k"`
# conformity score in prefit mode. At this stage, the MAPIE classifier is
# initialized but not conformalized yet.

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
# 2. Test exchangeability on the held-out dataset
# -----------------------------------------------
#
# To assess whether conformalization is appropriate for the downstream test
# block, we run the permutation test on the concatenation of the future
# conformalization and test datasets. The permutation test internally copies the
# provided MAPIE estimator, then fits its own conformity scores on this dataset.

X_eval = np.concatenate([X_conformalize, X_test], axis=0)
y_exchangeable = np.concatenate([y_conformalize, y_test], axis=0)

##############################################################################
# 3. Run the permutation test
# ---------------------------

confidence_level = 0.9
num_permutations = 200

exchangeability_test = PValuePermutationTest(
    confidence_level=confidence_level,
    random_state=RANDOM_STATE,
    num_permutations=num_permutations,
    mapie_estimator=mapie_classifier,
)

exchangeability_detected = exchangeability_test.run(X_eval, y_exchangeable)

print("\nExchangeable classification dataset")
print("----------------------------------")
print(f"PValuePermutationTest: data exchangeability={exchangeability_detected}")

##############################################################################
# 4. Plot the running p-values
# ----------------------------

delta = exchangeability_test.delta
plt.figure(figsize=(8, 4))
plt.plot(exchangeability_test.p_values, label="Exchangeable dataset")
plt.axhline(delta, color="black", linestyle="--", label=f"delta = {delta:.2f}")
plt.xlabel("Number of permutations")
plt.ylabel("Running p-value")
plt.title("Permutation test with a fitted top-k classifier")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

##############################################################################
# 5. Continue with the standard MAPIE pipeline
# --------------------------------------------
#
# Once exchangeability has been checked, we continue with the standard MAPIE
# pipeline on the predefined conformalization and test subsets.

mapie_classifier.conformalize(X_conformalize, y_conformalize)
y_pred, y_pred_set = mapie_classifier.predict_set(X_test)
average_set_size = np.mean(np.sum(y_pred_set[:, :, 0], axis=1))

print("\nClassical MAPIE pipeline")
print("------------------------")
print(f"Average prediction-set size: {average_set_size:.2f}")
