"""
Permutation test for exchangeability
====================================

This example illustrates how to run `PValuePermutationTest` on a toy
regression problem. We compare a dataset with exchangeable residuals to a
dataset where the second half of the residuals is shifted, which breaks the
exchangeability assumption used by conformal methods.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mapie.exchangeability_testing.permutation_tests import PValuePermutationTest
from mapie.regression import SplitConformalRegressor

##############################################################################
# 1. Fit and conformalize a regression model
# ------------------------------------------
#
# We first build a small regression problem and fit a
# `SplitConformalRegressor`. The fitted MAPIE object is then reused by the
# permutation test to compute conformity scores from supplied predictions.

rng = np.random.RandomState(7)
X = np.linspace(0, 1, 200).reshape(-1, 1)
y = 3 * X.ravel() + rng.normal(scale=0.1, size=X.shape[0])

X_train, X_conformalize, y_train, y_conformalize = train_test_split(
    X, y, test_size=0.5, shuffle=False
)

mapie_regressor = SplitConformalRegressor(
    estimator=LinearRegression(),
    prefit=False,
)
mapie_regressor.fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)

##############################################################################
# 2. Create two evaluation datasets
# ---------------------------------
#
# The first evaluation set has residuals centered around zero and is therefore
# compatible with exchangeability. In the second one, the second half of the
# targets is shifted upward, creating a visible distribution change.

X_eval = np.linspace(0.1, 0.9, 100).reshape(-1, 1)
y_pred = 3 * X_eval.ravel()

y_exchangeable = y_pred + rng.normal(scale=0.1, size=X_eval.shape[0])
y_shifted = y_exchangeable.copy()
y_shifted[len(y_shifted) // 2 :] += 0.8

##############################################################################
# 3. Run the permutation test
# ---------------------------
#
# We track the p-value after each permutation. The shifted dataset should end
# up with a much smaller p-value than the exchangeable one.

exchangeable_test = PValuePermutationTest(
    method="p-value permutation",
    confidence_level=0.9,
    random_state=7,
    num_permutations=200,
    mapie_estimator=mapie_regressor,
)
shifted_test = PValuePermutationTest(
    method="p-value permutation",
    confidence_level=0.9,
    random_state=7,
    num_permutations=200,
    mapie_estimator=mapie_regressor,
)

exchangeable_decision = exchangeable_test.run(X_eval, y_exchangeable, y_pred=y_pred)
shifted_decision = shifted_test.run(X_eval, y_shifted, y_pred=y_pred)

print(
    "Exchangeable residuals:",
    f"decision={exchangeable_decision}",
    f"final p-value={exchangeable_test.p_values[-1]:.3f}",
)
print(
    "Shifted residuals:",
    f"decision={shifted_decision}",
    f"final p-value={shifted_test.p_values[-1]:.3f}",
)

##############################################################################
# 4. Visualize the running p-values
# ---------------------------------
#
# The horizontal line marks `delta = 1 - confidence_level`. A final p-value
# below this threshold leads to a rejection of exchangeability.

steps = np.arange(exchangeable_test.num_permutations + 1)
delta = exchangeable_test.delta

plt.figure(figsize=(8, 4))
plt.plot(steps, exchangeable_test.p_values, label="Exchangeable residuals")
plt.plot(steps, shifted_test.p_values, label="Shifted residuals")
plt.axhline(delta, color="black", linestyle="--", label=f"delta = {delta:.2f}")
plt.xlabel("Number of permutations")
plt.ylabel("Running p-value")
plt.title("P-value permutation test for exchangeability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
