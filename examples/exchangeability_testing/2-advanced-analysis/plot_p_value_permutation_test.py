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

from mapie.exchangeability_testing.permutation_tests import (
    PValuePermutationTest,
    SequentialMonteCarloTest,
)
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
# 3. Exchangeable case: run all methods and plot
# -----------------------------------------------
#
# We start with a dataset that should satisfy exchangeability.

confidence_level = 0.9
num_permutations = 200

exchangeable_pvalue_test = PValuePermutationTest(
    method="p-value permutation",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)
exchangeable_aggressive_test = SequentialMonteCarloTest(
    strategy="aggressive",
    method="Monte Carlo",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)
exchangeable_binomial_test = SequentialMonteCarloTest(
    strategy="binomial",
    method="Monte Carlo",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)
exchangeable_binomial_mixture_test = SequentialMonteCarloTest(
    strategy="binomial_mixture",
    method="Monte Carlo",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)

exchangeable_pvalue_detected = exchangeable_pvalue_test.run(
    X_eval, y_exchangeable, y_pred=y_pred
)

exchangeable_aggressive_detected = exchangeable_aggressive_test.run(
    X_eval, y_exchangeable, y_pred=y_pred
)

exchangeable_binomial_detected = exchangeable_binomial_test.run(
    X_eval, y_exchangeable, y_pred=y_pred
)

exchangeable_binomial_mixture_detected = exchangeable_binomial_mixture_test.run(
    X_eval, y_exchangeable, y_pred=y_pred
)

print("\nExchangeable dataset")
print("--------------------")
print(f"PValuePermutationTest: detected={exchangeable_pvalue_detected}")
print(
    f"SequentialMonteCarloTest (aggressive): detected={exchangeable_aggressive_detected}"
)
print(f"SequentialMonteCarloTest (binomial): detected={exchangeable_binomial_detected}")
print(
    "SequentialMonteCarloTest (binomial_mixture): "
    f"detected={exchangeable_binomial_mixture_detected}"
)

delta = exchangeable_pvalue_test.delta
plt.figure(figsize=(8, 4))
plt.plot(exchangeable_pvalue_test.p_values, label="PValuePermutationTest", zorder=4)
plt.plot(
    exchangeable_aggressive_test.p_values,
    label="SMC aggressive",
    linestyle="-",
    linewidth=3.0,
    zorder=1,
)
plt.plot(
    exchangeable_binomial_test.p_values,
    label="SMC binomial",
    linestyle="--",
    linewidth=2.0,
    zorder=2,
)
plt.plot(
    exchangeable_binomial_mixture_test.p_values,
    label="SMC binomial_mixture",
    linestyle=":",
    linewidth=2.0,
    zorder=3,
)
plt.axhline(delta, color="black", linestyle="--", label=f"delta = {delta:.2f}")
plt.xlabel("Number of permutations")
plt.ylabel("Running p-value")
plt.title("Exchangeable residuals")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

##############################################################################
# 4. Non-exchangeable case: run all methods and plot
# ---------------------------------------------------
#
# We now repeat the same comparison on the shifted dataset.

shifted_pvalue_test = PValuePermutationTest(
    method="p-value permutation",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)
shifted_aggressive_test = SequentialMonteCarloTest(
    strategy="aggressive",
    method="Monte Carlo",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)
shifted_binomial_test = SequentialMonteCarloTest(
    strategy="binomial",
    method="Monte Carlo",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)
shifted_binomial_mixture_test = SequentialMonteCarloTest(
    strategy="binomial_mixture",
    method="Monte Carlo",
    confidence_level=confidence_level,
    random_state=7,
    num_permutations=num_permutations,
    mapie_estimator=mapie_regressor,
)

shifted_pvalue_detected = shifted_pvalue_test.run(X_eval, y_shifted, y_pred=y_pred)
shifted_aggressive_detected = shifted_aggressive_test.run(
    X_eval, y_shifted, y_pred=y_pred
)
shifted_binomial_detected = shifted_binomial_test.run(X_eval, y_shifted, y_pred=y_pred)
shifted_binomial_mixture_detected = shifted_binomial_mixture_test.run(
    X_eval, y_shifted, y_pred=y_pred
)

print("\nNon-exchangeable dataset")
print("------------------------")
print(f"PValuePermutationTest: detected={shifted_pvalue_detected}")
print(f"SequentialMonteCarloTest (aggressive): detected={shifted_aggressive_detected}")
print(f"SequentialMonteCarloTest (binomial): detected={shifted_binomial_detected}")
print(
    "SequentialMonteCarloTest (binomial_mixture): "
    f"detected={shifted_binomial_mixture_detected}"
)

plt.figure(figsize=(8, 4))
plt.plot(shifted_pvalue_test.p_values, label="PValuePermutationTest", zorder=4)
plt.plot(
    shifted_aggressive_test.p_values,
    label="SMC aggressive",
    linestyle="-",
    linewidth=3.0,
    zorder=1,
)
plt.plot(
    shifted_binomial_test.p_values,
    label="SMC binomial",
    linestyle="--",
    linewidth=2.0,
    zorder=2,
)
plt.plot(
    shifted_binomial_mixture_test.p_values,
    label="SMC binomial_mixture",
    linestyle=":",
    linewidth=2.0,
    zorder=3,
)
plt.axhline(delta, color="black", linestyle="--", label=f"delta = {delta:.2f}")
plt.xlabel("Number of permutations")
plt.ylabel("Running p-value")
plt.title("Non-exchangeable residuals")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
