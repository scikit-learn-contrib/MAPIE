"""
Permutation test for exchangeability
====================================

This example illustrates how to run `PValuePermutationTest`  and
`SequentialMonteCarloTest` on a toy regression problem.
We compare a dataset with exchangeable residuals to a
dataset where the second half of the residuals is shifted, which breaks the
exchangeability assumption used by conformal methods.
"""

import matplotlib.pyplot as plt
import numpy as np

from mapie.exchangeability_testing.permutations import (
    PValuePermutationTest,
    SequentialMonteCarloTest,
)

##############################################################################
# Build two fixed datasets
# ------------------------
#
# We build a toy regression signal and compare two datasets:
#
# - one with residuals centered around zero, which should be close to
#   exchangeable
# - one where the second half of the targets is shifted upward, breaking
#   exchangeability
#
# No estimator is provided explicitly in this example. The permutation tests
# internally build and fit a default `SplitConformalRegressor` on a split of the data
# passed to `run`. This is necessary to be able to compute non-conformity scores
# which measure some sort of model prediction error. Permutation tests then test
# the exchangeability of the non-conformity scores. This approach allows more flexibility
# as complex dataset are reduced to a list of values. As non-conformity scores are
# exchangeability preserving transformations, the exchangeability test result is
# valid for the original dataset.

# If an estimator fitted on some training data is provided, the permutation tests
# should be run on the test data. Results would be better as no split would be needed
# to fit a model internally.

rng = np.random.RandomState(0)
X = np.linspace(0.1, 0.9, 500).reshape(-1, 1)
y_exchangeable = 3 * X.ravel() + rng.normal(scale=0.1, size=X.shape[0])
y_shifted = y_exchangeable.copy()
y_shifted[len(y_shifted) // 2 :] += 0.8

##############################################################################
# Run the exchangeable case
# -------------------------
#
# We start with a dataset that should satisfy exchangeability.

test_level = 0.1
num_permutations = 200

exchangeable_pvalue_test = PValuePermutationTest(
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)
exchangeable_aggressive_test = SequentialMonteCarloTest(
    strategy="aggressive",
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)
exchangeable_binomial_test = SequentialMonteCarloTest(
    strategy="binomial",
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)
exchangeable_binomial_mixture_test = SequentialMonteCarloTest(
    strategy="binomial_mixture",
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)

exchangeable_pvalue_test.run(X, y_exchangeable)
exchangeable_pvalue_detected = exchangeable_pvalue_test.is_exchangeable

exchangeable_aggressive_test.run(X, y_exchangeable)
exchangeable_aggressive_detected = exchangeable_aggressive_test.is_exchangeable

exchangeable_binomial_test.run(X, y_exchangeable)
exchangeable_binomial_detected = exchangeable_binomial_test.is_exchangeable

exchangeable_binomial_mixture_test.run(X, y_exchangeable)
exchangeable_binomial_mixture_detected = (
    exchangeable_binomial_mixture_test.is_exchangeable
)

print("\nExchangeable dataset")
print("--------------------")
print(
    f"PValuePermutationTest: data exchangeability={exchangeable_pvalue_detected}. Detection after {len(exchangeable_pvalue_test.p_values) - 1} permutations."
)
print(
    f"SequentialMonteCarloTest (aggressive): data exchangeability={exchangeable_aggressive_detected}. Detection after {len(exchangeable_aggressive_test.p_values) - 1} permutations."
)
print(
    f"SequentialMonteCarloTest (binomial): data exchangeability={exchangeable_binomial_detected}. Detection after {len(exchangeable_binomial_test.p_values) - 1} permutations."
)
print(
    "SequentialMonteCarloTest (binomial_mixture): "
    f"data exchangeability={exchangeable_binomial_mixture_detected}. Detection after {len(exchangeable_binomial_mixture_test.p_values) - 1} permutations."
)

test_level = exchangeable_pvalue_test.test_level
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
plt.axhline(
    test_level,
    color="black",
    linestyle="--",
    label=f"test_level = {test_level:.2f}",
)
plt.xlabel("Number of permutations")
plt.ylabel("Running p-value")
plt.title("Exchangeable residuals")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

##############################################################################
# Run the non-exchangeable case
# -----------------------------
#
# We now repeat the same comparison on the shifted dataset.

shifted_pvalue_test = PValuePermutationTest(
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)
shifted_aggressive_test = SequentialMonteCarloTest(
    strategy="aggressive",
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)
shifted_binomial_test = SequentialMonteCarloTest(
    strategy="binomial",
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)
shifted_binomial_mixture_test = SequentialMonteCarloTest(
    strategy="binomial_mixture",
    test_level=test_level,
    random_state=7,
    num_permutations=num_permutations,
)

shifted_pvalue_test.run(X, y_shifted)
shifted_pvalue_detected = shifted_pvalue_test.is_exchangeable
shifted_aggressive_test.run(X, y_shifted)
shifted_aggressive_detected = shifted_aggressive_test.is_exchangeable
shifted_binomial_test.run(X, y_shifted)
shifted_binomial_detected = shifted_binomial_test.is_exchangeable
shifted_binomial_mixture_test.run(X, y_shifted)
shifted_binomial_mixture_detected = shifted_binomial_mixture_test.is_exchangeable

print("\nNon-exchangeable dataset")
print("------------------------")
print(
    f"PValuePermutationTest: data exchangeability={shifted_pvalue_detected}. Detection after {len(shifted_pvalue_test.p_values) - 1} permutations."
)
print(
    f"SequentialMonteCarloTest (aggressive): data exchangeability={shifted_aggressive_detected}. Detection after {len(shifted_aggressive_test.p_values) - 1} permutations."
)
print(
    f"SequentialMonteCarloTest (binomial): data exchangeability={shifted_binomial_detected}. Detection after {len(shifted_binomial_test.p_values) - 1} permutations."
)
print(
    "SequentialMonteCarloTest (binomial_mixture): "
    f"data exchangeability={shifted_binomial_mixture_detected}. Detection after {len(shifted_binomial_mixture_test.p_values) - 1} permutations."
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
plt.axhline(
    test_level,
    color="black",
    linestyle="--",
    label=f"test_level = {test_level:.2f}",
)
plt.xlabel("Number of permutations")
plt.ylabel("Running p-value")
plt.title("Non-exchangeable residuals")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
