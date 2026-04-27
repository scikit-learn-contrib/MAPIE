"""
# Permutation tests for exchangeability

This example illustrates how to run
:class:`~mapie.exchangeability_testing.permutations.PValuePermutationTest` and
:class:`~mapie.exchangeability_testing.permutations.SequentialMonteCarloTest`
on a toy regression problem.

We compare a dataset with exchangeable residuals to a dataset where the
second half of the residuals is shifted, which breaks the exchangeability
assumption used by conformal methods.

**How permutation tests work.**
A non-conformity score is computed for every sample. Under the
exchangeability hypothesis, permuting those scores should leave their
distribution unchanged. A test statistic (here, a trend statistic
sensitive to ordering) is compared between the original score sequence
and many random permutations. The running p-value at step ``k`` is the
proportion of permutations whose statistic is at least as extreme as the
observed one after ``k`` random permutations have been drawn.

We compare four strategies:

- :class:`~mapie.exchangeability_testing.permutations.PValuePermutationTest`:
  draws ``num_permutations`` permutations and reports the classical Monte
  Carlo p-value.
- :class:`~mapie.exchangeability_testing.permutations.SequentialMonteCarloTest`
  with three variants (``aggressive``, ``binomial``, ``binomial_mixture``):
  sequential Monte Carlo tests that can stop early when there is enough
  evidence for or against exchangeability, often saving permutations.

**Estimator handling.**
No estimator is provided explicitly in this example. The tests internally
build and fit a default :class:`~mapie.regression.SplitConformalRegressor`
on a split of the data passed to ``run``. This is required to compute
non-conformity scores (a measure of prediction error). Since non-conformity
scores are exchangeability-preserving transformations, testing
exchangeability on them is valid for the original dataset.

If a fitted estimator is available, it can be passed via the
``mapie_estimator`` argument; the test then runs directly on the
held-out data without any internal split and tends to be more powerful.
See the companion example for that workflow.
"""

# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np
from examples.exchangeability_testing.utils import plot_running_pvalues

from mapie.exchangeability_testing.permutations import (
    PValuePermutationTest,
    SequentialMonteCarloTest,
)

##############################################################################
# Build two fixed datasets
# ------------------------
#
# We build a simple regression signal and compare two datasets:
#
# - one with residuals centered around zero, which should be close to
#   exchangeable;
# - one where the second half of the targets is shifted upward, breaking
#   exchangeability.

rng = np.random.RandomState(0)
X = np.linspace(0.1, 0.9, 100).reshape(-1, 1)
y_exchangeable = 3 * X.ravel() + rng.normal(scale=0.1, size=X.shape[0])
y_shifted = y_exchangeable.copy()
y_shifted[len(y_shifted) // 2 :] += 0.8

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
axes[0].scatter(X.ravel(), y_exchangeable, s=22, alpha=0.8)
axes[0].set_title("Exchangeable residuals")
axes[0].set_xlabel("Feature")
axes[0].set_ylabel("Target")

midpoint = len(y_shifted) // 2
axes[1].scatter(
    X[:midpoint, 0], y_shifted[:midpoint], s=22, alpha=0.8, label="Before shift"
)
axes[1].scatter(
    X[midpoint:, 0],
    y_shifted[midpoint:],
    s=28,
    alpha=0.9,
    marker="x",
    label="After shift",
)
axes[1].set_title("Non-exchangeable residuals")
axes[1].set_xlabel("Feature")
axes[1].legend()
plt.tight_layout()
plt.show()


##############################################################################
# Helper function
# ---------------
#
# To avoid code duplication, we define a small helper that runs all four
# permutation tests on a given dataset and plots their running p-values.

test_level = 0.1
num_permutations = 200

test_labels = [
    "PValuePermutationTest",
    "SMC aggressive",
    "SMC binomial",
    "SMC binomial_mixture",
]


def run_all_tests(X_data, y_data):
    """Run every permutation test and return the fitted instances."""
    pvalue_test = PValuePermutationTest(
        test_level=test_level,
        random_state=7,
        num_permutations=num_permutations,
    )
    aggressive_test = SequentialMonteCarloTest(
        strategy="aggressive",
        test_level=test_level,
        random_state=7,
        num_permutations=num_permutations,
    )
    binomial_test = SequentialMonteCarloTest(
        strategy="binomial",
        test_level=test_level,
        random_state=7,
        num_permutations=num_permutations,
    )
    binomial_mixture_test = SequentialMonteCarloTest(
        strategy="binomial_mixture",
        test_level=test_level,
        random_state=7,
        num_permutations=num_permutations,
    )
    for test in (pvalue_test, aggressive_test, binomial_test, binomial_mixture_test):
        test.run(X_data, y_data)
    return pvalue_test, aggressive_test, binomial_test, binomial_mixture_test


def print_summary(name, tests):
    print(f"\n{name}")
    print("-" * len(name))
    for label, test in zip(test_labels, tests):
        print(
            f"{label:<24}: is_exchangeable={test.is_exchangeable} "
            f"(stopped after {len(test.p_values)} permutations)"
        )


##############################################################################
# Exchangeable case
# -----------------
#
# We start with a dataset that should satisfy exchangeability. All tests
# should fail to reject the null hypothesis. Sequential Monte Carlo tests
# typically stop early without exhausting the permutation budget.

tests_exch = run_all_tests(X, y_exchangeable)
print_summary("Exchangeable dataset", tests_exch)
plot_running_pvalues(tests_exch, test_labels, test_level, "Exchangeable residuals")

##############################################################################
# Non-exchangeable case
# ---------------------
#
# We now repeat the same comparison on the shifted dataset. All tests
# should reject exchangeability. Sequential Monte Carlo variants typically
# stop much earlier than the classical p-value test because the evidence
# against the null quickly dominates.

tests_shifted = run_all_tests(X, y_shifted)
print_summary("Non-exchangeable dataset", tests_shifted)
plot_running_pvalues(
    tests_shifted, test_labels, test_level, "Non-exchangeable residuals"
)

##############################################################################
# Takeaways
# ---------
#
# - All permutation tests agree on the qualitative conclusion: the
#   exchangeable dataset is not rejected, the shifted one is.
# - Sequential Monte Carlo variants can save computation by stopping early
#   when the evidence is strong, which is especially useful when the
#   permutation budget is large and the true answer is clear-cut.
# - ``binomial`` and ``binomial_mixture`` tend to be conservative; the
#   ``aggressive`` strategy stops earliest on a clear alternative at the
#   price of slightly higher variance.
