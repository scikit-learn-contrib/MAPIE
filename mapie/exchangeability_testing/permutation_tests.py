from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom
from sklearn.model_selection import train_test_split

from mapie.regression import SplitConformalRegressor


class TestStatistic:
    def __init__(self):
        pass

    def compute(self):
        pass


class TestStatisticOnLabeledDataset(TestStatistic):
    def __init__(self):
        pass

    def compute(self, X, y):
        pass


class TestStatisticOnUnlabeledDataset(TestStatistic):
    def __init__(self):
        pass

    def compute(self, X):
        pass


class TestStatisticOnNonConformityScores(TestStatistic):
    def __init__(self):
        pass

    def compute(self, scores):
        middle_idx = len(scores) // 2

        mean_left = np.mean(scores[:middle_idx])
        mean_right = np.mean(scores[middle_idx:])

        diff = np.abs(mean_left - mean_right)

        return diff

    def __call__(self, scores):
        return self.compute(scores)


class PermutationTest:
    def __init__(
        self,
        method: Literal["p-value permutation", "Monte Carlo"],
        confidence_level=0.95,
        mapie_estimator: Optional[
            Callable
        ] = None,  # to get predictions and score function
    ):
        self.method = method
        self.delta = 1 - confidence_level
        self.mapie_estimator = mapie_estimator

    def _compute_non_conformity_scores(
        self, X: NDArray, y: NDArray, y_pred: Optional[NDArray]
    ):
        if y_pred is None:
            if self.mapie_estimator is None:
                X_train, X, y_train, y = train_test_split(
                    X, y, test_size=0.8, shuffle=False
                )
                self.mapie_estimator = SplitConformalRegressor(
                    prefit=False
                )  # TODO: handle classif
                self.mapie_estimator.fit(X_train, y_train)
            y_pred = self.mapie_estimator.predict(X)
        else:
            if self.mapie_estimator is None:
                self.mapie_estimator = SplitConformalRegressor()  # dummy estimator used to compute scores (no need to compute y_pred). TODO: handle classif

        scores = self.mapie_estimator._mapie_regressor.conformity_score_function_.get_conformity_scores(
            y,
            y_pred,
            X=X,  # TODO: check les kwargs possibles (X, ...)
        )

        return scores

    def run(self, X: NDArray, y: NDArray, y_pred: Optional[NDArray] = None):
        # 1. Transform the dataset (X, y) into a suitable non-conformity score preserving the exchangeability property
        # scores = self._compute_non_conformity_scores(X, y, y_pred)

        # 2. Compute the test statistic for the non-conformity scores of the original dataset (X_test, y_test)

        pass


class PValuePermutationTest(PermutationTest):
    """
    Permutation test based on p-values computed from conformity scores.

    Parameters
    ----------
    random_state : Optional[int]
        Controls the randomness of the permutations.

        By default `None`.

    num_permutations : int
        Number of permutations used to estimate the p-value.

        By default `1000`.

    *args : tuple
        Additional positional arguments forwarded to `PermutationTest`.

    **kwargs : dict
        Additional keyword arguments forwarded to `PermutationTest`.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.exchangeability_testing.permutation_tests import (
    ...     PValuePermutationTest,
    ... )
    >>> X = np.arange(20, dtype=float).reshape(-1, 1)
    >>> y = 2 * X.ravel() + np.array([0.0, 0.1] * 10)
    >>> test = PValuePermutationTest(
    ...     method="p-value permutation",
    ...     confidence_level=0.8,
    ...     random_state=0,
    ...     num_permutations=10,
    ... )
    >>> test.run(X, y)
    0
    >>> test.p_values.shape
    (11,)
    """

    def __init__(self, random_state=None, num_permutations=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.RandomState(random_state)
        self.num_permutations = num_permutations

        self.test_statistic = TestStatisticOnNonConformityScores()

    def run(self, X: NDArray, y: NDArray, y_pred: Optional[NDArray] = None):
        scores = self._compute_non_conformity_scores(X, y, y_pred)

        test_statistic_reference = self.test_statistic(scores)

        rank = 1
        self.p_values = np.empty(self.num_permutations + 1)
        self.p_values[0] = 1.0
        n = len(scores)
        for t in range(1, self.num_permutations + 1):
            permuted = self.rng.permutation(n)
            scores_permuted = scores[permuted]
            test_statistic_permutation = self.test_statistic(scores_permuted)

            if test_statistic_permutation >= test_statistic_reference:
                rank += 1
            self.p_values[t] = rank / (t + 1)

        is_exchangeable = bool(self.p_values[-1] > self.delta)

        return is_exchangeable


class SequentialMonteCarloTest(PermutationTest):
    def __init__(
        self, strategy, num_permutations=1000, random_state=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.rng = np.random.RandomState(random_state)
        self.num_permutations = num_permutations

        valid_strategies = {"aggressive", "binomial", "binomial_mixture"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Expected one of {valid_strategies}."
            )

        self.test_statistic = TestStatisticOnNonConformityScores()

    def run(self, X: NDArray, y: NDArray, y_pred: Optional[NDArray] = None):
        scores = self._compute_non_conformity_scores(X, y, y_pred)

        test_statistic_reference = self.test_statistic(scores)

        c = self.delta * 0.90
        p_zero = 1 / np.ceil(np.sqrt(2 * np.pi * np.exp(1 / 6)) / self.delta)

        rank = 1
        wealth_bin = np.array([1.0])
        wealth_agg = np.array([1.0])
        wealth_bm = np.array(
            [1.0]
        )  # TODO: check why it was [] before. I also removed the concat at the end.
        n = len(scores)
        for i in range(1, self.num_permutations + 1):
            permuted = self.rng.permutation(n)
            scores_permuted = scores[permuted]
            test_statistic_permutation = self.test_statistic(scores_permuted)

            if wealth_bin[-1] * p_zero * (i + 1) / rank < self.delta:
                pt = 0
            else:
                pt = p_zero

            if test_statistic_permutation >= test_statistic_reference:
                bet_bin_i = pt * (i + 1) / rank
                bet_agg_i = 0.0
                rank += 1
            else:
                bet_bin_i = (1 - pt) * (i + 1) / (i - (rank - 1))
                bet_agg_i = (i + 1) / i
            wealth_bm_i = (1 - binom.cdf(rank - 1, i + 1, c)) / c

            wealth_bin = np.append(wealth_bin, wealth_bin[-1] * bet_bin_i)
            wealth_agg = np.append(wealth_agg, wealth_agg[-1] * bet_agg_i)
            wealth_bm = np.append(wealth_bm, wealth_bm_i)

            # early stopping if possible
            if (
                self.strategy == "aggressive"
                and (wealth_agg[-1] < self.delta or wealth_agg[-1] >= 1 / self.delta)
                and (i > 50)
            ):
                break
            if (
                self.strategy == "binomial"
                and (wealth_bin[-1] < self.delta or wealth_bin[-1] >= 1 / self.delta)
                and (i > 50)
            ):
                break
            if (
                self.strategy == "binomial_mixture"
                and (wealth_agg[-1] < self.delta or wealth_agg[-1] >= 1 / self.delta)
                and (i > 50)
            ):
                break

        if self.strategy == "binomial":
            self.p_values = np.minimum(1 / wealth_bin, 1)
        elif self.strategy == "aggressive":
            self.p_values = np.minimum(1 / wealth_agg, 1)
        elif self.strategy == "binomial_mixture":
            self.p_values = np.minimum(1 / wealth_bm, 1)

        is_exchangeable = bool(self.p_values[-1] < self.delta)

        return is_exchangeable
