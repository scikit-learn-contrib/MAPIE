from ast import Tuple
from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import NDArray
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
        reference_dataset: Optional[
            Tuple[NDArray, NDArray]
        ] = None,  # to compute reference test statistic
        reference_non_conformity_scores: Optional[
            NDArray
        ] = None,  # to compute reference test statistic (cannot provide both dataset and scores)
    ):
        self.method = method
        self.delta = 1 - confidence_level

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
                self.mapie_estimator = SplitConformalRegressor(
                    prefit=False
                )  # dummy estimator used to compute scores (no need to compute y_pred). TODO: handle classif

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
    def __init__(self, random_state=None, num_permutations=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_statistic = TestStatisticOnNonConformityScores()
        self.rng = np.random.RandomState(random_state)
        self.num_permutations = num_permutations

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

        is_exchangeable = int(self.p_values[-1] > self.delta)

        return is_exchangeable
