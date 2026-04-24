from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from mapie.classification import CrossConformalClassifier, SplitConformalClassifier
from mapie.regression import (
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    SplitConformalRegressor,
)

MapieEstimator = Union[
    SplitConformalClassifier,
    SplitConformalRegressor,
]


class TestStatistic(ABC):
    """Base class for test statistics used in exchangeability tests."""

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> float:
        """Compute the test statistic value."""
        raise NotImplementedError  # pragma: no cover


class MeanShiftTestStatistic(TestStatistic):
    """Mean-shift statistic on two score halves.

    The statistic is the absolute difference between the mean score of
    the first half and the mean score of the second half.
    """

    def compute(self, scores: NDArray) -> float:
        """Compute the absolute mean difference between score halves.

        Parameters
        ----------
        scores : NDArray
            One-dimensional non-conformity scores.

        Returns
        -------
        float
            Absolute difference between the means of both score halves.
        """
        middle_idx = len(scores) // 2

        mean_left = np.mean(scores[:middle_idx])
        mean_right = np.mean(scores[middle_idx:])

        diff = np.abs(mean_left - mean_right)
        return float(diff)

    def __call__(self, scores: NDArray) -> float:
        """Alias to :meth:`compute`."""
        return self.compute(scores)


class PermutationTest(ABC):
    """Base class for exchangeability tests based on permutations.

    Parameters
    ----------
    test_level : float, default=0.05
        Level used to test the hypothesis that the dataset is exchangeable.
        The probability that the test gives a false positive is at most
        `test_level` (type I error).
    mapie_estimator : Optional[MapieEstimator], default=None
        MAPIE estimator used to compute predictions and non-conformity
        scores. Supported estimators are
        :class:`SplitConformalClassifier`,
        and :class:`SplitConformalRegressor`.
        If ``None``, a default
        :class:`SplitConformalClassifier` or
        :class:`SplitConformalRegressor` is built
        when needed.
        If the estimator is not fitted or not provided, it will be fitted on a
        slice of the data in order to compute non-conformity scores.
    random_state : Optional[int], default=None
        Seed controlling the randomness of permutations.
    num_permutations : int, default=1000
        Number of permutations used by permutation-based tests.
    """

    def __init__(
        self,
        test_level: float = 0.05,
        mapie_estimator: Optional[MapieEstimator] = None,
        task: Optional[Literal["classification", "regression"]] = None,
        random_state: Optional[int] = None,
        num_permutations: int = 1000,
    ) -> None:
        if not (0.0 < test_level < 1.0):
            raise ValueError("test_level must be in (0, 1).")
        self.test_level = test_level
        self.mapie_estimator = self._prepare_estimator(mapie_estimator)
        self.task = task
        self.rng = np.random.RandomState(random_state)
        self.num_permutations = num_permutations
        self.p_values: NDArray = np.array([])
        self.test_statistic = MeanShiftTestStatistic()

    @staticmethod
    def _prepare_estimator(
        mapie_estimator: Optional[MapieEstimator],
    ) -> Optional[MapieEstimator]:
        """Copy an estimator to avoid modifying the original estimator
        and clear conformalization state to allow calling conformalize method."""
        if mapie_estimator is None:
            return None

        if isinstance(
            mapie_estimator,
            (
                CrossConformalClassifier,
                CrossConformalRegressor,
                JackknifeAfterBootstrapRegressor,
            ),
        ):
            raise ValueError(
                "Cross conformal and jackknife-after-bootstrap estimators are not "
                "supported in permutation tests because they mix the data."
            )

        estimator_copy = deepcopy(mapie_estimator)

        if hasattr(estimator_copy, "_is_conformalized"):
            estimator_copy._is_conformalized = False
        if hasattr(estimator_copy, "_predict_params"):
            estimator_copy._predict_params = {}

        for attr_name in ("conformity_scores_", "quantiles_"):
            if hasattr(estimator_copy, attr_name):
                delattr(estimator_copy, attr_name)

        for inner_estimator_name in ("_mapie_regressor", "_mapie_classifier"):
            if hasattr(estimator_copy, inner_estimator_name):
                inner_estimator = getattr(estimator_copy, inner_estimator_name)
                for attr_name in ("conformity_scores_", "quantiles_"):
                    if hasattr(inner_estimator, attr_name):
                        delattr(inner_estimator, attr_name)

        return estimator_copy

    def _infer_task(self, y: NDArray) -> Literal["classification", "regression"]:
        """Infer whether the current data should use classification scores."""
        if self.mapie_estimator is not None:
            if hasattr(self.mapie_estimator, "_mapie_classifier"):
                return "classification"
            if hasattr(self.mapie_estimator, "_mapie_regressor"):
                return "regression"
            raise ValueError(
                "Unable to infer the task from the provided MAPIE estimator."
            )

        type_of_target_y = type_of_target(y)
        if "multiclass" in type_of_target_y or "binary" in type_of_target_y:
            return "classification"
        if "continuous" in type_of_target_y:
            return "regression"
        raise ValueError("Unknown type of target, please manually set the task type.")

    def _compute_non_conformity_scores(self, X: NDArray, y: NDArray) -> NDArray:
        """Compute non-conformity scores from inputs and predictions.

        Parameters
        ----------
        X : NDArray
            Feature matrix.
        y : NDArray
            Target values.
        Returns
        -------
        NDArray
            Non-conformity scores associated with ``(X, y)``.
        """

        if self.task is None:
            self.task = self._infer_task(y)

        if self.mapie_estimator is None:
            if self.task == "classification":
                self.mapie_estimator = SplitConformalClassifier(prefit=False)
            elif self.task == "regression":
                self.mapie_estimator = SplitConformalRegressor(prefit=False)
            else:
                raise ValueError("Unknown task type.")

        if not self.mapie_estimator._is_fitted:
            X_train, X, y_train, y = train_test_split(
                X,
                y,
                test_size=0.7,
                shuffle=False,
            )
            self.mapie_estimator.fit(X_train, y_train)

        self.mapie_estimator.conformalize(X, y)  # compute scores internally

        if self.task == "classification":
            self.mapie_estimator = cast(SplitConformalClassifier, self.mapie_estimator)
            scores = cast(
                NDArray,
                self.mapie_estimator._mapie_classifier.conformity_scores_,
            )
        else:
            self.mapie_estimator = cast(SplitConformalRegressor, self.mapie_estimator)
            scores = cast(
                NDArray,
                self.mapie_estimator._mapie_regressor.conformity_scores_,
            )

        return scores

    @abstractmethod
    def run(self, X: NDArray, y: NDArray) -> bool:
        """Run a permutation-based exchangeability test."""
        raise NotImplementedError  # pragma: no cover


class PValuePermutationTest(PermutationTest):
    """
    Permutation test based on p-values computed from conformity scores.

    Parameters
    ----------
    test_level : float, default=0.05
        Level used to test the hypothesis that the dataset is exchangeable.
        The probability that the test gives a false positive is at most
        `test_level` (type I error).
    mapie_estimator : Optional[MapieEstimator], default=None
        MAPIE estimator used to compute predictions and non-conformity
        scores. Supported estimators are
        :class:`SplitConformalClassifier`,
        and :class:`SplitConformalRegressor`.
        If ``None``, a default
        :class:`SplitConformalClassifier` or
        :class:`SplitConformalRegressor` is built
        when needed.
        If the estimator is not fitted or not provided, it will be fitted on a
        slice of the data in order to compute non-conformity scores.
    task : Optional[Literal["classification", "regression"]], default=None
        Task type. If ``None``, the task is inferred from `y`.
    random_state : Optional[int], default=None
        Seed controlling the randomness of permutations.
    num_permutations : int, default=1000
        Number of permutations used to estimate the p-value.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.exchangeability_testing.permutation_tests import (
    ...     PValuePermutationTest,
    ... )
    >>> X = np.arange(100, dtype=float).reshape(-1, 1)
    >>> y = 2 * X.ravel() + np.array([0.0, 0.1] * 50)
    >>> test = PValuePermutationTest(
    ...     test_level=0.2,
    ... )
    >>> test.run(X, y)
    True
    """

    def __init__(
        self,
        test_level: float = 0.05,
        mapie_estimator: Optional[MapieEstimator] = None,
        task: Optional[Literal["classification", "regression"]] = None,
        random_state: Optional[int] = None,
        num_permutations: int = 1000,
    ) -> None:
        super().__init__(
            test_level=test_level,
            mapie_estimator=mapie_estimator,
            task=task,
            random_state=random_state,
            num_permutations=num_permutations,
        )

    def run(self, X: NDArray, y: NDArray) -> bool:
        """Run a p-value permutation test.

        Parameters
        ----------
        X : NDArray
            Feature matrix.
        y : NDArray
            Target values.
        Returns
        -------
        bool
            Whether the dataset is deemed exchangeable.
        """
        scores = self._compute_non_conformity_scores(X, y)

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

        is_exchangeable = bool(self.p_values[-1] > self.test_level)

        return is_exchangeable


class SequentialMonteCarloTest(PermutationTest):
    """Sequential Monte Carlo exchangeability test.

    Parameters
    ----------
    strategy : {"aggressive", "binomial", "binomial_mixture"}
        Wealth update strategy for the sequential test.
    test_level : float, default=0.05
        Level used to test the hypothesis that the dataset is exchangeable.
        The probability that the test gives a false positive is at most
        `test_level` (type I error).
    mapie_estimator : Optional[MapieEstimator], default=None
        MAPIE estimator used to compute predictions and non-conformity
        scores. Supported estimators are
        :class:`SplitConformalClassifier`,
        and :class:`SplitConformalRegressor`.
        If ``None``, a default
        :class:`SplitConformalClassifier` or
        :class:`SplitConformalRegressor` is built
        when needed.
        If the estimator is not fitted or not provided, it will be fitted on a
        slice of the data in order to compute non-conformity scores.
    task : Optional[Literal["classification", "regression"]], default=None
        Task type. If ``None``, the task is inferred from `y`.
    random_state : Optional[int], default=None
        Seed controlling the randomness of permutations.
    num_permutations : int, default=1000
        Maximum number of permutations.
    """

    def __init__(
        self,
        strategy: Literal["aggressive", "binomial", "binomial_mixture"],
        test_level: float = 0.05,
        mapie_estimator: Optional[MapieEstimator] = None,
        task: Optional[Literal["classification", "regression"]] = None,
        random_state: Optional[int] = None,
        num_permutations: int = 1000,
    ) -> None:
        super().__init__(
            test_level=test_level,
            mapie_estimator=mapie_estimator,
            task=task,
            random_state=random_state,
            num_permutations=num_permutations,
        )
        self.strategy = strategy

        valid_strategies = {"aggressive", "binomial", "binomial_mixture"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. Expected one of {valid_strategies}."
            )

    def run(self, X: NDArray, y: NDArray) -> bool:
        """Run a sequential Monte Carlo permutation test.

        Parameters
        ----------
        X : NDArray
            Feature matrix.
        y : NDArray
            Target values.
        Returns
        -------
        bool
            Whether the dataset is deemed exchangeable.
        """
        scores = self._compute_non_conformity_scores(X, y)

        test_statistic_reference = self.test_statistic(scores)

        c = self.test_level * 0.90
        p_zero = 1 / np.ceil(np.sqrt(2 * np.pi * np.exp(1 / 6)) / self.test_level)

        rank = 1
        wealth_bin = np.array([1.0])
        wealth_agg = np.array([1.0])
        wealth_bm = np.array([1.0])
        n = len(scores)
        for i in range(1, self.num_permutations + 1):
            permuted = self.rng.permutation(n)
            scores_permuted = scores[permuted]
            test_statistic_permutation = self.test_statistic(scores_permuted)

            if wealth_bin[-1] * p_zero * (i + 1) / rank < self.test_level:
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
            strategy_to_current_wealth = {
                "binomial": wealth_bin[-1],
                "aggressive": wealth_agg[-1],
                "binomial_mixture": wealth_bm[-1],
            }
            current_wealth = strategy_to_current_wealth[self.strategy]
            if (
                current_wealth < self.test_level
                or current_wealth >= 1 / self.test_level
            ):
                break

        strategy_to_wealth = {
            "binomial": wealth_bin,
            "aggressive": wealth_agg,
            "binomial_mixture": wealth_bm,
        }
        wealth_history = strategy_to_wealth[self.strategy]
        running_max_wealth = np.maximum.accumulate(wealth_history)
        self.p_values = np.minimum(1 / running_max_wealth, 1)

        is_exchangeable = bool(self.p_values[-1] > self.test_level)

        return is_exchangeable
