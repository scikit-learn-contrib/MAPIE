from __future__ import annotations

import numpy as np
import pytest

from mapie.exchangeability_testing.permutation_tests import (
    PValuePermutationTest,
    SequentialMonteCarloTest,
    TestStatisticOnNonConformityScores,
)


class DummyConformityScoreFunction:
    def get_conformity_scores(self, y, y_pred, X=None):
        return np.abs(y - y_pred)


class DummyMapieRegressor:
    def __init__(self):
        self.conformity_score_function_ = DummyConformityScoreFunction()


class DummyMapieEstimator:
    def __init__(self):
        self._mapie_regressor = DummyMapieRegressor()

    def predict(self, X):
        return np.zeros(len(X))

    def fit(self, X, y):
        return self


class TestStatisticOnNonConformityScoresClass:
    def test_compute(self) -> None:
        statistic = TestStatisticOnNonConformityScores()
        scores = np.array([1.0, 3.0, 2.0, 2.0])
        assert statistic.compute(scores) == 0.0

    def test_call_is_alias_of_compute(self) -> None:
        statistic = TestStatisticOnNonConformityScores()
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        assert statistic(scores) == statistic.compute(scores)


@pytest.fixture
def toy_exchangeability_data():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.concatenate((np.zeros(10), np.ones(10)))
    y_pred = np.zeros_like(y)
    return X, y, y_pred


class TestPValuePermutationTest:
    def test_compute_scores_uses_predict_if_y_pred_is_none(self) -> None:
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        test = PValuePermutationTest(
            random_state=123,
            num_permutations=10,
            mapie_estimator=DummyMapieEstimator(),
        )

        scores = test._compute_non_conformity_scores(X, y, y_pred=None)

        np.testing.assert_allclose(scores, y)

    def test_run_is_reproducible_with_fixed_random_state(
        self, toy_exchangeability_data
    ) -> None:
        X, y, y_pred = toy_exchangeability_data
        estimator_1 = DummyMapieEstimator()
        estimator_2 = DummyMapieEstimator()

        test_1 = PValuePermutationTest(
            random_state=42, num_permutations=50, mapie_estimator=estimator_1
        )
        test_2 = PValuePermutationTest(
            random_state=42, num_permutations=50, mapie_estimator=estimator_2
        )

        is_exchangeable_1 = test_1.run(X, y, y_pred=y_pred)
        is_exchangeable_2 = test_2.run(X, y, y_pred=y_pred)

        assert is_exchangeable_1 == is_exchangeable_2
        np.testing.assert_allclose(test_1.p_values, test_2.p_values)

    def test_run_sets_expected_outputs(self, toy_exchangeability_data) -> None:
        X, y, y_pred = toy_exchangeability_data
        test = PValuePermutationTest(
            random_state=7, num_permutations=30, mapie_estimator=DummyMapieEstimator()
        )

        is_exchangeable = test.run(X, y, y_pred=y_pred)

        assert isinstance(is_exchangeable, bool)
        assert test.p_values.shape == (31,)
        assert test.p_values[0] == 1.0
        assert np.all((test.p_values >= 0.0) & (test.p_values <= 1.0))
        assert is_exchangeable == bool(test.p_values[-1] > test.delta)


class TestSequentialMonteCarloTest:
    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown strategy"):
            SequentialMonteCarloTest(strategy="unknown")

    @pytest.mark.parametrize("strategy", ["aggressive", "binomial", "binomial_mixture"])
    def test_run_sets_expected_outputs(self, strategy, toy_exchangeability_data) -> None:
        X, y, y_pred = toy_exchangeability_data
        test = SequentialMonteCarloTest(
            strategy=strategy,
            random_state=7,
            num_permutations=80,
            mapie_estimator=DummyMapieEstimator(),
        )

        is_exchangeable = test.run(X, y, y_pred=y_pred)

        assert isinstance(is_exchangeable, bool)
        assert test.p_values.ndim == 1
        assert 1 <= len(test.p_values) <= 81
        assert np.all((test.p_values >= 0.0) & (test.p_values <= 1.0))
        assert is_exchangeable == bool(test.p_values[-1] < test.delta)

    def test_run_is_reproducible_with_fixed_random_state(
        self, toy_exchangeability_data
    ) -> None:
        X, y, y_pred = toy_exchangeability_data
        test_1 = SequentialMonteCarloTest(
            strategy="binomial",
            random_state=123,
            num_permutations=60,
            mapie_estimator=DummyMapieEstimator(),
        )
        test_2 = SequentialMonteCarloTest(
            strategy="binomial",
            random_state=123,
            num_permutations=60,
            mapie_estimator=DummyMapieEstimator(),
        )

        is_exchangeable_1 = test_1.run(X, y, y_pred=y_pred)
        is_exchangeable_2 = test_2.run(X, y, y_pred=y_pred)

        assert is_exchangeable_1 == is_exchangeable_2
        np.testing.assert_allclose(test_1.p_values, test_2.p_values)
