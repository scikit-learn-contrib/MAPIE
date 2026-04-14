from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from mapie.exchangeability_testing.permutation_tests import (
    MapieEstimator,
    PValuePermutationTest,
    SequentialMonteCarloTest,
    TestStatisticOnNonConformityScores,
)
from mapie.classification import CrossConformalClassifier
from mapie.regression import CrossConformalRegressor, JackknifeAfterBootstrapRegressor


class DummyConformityScoreFunction:
    def get_conformity_scores(self, y, y_pred, X=None):
        return np.abs(y - y_pred)


class DummyMapieRegressor:
    def __init__(self):
        self.conformity_score_function_ = DummyConformityScoreFunction()
        self.conformity_scores_ = np.array([99.0])
        self.quantiles_ = np.array([0.9])


class DummyMapieEstimator:
    def __init__(self):
        self._mapie_regressor = DummyMapieRegressor()
        self._is_fitted = True
        self._is_conformalized = False
        self._predict_params = {"stale": True}
        self.conformity_scores_ = np.array([42.0])
        self.quantiles_ = np.array([0.5])

    def predict(self, X):
        return np.zeros(len(X))

    def fit(self, X, y):
        self._is_fitted = True
        return self

    def conformalize(self, X, y):
        if self._is_conformalized:
            raise ValueError("conformalize method already called")
        self._is_conformalized = True
        self._predict_params = {}
        self.conformity_scores_ = np.abs(y - self.predict(X))
        self._mapie_regressor.conformity_scores_ = np.abs(y - self.predict(X))
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
    return X, y


class TestPValuePermutationTest:
    def test_init_copies_provided_estimator(self) -> None:
        estimator = DummyMapieEstimator()
        estimator._is_conformalized = True

        test = PValuePermutationTest(
            random_state=123,
            num_permutations=10,
            mapie_estimator=cast(MapieEstimator, estimator),
        )

        assert test.mapie_estimator is not estimator
        assert test.mapie_estimator._is_conformalized is False
        assert test.mapie_estimator._predict_params == {}
        assert not hasattr(test.mapie_estimator, "conformity_scores_")
        assert not hasattr(test.mapie_estimator, "quantiles_")
        assert not hasattr(test.mapie_estimator._mapie_regressor, "conformity_scores_")
        assert not hasattr(test.mapie_estimator._mapie_regressor, "quantiles_")
        assert estimator._is_conformalized is True
        assert estimator._predict_params == {"stale": True}
        assert hasattr(estimator, "conformity_scores_")
        assert hasattr(estimator._mapie_regressor, "conformity_scores_")

    def test_compute_scores_uses_predict_if_y_pred_is_none(self) -> None:
        X = np.arange(8, dtype=float).reshape(-1, 1)
        y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        test = PValuePermutationTest(
            random_state=123,
            num_permutations=10,
            mapie_estimator=cast(MapieEstimator, DummyMapieEstimator()),
        )

        scores = test._compute_non_conformity_scores(X, y)

        np.testing.assert_allclose(scores, y)

    def test_run_fits_provided_unfitted_estimator(self, toy_exchangeability_data) -> None:
        X, y = toy_exchangeability_data
        estimator = DummyMapieEstimator()
        estimator._is_fitted = False
        test = PValuePermutationTest(
            random_state=123,
            num_permutations=10,
            mapie_estimator=cast(MapieEstimator, estimator),
        )

        is_exchangeable = test.run(X, y)

        assert isinstance(is_exchangeable, bool)
        assert test.mapie_estimator._is_fitted is True

    def test_run_is_reproducible_with_fixed_random_state(
        self, toy_exchangeability_data
    ) -> None:
        X, y = toy_exchangeability_data
        estimator_1 = DummyMapieEstimator()
        estimator_2 = DummyMapieEstimator()

        test_1 = PValuePermutationTest(
            random_state=42,
            num_permutations=50,
            mapie_estimator=cast(MapieEstimator, estimator_1),
        )
        test_2 = PValuePermutationTest(
            random_state=42,
            num_permutations=50,
            mapie_estimator=cast(MapieEstimator, estimator_2),
        )

        is_exchangeable_1 = test_1.run(X, y)
        is_exchangeable_2 = test_2.run(X, y)

        assert is_exchangeable_1 == is_exchangeable_2
        np.testing.assert_allclose(test_1.p_values, test_2.p_values)

    def test_run_sets_expected_outputs(self, toy_exchangeability_data) -> None:
        X, y = toy_exchangeability_data
        test = PValuePermutationTest(
            random_state=7,
            num_permutations=30,
            mapie_estimator=cast(MapieEstimator, DummyMapieEstimator()),
        )

        is_exchangeable = test.run(X, y)

        assert isinstance(is_exchangeable, bool)
        assert test.p_values.shape == (31,)
        assert test.p_values[0] == 1.0
        assert np.all((test.p_values >= 0.0) & (test.p_values <= 1.0))
        assert is_exchangeable == bool(test.p_values[-1] > test.delta)

    @pytest.mark.parametrize(
        "estimator",
        [
            CrossConformalRegressor(cv=3),
            CrossConformalClassifier(cv=3),
            JackknifeAfterBootstrapRegressor(),
        ],
    )
    def test_init_rejects_order_agnostic_estimators(self, estimator) -> None:
        with pytest.raises(
            ValueError,
            match="are not supported in permutation tests",
        ):
            PValuePermutationTest(
                random_state=123,
                num_permutations=10,
                mapie_estimator=cast(Any, estimator),
            )

class TestSequentialMonteCarloTest:
    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match=r"Unknown strategy"):
            SequentialMonteCarloTest(strategy=cast(Any, "unknown"))

    @pytest.mark.parametrize("strategy", ["aggressive", "binomial", "binomial_mixture"])
    def test_run_sets_expected_outputs(
        self, strategy, toy_exchangeability_data
    ) -> None:
        X, y = toy_exchangeability_data
        test = SequentialMonteCarloTest(
            strategy=strategy,
            random_state=7,
            num_permutations=80,
            mapie_estimator=cast(MapieEstimator, DummyMapieEstimator()),
        )

        is_exchangeable = test.run(X, y)

        assert isinstance(is_exchangeable, bool)
        assert test.p_values.ndim == 1
        assert 1 <= len(test.p_values) <= 81
        assert np.all((test.p_values >= 0.0) & (test.p_values <= 1.0))
        assert is_exchangeable == bool(test.p_values[-1] < test.delta)

    def test_run_is_reproducible_with_fixed_random_state(
        self, toy_exchangeability_data
    ) -> None:
        X, y = toy_exchangeability_data
        test_1 = SequentialMonteCarloTest(
            strategy="binomial",
            random_state=123,
            num_permutations=60,
            mapie_estimator=cast(MapieEstimator, DummyMapieEstimator()),
        )
        test_2 = SequentialMonteCarloTest(
            strategy="binomial",
            random_state=123,
            num_permutations=60,
            mapie_estimator=cast(MapieEstimator, DummyMapieEstimator()),
        )

        is_exchangeable_1 = test_1.run(X, y)
        is_exchangeable_2 = test_2.run(X, y)

        assert is_exchangeable_1 == is_exchangeable_2
        np.testing.assert_allclose(test_1.p_values, test_2.p_values)
