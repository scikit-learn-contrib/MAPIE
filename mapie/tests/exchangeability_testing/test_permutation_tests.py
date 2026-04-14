from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from mapie.classification import CrossConformalClassifier
from mapie.exchangeability_testing.permutation_tests import (
    MapieEstimator,
    PermutationTest,
    PValuePermutationTest,
    SequentialMonteCarloTest,
    TestStatistic,
    TestStatisticOnLabeledDataset,
    TestStatisticOnNonConformityScores,
    TestStatisticOnUnlabeledDataset,
)
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


class DummyMapieClassifier:
    def __init__(self):
        self.conformity_scores_ = np.array([99.0])


class DummyClassificationEstimator:
    def __init__(self, prefit=True):
        self._mapie_classifier = DummyMapieClassifier()
        self._is_fitted = prefit

    def fit(self, X, y):
        self._is_fitted = True
        return self

    def conformalize(self, X, y):
        self._mapie_classifier.conformity_scores_ = np.arange(len(y), dtype=float)
        return self


class DummyUnknownEstimator:
    def __init__(self):
        self._is_fitted = True

    def fit(self, X, y):
        return self

    def conformalize(self, X, y):
        return self


class ConstantStatistic:
    def __call__(self, scores):
        return 1.0


class TestStatisticOnNonConformityScoresClass:
    def test_compute(self) -> None:
        statistic = TestStatisticOnNonConformityScores()
        scores = np.array([1.0, 3.0, 2.0, 2.0])
        assert statistic.compute(scores) == 0.0

    def test_call_is_alias_of_compute(self) -> None:
        statistic = TestStatisticOnNonConformityScores()
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        assert statistic(scores) == statistic.compute(scores)

    def test_abstract_base_methods_raise(self) -> None:
        with pytest.raises(NotImplementedError):
            TestStatistic.compute(object())
        with pytest.raises(NotImplementedError):
            TestStatisticOnLabeledDataset.compute(object())
        with pytest.raises(NotImplementedError):
            TestStatisticOnUnlabeledDataset.compute(object())
        with pytest.raises(NotImplementedError):
            PermutationTest.run(object(), np.array([[0.0]]), np.array([0.0]))


@pytest.fixture
def toy_exchangeability_data():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.concatenate((np.zeros(10), np.ones(10)))
    return X, y


class TestPValuePermutationTest:
    def test_dummy_conformity_score_function(self) -> None:
        score_function = DummyConformityScoreFunction()
        scores = score_function.get_conformity_scores(
            np.array([1.0, 2.0]),
            np.array([0.0, 1.5]),
        )
        np.testing.assert_allclose(scores, np.array([1.0, 0.5]))

    def test_init_rejects_invalid_test_level(self) -> None:
        with pytest.raises(ValueError, match="test_level must be in"):
            PValuePermutationTest(test_level=1.0)
        with pytest.raises(ValueError, match="test_level must be in"):
            PValuePermutationTest(test_level=0.0)

    def test_init_copies_provided_estimator(self) -> None:
        estimator = DummyMapieEstimator()
        estimator._is_conformalized = True

        test = PValuePermutationTest(
            random_state=123,
            num_permutations=10,
            mapie_estimator=cast(MapieEstimator, estimator),
        )
        estimator_copy = cast(DummyMapieEstimator, test.mapie_estimator)

        assert test.mapie_estimator is not estimator
        assert estimator_copy._is_conformalized is False
        assert estimator_copy._predict_params == {}
        assert not hasattr(estimator_copy, "conformity_scores_")
        assert not hasattr(estimator_copy, "quantiles_")
        assert not hasattr(estimator_copy._mapie_regressor, "conformity_scores_")
        assert not hasattr(estimator_copy._mapie_regressor, "quantiles_")
        assert estimator._is_conformalized is True
        assert estimator._predict_params == {"stale": True}
        assert hasattr(estimator, "conformity_scores_")
        assert hasattr(estimator._mapie_regressor, "conformity_scores_")

    def test_init_copies_classifier_estimator_without_optional_attrs(self) -> None:
        estimator = DummyClassificationEstimator()

        test = PValuePermutationTest(
            random_state=123,
            num_permutations=10,
            mapie_estimator=cast(Any, estimator),
        )
        estimator_copy = cast(DummyClassificationEstimator, test.mapie_estimator)

        assert estimator_copy is not estimator
        assert not hasattr(estimator_copy, "_is_conformalized")
        assert not hasattr(estimator_copy, "_predict_params")
        assert not hasattr(estimator_copy, "conformity_scores_")
        assert not hasattr(estimator_copy, "quantiles_")
        assert not hasattr(estimator_copy._mapie_classifier, "conformity_scores_")
        assert not hasattr(estimator_copy._mapie_classifier, "quantiles_")

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

    def test_run_fits_provided_unfitted_estimator(
        self, toy_exchangeability_data
    ) -> None:
        X, y = toy_exchangeability_data
        estimator = DummyMapieEstimator()
        estimator._is_fitted = False
        test = PValuePermutationTest(
            random_state=123,
            num_permutations=10,
            mapie_estimator=cast(MapieEstimator, estimator),
        )

        is_exchangeable = test.run(X, y)
        estimator_copy = cast(DummyMapieEstimator, test.mapie_estimator)

        assert isinstance(is_exchangeable, bool)
        assert estimator_copy._is_fitted is True

    def test_infer_task_from_estimator_and_target_type(self) -> None:
        classification_test = PValuePermutationTest(
            mapie_estimator=cast(Any, DummyClassificationEstimator())
        )
        regression_test = PValuePermutationTest(
            mapie_estimator=cast(MapieEstimator, DummyMapieEstimator())
        )
        default_test = PValuePermutationTest()

        assert classification_test._infer_task(np.array([0, 1])) == "classification"
        assert regression_test._infer_task(np.array([0.1, 0.2])) == "regression"
        assert default_test._infer_task(np.array([0, 1, 0, 1])) == "classification"
        assert default_test._infer_task(np.array([0.1, 0.2, 0.3])) == "regression"

    def test_infer_task_raises_for_unknown_estimator_or_target(self) -> None:
        test_with_unknown_estimator = PValuePermutationTest(
            mapie_estimator=cast(Any, DummyUnknownEstimator())
        )
        default_test = PValuePermutationTest()

        with pytest.raises(ValueError, match="Unable to infer the task"):
            test_with_unknown_estimator._infer_task(np.array([0, 1]))
        with pytest.raises(ValueError, match="Unknown type of target"):
            default_test._infer_task(np.array([[[1.0]], [[2.0]]]))

    def test_unknown_estimator_methods_return_self(self) -> None:
        estimator = DummyUnknownEstimator()

        assert estimator.fit(np.array([[0.0]]), np.array([0.0])) is estimator
        assert estimator.conformalize(np.array([[0.0]]), np.array([0.0])) is estimator

    def test_compute_scores_with_default_classification_estimator(
        self, monkeypatch
    ) -> None:
        from mapie.exchangeability_testing import permutation_tests

        monkeypatch.setattr(
            permutation_tests,
            "SplitConformalClassifier",
            DummyClassificationEstimator,
        )
        test = PValuePermutationTest()
        test.task = "classification"

        scores = test._compute_non_conformity_scores(
            np.arange(6, dtype=float).reshape(-1, 1),
            np.array([0, 1, 0, 1, 0, 1]),
        )

        np.testing.assert_allclose(scores, np.arange(5, dtype=float))

    def test_compute_scores_raises_for_unknown_task(self) -> None:
        test = PValuePermutationTest()
        test.task = cast(Any, "unknown")

        with pytest.raises(ValueError, match="Unknown task type"):
            test._compute_non_conformity_scores(
                np.arange(6, dtype=float).reshape(-1, 1),
                np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            )

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

    def test_init_rejects_invalid_test_level(self) -> None:
        with pytest.raises(ValueError, match="test_level must be in"):
            SequentialMonteCarloTest(strategy="binomial", test_level=1.0)
        with pytest.raises(ValueError, match="test_level must be in"):
            SequentialMonteCarloTest(strategy="binomial", test_level=0.0)

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

    def test_run_without_early_stopping_and_with_rank_update(
        self, toy_exchangeability_data
    ) -> None:
        X, y = toy_exchangeability_data
        test = SequentialMonteCarloTest(
            strategy="binomial_mixture",
            random_state=123,
            num_permutations=3,
            mapie_estimator=cast(MapieEstimator, DummyMapieEstimator()),
        )
        test.test_statistic = ConstantStatistic()

        is_exchangeable = test.run(X, y)

        assert isinstance(is_exchangeable, bool)
        assert len(test.p_values) == 4
        assert np.all((test.p_values >= 0.0) & (test.p_values <= 1.0))
