from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict
from inspect import signature

import pytest
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.dummy import DummyClassifier

from mapie.classification import MapieClassifier

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([1, 3, 0, 2, 1, 0])


class DumbClassifier:

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> DumbClassifier:
        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return X


METHODS = ["naive"]

Params = TypedDict(
    "Params", {"method": str, "cv": Optional[str]}
)

STRATEGIES = {
    "naive": Params(method="naive", cv=None)
}

X_lr, y_lr = make_classification(
    n_samples=500, n_features=10, n_informative=3, n_classes=4, random_state=1
)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieClassifier()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie = MapieClassifier()
    assert mapie.estimator is None
    assert mapie.method == "naive"
    assert mapie.cv is None
    assert mapie.verbose == 0
    assert mapie.n_jobs is None


def test_default_sample_weight() -> None:
    """Test default sample weights."""
    mapie = MapieClassifier()
    assert signature(mapie.fit).parameters["sample_weight"].default is None


def test_fit() -> None:
    """Test that fit raises no errors."""
    mapie = MapieClassifier()
    mapie.fit(X_toy, y_toy)


def test_fit_predict() -> None:
    """Test that fit-predict raises no errors."""
    mapie = MapieClassifier()
    mapie.fit(X_toy, y_toy)
    mapie.predict(X_toy)


def test_no_fit_predict() -> None:
    """Test that predict before fit raises errors"""
    mapie = MapieClassifier(estimator=DummyClassifier())
    with pytest.raises(NotFittedError, match=r".*not fitted.*"):
        mapie.predict(X_toy)


def test_none_estimator() -> None:
    """Test that None estimator defaults to LogisticRegression."""
    mapie = MapieClassifier(estimator=None)
    mapie.fit(X_toy, y_toy)
    assert isinstance(mapie.single_estimator_, LogisticRegression)


@pytest.mark.parametrize("estimator", [0, "estimator", KFold(), ["a", "b"]])
def test_invalid_estimator(estimator: Any) -> None:
    """Test that invalid estimators raise errors."""
    mapie = MapieClassifier(estimator=estimator)
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    mapie = MapieClassifier(estimator=DummyClassifier(),
                            **STRATEGIES[strategy])
    mapie.fit(X_toy, y_toy)
    assert isinstance(mapie.single_estimator_, DummyClassifier)


@pytest.mark.parametrize(
    "estimator", [
        LogisticRegression(),
        make_pipeline(LogisticRegression())
    ]
)
def test_invalid_prefit_estimator(estimator: ClassifierMixin) -> None:
    """Test that non-fitted estimator with prefit cv raise errors."""
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    with pytest.raises(NotFittedError):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize(
    "estimator", [
        LogisticRegression(),
        make_pipeline(LogisticRegression())
    ]
)
def test_valid_prefit_estimator(estimator: ClassifierMixin) -> None:
    """Test that fitted estimators with prefit cv raise no errors."""
    estimator.fit(X_toy, y_toy)
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    mapie.fit(X_toy, y_toy)
    if isinstance(estimator, Pipeline):
        check_is_fitted(mapie.single_estimator_[-1])
    else:
        check_is_fitted(mapie.single_estimator_)
    check_is_fitted(
        mapie,
        [
            "single_estimator_",
            "n_features_in_",
            "n_samples_in_train_"
        ]
    )
    assert mapie.n_features_in_ == 1


def test_invalid_prefit_estimator_shape() -> None:
    """
    Test that estimators fitted with a wrong number of features raise errors.
    """
    estimator = LogisticRegression().fit(X_lr, y_lr)
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    with pytest.raises(ValueError, match=r".*mismatch between.*"):
        mapie.fit(X_toy, y_toy)


def test_valid_prefit_estimator_shape_no_n_features_in() -> None:
    """
    Test that estimators fitted with a right number of features
    but missing an n_features_in_ attribute raise no errors.
    """
    estimator = DumbClassifier().fit(X_lr, y_lr)
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    mapie.fit(X_lr, y_lr)
    assert mapie.n_features_in_ == 10


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    mapie_single = MapieClassifier(n_jobs=1, **STRATEGIES[strategy])
    mapie_multi = MapieClassifier(n_jobs=-1, **STRATEGIES[strategy])
    mapie_single.fit(X_toy, y_toy)
    mapie_multi.fit(X_toy, y_toy)
    y_pred_single = mapie_single.predict(X_toy)
    y_pred_multi = mapie_multi.predict(X_toy)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(strategy: str) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X_lr)
    mapie0 = MapieClassifier(**STRATEGIES[strategy])
    mapie1 = MapieClassifier(**STRATEGIES[strategy])
    mapie2 = MapieClassifier(**STRATEGIES[strategy])
    mapie0.fit(X_lr, y_lr, sample_weight=None)
    mapie1.fit(X_lr, y_lr, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X_lr, y_lr, sample_weight=np.ones(shape=n_samples)*5)
    y_pred0 = mapie0.predict(X_lr)
    y_pred1 = mapie1.predict(X_lr)
    y_pred2 = mapie2.predict(X_lr)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)


@pytest.mark.parametrize("n_jobs", ["dummy", 0, 1.5, [1, 2]])
def test_invalid_n_jobs(n_jobs: Any) -> None:
    """Test that invalid n_jobs raise errors."""
    mapie = MapieClassifier(n_jobs=n_jobs)
    with pytest.raises(ValueError, match=r".*Invalid n_jobs argument.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("n_jobs", [-5, -1, 1, 4])
def test_valid_n_jobs(n_jobs: Any) -> None:
    """Test that valid n_jobs raise no errors."""
    mapie = MapieClassifier(n_jobs=n_jobs)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("verbose", ["dummy", -1, 1.5, [1, 2]])
def test_invalid_verbose(verbose: Any) -> None:
    """Test that invalid verboses raise errors."""
    mapie = MapieClassifier(verbose=verbose)
    with pytest.raises(ValueError, match=r".*Invalid verbose argument.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("verbose", [0, 10, 50])
def test_valid_verbose(verbose: Any) -> None:
    """Test that valid verboses raise no errors."""
    mapie = MapieClassifier(verbose=verbose)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize(
    "cv", [-3.14, -2, 0, 1, "cv", DummyClassifier(), [1, 2]]
)
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie = MapieClassifier(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [None])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = MapieClassifier(cv=cv)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", [0, 1, "jackknife", "cv", ["base", "plus"]])
def test_invalid_method(method: str) -> None:
    """Test that invalid methods raise errors."""
    mapie = MapieClassifier(method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", METHODS)
def test_valid_method(method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie = MapieClassifier(method=method)
    mapie.fit(X_toy, y_toy)
    check_is_fitted(
        mapie,
        [
            "single_estimator_",
            "n_features_in_",
            "n_samples_in_train_"
        ]
    )


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2, 2.5, "a", [[0.5]], ["a", "b"]])
def test_invalid_alpha(alpha: Any) -> None:
    """Test that invalid alphas raise errors."""
    mapie = MapieClassifier()
    mapie.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        mapie.predict(X_toy, alpha=alpha)


@pytest.mark.parametrize(
    "alpha",
    [
        np.linspace(0.05, 0.95, 5),
        [0.05, 0.95],
        (0.05, 0.95),
        np.array([0.05, 0.95]),
        None
    ]
)
def test_valid_alpha(alpha: Any) -> None:
    """Test that valid alphas raise no errors."""
    mapie = MapieClassifier()
    mapie.fit(X_toy, y_toy)
    mapie.predict(X_toy, alpha=alpha)


@pytest.mark.parametrize(
    "classifier",
    [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
)
@pytest.mark.parametrize("CV", ["prefit", None])
def test_none_alpha_results(classifier: Any, CV: Any) -> None:
    """
    Test that alpha set to None in MapieClassifier gives same predictions
    as base Classifier.
    """
    estimator = classifier
    estimator.fit(X_lr, y_lr)
    y_pred_est = estimator.predict(X_lr)
    mapie = MapieClassifier(estimator=estimator, cv=CV)
    mapie.fit(X_lr, y_lr)
    y_pred_mapie = mapie.predict(X_lr)
    np.testing.assert_allclose(y_pred_est, y_pred_mapie)


def test_valid_prediction() -> None:
    X, Y = make_classification(
        n_samples=10000,
        n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, n_classes=4)
    x_train_cal, x_test, y_train_cal, y_test = train_test_split(
        X, Y, test_size=1/10
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_train_cal, y_train_cal, test_size=1/9
    )
    model = LogisticRegression(multi_class="multinomial")
    model.fit(x_train, y_train)
    mapie = MapieClassifier(estimator=model, cv="prefit")
    mapie.fit(x_cal, y_cal)
    mapie.predict(x_test, alpha=0.05)
