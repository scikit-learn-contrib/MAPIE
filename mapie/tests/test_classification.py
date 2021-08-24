from __future__ import annotations
from typing import Any, Optional, Tuple
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
from sklearn.naive_bayes import GaussianNB

from mapie.classification import MapieClassifier


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


METHODS = ["score"]

Params = TypedDict(
    "Params", {"method": str, "cv": Optional[str]}
)

STRATEGIES = {
    "score": Params(method="score", cv=None)
}

X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
y_toy_mapie = [
    [True, False, False],
    [True, False, False],
    [True, False, False],
    [True, True, False],
    [False, True, False],
    [False, True, True],
    [False, False, True],
    [False, False, True],
    [False, False, True]
]

X_lr, y_lr = make_classification(
    n_samples=500, n_features=10, n_informative=3, n_classes=4, random_state=1
)

X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X_lr, y_lr, test_size=1/10
    )

X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_cal, y_train_cal, test_size=1/9
    )


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieClassifier()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie = MapieClassifier()
    assert mapie.estimator is None
    assert mapie.method == "score"
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
    mapie.fit(X_lr, y_lr)


def test_fit_predict() -> None:
    """Test that fit-predict raises no errors."""
    mapie = MapieClassifier()
    mapie.fit(X_lr, y_lr)
    mapie.predict(X_lr)


def test_no_fit_predict() -> None:
    """Test that predict before fit raises errors"""
    mapie = MapieClassifier(estimator=DummyClassifier())
    with pytest.raises(NotFittedError, match=r".*not fitted.*"):
        mapie.predict(X_toy)


def test_none_estimator() -> None:
    """Test that None estimator defaults to LogisticRegression."""
    mapie = MapieClassifier(estimator=None)
    mapie.fit(X_lr, y_lr)
    assert isinstance(mapie.single_estimator_, LogisticRegression)


@pytest.mark.parametrize("estimator", [0, "estimator", KFold(), ["a", "b"]])
def test_invalid_estimator(estimator: Any) -> None:
    """Test that invalid estimators raise errors."""
    mapie = MapieClassifier(estimator=estimator)
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    mapie = MapieClassifier(estimator=DummyClassifier(),
                            **STRATEGIES[strategy])
    mapie.fit(X_lr, y_lr)
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
    estimator.fit(X_lr, y_lr)
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    mapie.fit(X_lr, y_lr)
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
    assert mapie.n_features_in_ == 10


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
    mapie_single.fit(X_lr, y_lr)
    mapie_multi.fit(X_lr, y_lr)
    y_pred_single = mapie_single.predict(X_lr)
    y_pred_multi = mapie_multi.predict(X_lr)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(strategy: str) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = 500
    mapie0 = MapieClassifier(**STRATEGIES[strategy])
    mapie1 = MapieClassifier(**STRATEGIES[strategy])
    mapie0.fit(X_lr, y_lr, sample_weight=None)
    mapie1.fit(X_lr, y_lr, sample_weight=np.ones(shape=n_samples))
    y_pred0 = mapie0.predict(X_lr)
    y_pred1 = mapie1.predict(X_lr)
    np.testing.assert_allclose(y_pred0, y_pred1)


@pytest.mark.parametrize("n_jobs", ["dummy", 0, 1.5, [1, 2]])
def test_invalid_n_jobs(n_jobs: Any) -> None:
    """Test that invalid n_jobs raise errors."""
    mapie = MapieClassifier(n_jobs=n_jobs)
    with pytest.raises(ValueError, match=r".*Invalid n_jobs argument.*"):
        mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize("n_jobs", [-5, -1, 1, 4])
def test_valid_n_jobs(n_jobs: Any) -> None:
    """Test that valid n_jobs raise no errors."""
    mapie = MapieClassifier(n_jobs=n_jobs)
    mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize("verbose", ["dummy", -1, 1.5, [1, 2]])
def test_invalid_verbose(verbose: Any) -> None:
    """Test that invalid verboses raise errors."""
    mapie = MapieClassifier(verbose=verbose)
    with pytest.raises(ValueError, match=r".*Invalid verbose argument.*"):
        mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize("verbose", [0, 10, 50])
def test_valid_verbose(verbose: Any) -> None:
    """Test that valid verboses raise no errors."""
    mapie = MapieClassifier(verbose=verbose)
    mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize(
    "cv", [-3.14, -2, 0, 1, "cv", DummyClassifier(), [1, 2]]
)
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie = MapieClassifier(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize("cv", [None, "prefit"])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    if cv == "prefit":
        model = LogisticRegression(multi_class="multinomial")
        model.fit(X_train, y_train)
        mapie = MapieClassifier(estimator=model, cv=cv)
        mapie.fit(X_cal, y_cal)
    else:
        mapie = MapieClassifier(cv=cv)
        mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize("method", [0, 1, "jackknife", "cv", ["base", "plus"]])
def test_invalid_method(method: str) -> None:
    """Test that invalid methods raise errors."""
    mapie = MapieClassifier(method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.fit(X_lr, y_lr)


@pytest.mark.parametrize("method", METHODS)
def test_valid_method(method: str) -> None:
    """Test that valid methods raise no errors."""
    mapie = MapieClassifier(method=method)
    mapie.fit(X_lr, y_lr)
    check_is_fitted(
        mapie,
        [
            "single_estimator_",
            "n_features_in_",
            "n_samples_in_train_",
            "scores_"
        ]
    )


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2, 2.5, "a", [[0.5]], ["a", "b"]])
def test_invalid_alpha(alpha: Any) -> None:
    """Test that invalid alphas raise errors."""
    mapie = MapieClassifier()
    mapie.fit(X_lr, y_lr)
    with pytest.raises(ValueError, match=r".*Invalid alpha.*"):
        mapie.predict(X_lr, alpha=alpha)


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
    mapie.fit(X_lr, y_lr)
    mapie.predict(X_lr, alpha=alpha)


@pytest.mark.parametrize(
    "classifier",
    [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
)
@pytest.mark.parametrize("CV", ["prefit"])
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


@pytest.mark.parametrize(
    "alpha",
    [
        [0.05, 0.95],
        (0.05, 0.95),
        np.array([0.05, 0.95]),
        None
    ]
)
def test_valid_prediction(alpha: Any) -> None:
    """Test fit and predict. """
    model = LogisticRegression(multi_class="multinomial")
    model.fit(X_train, y_train)
    mapie = MapieClassifier(estimator=model, cv="prefit")
    mapie.fit(X_cal, y_cal)
    mapie.predict(X_test, alpha=alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X_lr, y_lr), (X_test, y_test)])
@pytest.mark.parametrize("alpha", [0.1, [0.1, 0.2], (0.1, 0.2)])
def test_predict_output_shape(
    strategy: str,
    alpha: Any,
    dataset: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test predict output shape."""
    mapie = MapieClassifier(**STRATEGIES[strategy])
    X, y = dataset
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=alpha)
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pis.shape == (X.shape[0], 4, n_alpha)
    assert y_pred.shape == (X.shape[0],)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie = MapieClassifier(**STRATEGIES[strategy])
    mapie.fit(X_lr, y_lr)
    _, y_pis = mapie.predict(X_lr, alpha=[0.1, 0.1])
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 0, 1])
    np.testing.assert_allclose(y_pis[:, 1, 0], y_pis[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str,
    alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    mapie = MapieClassifier(**STRATEGIES[strategy])
    mapie.fit(X_lr, y_lr)
    y_pred_float1, y_pis_float1 = mapie.predict(X_lr, alpha=alpha[0])
    y_pred_float2, y_pis_float2 = mapie.predict(X_lr, alpha=alpha[1])
    y_pred_array, y_pis_array = mapie.predict(X_lr, alpha=alpha)
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_pis_float1[:, :, 0], y_pis_array[:, :, 0])
    np.testing.assert_allclose(y_pis_float2[:, :, 0], y_pis_array[:, :, 1])


def test_toy_dataset_predictions():
    """Test prediction sets estimated by MapieClassifier on a toy dataset"""
    clf = GaussianNB().fit(X_toy, y_toy)
    mapie = MapieClassifier(estimator=clf, cv="prefit").fit(X_toy, y_toy)
    _, y_pi_mapie = mapie.predict(X_toy, alpha=0.1)
    np.testing.assert_allclose(y_pi_mapie[:, :, 0], y_toy_mapie)
