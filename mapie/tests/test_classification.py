from __future__ import annotations
from typing import Any, Optional, Tuple, Union, Iterable
from typing_extensions import TypedDict
from inspect import signature

import pytest
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB

from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from mapie._typing import ArrayLike


METHODS = ["score", "cumulated_score"]
WRONG_METHODS = ["scores", "cumulated", "test", ""]
WRONG_INCLUDE_LABELS = ["randomised", "True", "False", "", "other"]
Y_PRED_PROBA = [
    np.array(
        [
            [.8, .01, .1, .05],
            [1, .1, 0, 0]
        ]
    ),
    np.array(
        [
            [1, .0001, 0]
        ]
    ),
    np.array(
        [
            [.8, .1, .05, .05],
            [.9, .01, .04, .06]
        ]
    ),
    np.array(
        [
            [.8, .1, .02, .05],
            [.9, .01, .03, .06]
        ]
    )
]
Params = TypedDict(
    "Params",
    {
        "method": str,
        "cv": Optional[str],
        "random_state": Optional[int]
    }
)
ParamsPredict = TypedDict(
    "ParamsPredict",
    {
        "include_last_label": Union[bool, str],
    }
)

STRATEGIES = {
    "score": (
        Params(
            method="score",
            cv="prefit",
            random_state=None
        ),
        ParamsPredict(
            include_last_label=False
        )
    ),
    "cumulated_score_include": (
        Params(
            method="cumulated_score",
            cv="prefit",
            random_state=42
        ),
        ParamsPredict(
            include_last_label=True
        )
    ),
    "cumulated_score_not_include": (
        Params(
            method="cumulated_score",
            cv="prefit",
            random_state=42
        ),
        ParamsPredict(
            include_last_label=False
        )
    ),
    "cumulated_score_randomized": (
        Params(
            method="cumulated_score",
            cv="prefit",
            random_state=42
        ),
        ParamsPredict(
            include_last_label='randomized'
        )
    ),
}

COVERAGES = {
    "score": 7 / 9,
    "cumulated_score_include": 1,
    "cumulated_score_not_include": 5/9,
    "cumulated_score_randomized": 8/9,
}

y_toy_mapie = {
    "score": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [True, True, False],
        [False, True, False],
        [False, True, True],
        [False, False, True],
        [False, False, True],
        [False, False, True],
    ],
    "cumulated_score_include": [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, True],
        [False, True, True],
    ],
    "cumulated_score_not_include": [
        [True, False, False],
        [True, False, False],
        [True, False, False],
        [False, True, False],
        [True, True, True],
        [False, True, False],
        [False, False, True],
        [False, False, True],
        [False, False, True],
    ],
    "cumulated_score_randomized": [
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [False, True, True],
        [False, False, True],
        [False, True, True],
        [False, True, True],
    ],
}
X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.array([0, 0, 1, 0, 1, 2, 1, 2, 2])

n_classes = 4
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=3,
    n_classes=n_classes,
    random_state=1,
)


class CumulatedScoreClassifier:
    def __init__(self) -> None:
        self.X_calib = np.array([0, 1, 2]).reshape(-1, 1)
        self.y_calib = np.array([0, 1, 2])
        self.y_calib_scores = np.array(
            [[0.750183952461055], [0.029571416154050345], [0.9268006058188594]]
        )
        self.X_test = np.array([3, 4, 5]).reshape(-1, 1)
        self.y_pred_sets = np.array(
            [[True, True, False], [False, True, True], [True, True, False]]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> CumulatedScoreClassifier:
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1, 2, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if np.max(X) <= 2:
            return np.array(
                [[0.4, 0.5, 0.1], [0.2, 0.6, 0.2], [0.6, 0.3, 0.1]]
            )
        else:
            return np.array(
                [[0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.3, 0.5, 0.2]]
            )


class WrongOutputModel():

    def __init__(self, proba_out: ArrayLike):
        self.trained_ = True
        self.proba_out = proba_out

    def fit(self, *args: Any) -> None:
        pass

    def predict_proba(self, *args: Any) -> ArrayLike:
        return self.proba_out

    def predict(self, *args: Any) -> ArrayLike:
        pred = (
            self.proba_out == self.proba_out.max(axis=1)[:, None]
        ).astype(int)

        return pred


def do_nothing(*args: Any) -> None:
    "Mock function that does nothing."
    pass


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieClassifier()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie = MapieClassifier()
    assert mapie.estimator is None
    assert mapie.method == "score"
    assert mapie.cv == "prefit"
    assert mapie.verbose == 0
    assert mapie.random_state is None
    assert mapie.n_jobs is None


def test_default_sample_weight() -> None:
    """Test default sample weights."""
    mapie = MapieClassifier()
    assert signature(mapie.fit).parameters["sample_weight"].default is None


def test_default_alpha() -> None:
    """Test default alpha."""
    mapie = MapieClassifier()
    assert signature(mapie.predict).parameters["alpha"].default is None


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


@pytest.mark.parametrize("method", WRONG_METHODS)
def test_method_error_in_fit(monkeypatch: Any, method: str) -> None:
    """Test else condition for the method in .fit"""
    monkeypatch.setattr(
        MapieClassifier, "_check_parameters", do_nothing
    )
    mapie = MapieClassifier(method=method)
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", WRONG_METHODS)
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_method_error_in_predict(method: Any, alpha: float) -> None:
    """Test else condition for the method in .predict"""
    mapie = MapieClassifier(method='score')
    mapie.fit(X_toy, y_toy)
    mapie.method = method
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie.predict(X_toy, alpha=alpha)


@pytest.mark.parametrize("include_labels", WRONG_INCLUDE_LABELS)
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_include_label_error_in_predict(
    monkeypatch: Any, include_labels: Union[bool, str], alpha: float
) -> None:
    """Test else condition for include_label parameter in .predict"""
    monkeypatch.setattr(
        MapieClassifier,
        "_check_include_last_label",
        do_nothing
    )
    mapie = MapieClassifier(method='cumulated_score')
    mapie.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Invalid include.*"):
        mapie.predict(
            X_toy, alpha=alpha,
            include_last_label=include_labels
        )


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
    clf = LogisticRegression().fit(X_toy, y_toy)
    mapie = MapieClassifier(estimator=clf, **STRATEGIES[strategy][0])
    mapie.fit(X_toy, y_toy)
    assert isinstance(mapie.single_estimator_, LogisticRegression)


@pytest.mark.parametrize(
    "estimator", [LogisticRegression(), make_pipeline(LogisticRegression())]
)
def test_invalid_prefit_estimator(estimator: ClassifierMixin) -> None:
    """Test that non-fitted estimator with prefit cv raise errors."""
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    with pytest.raises(NotFittedError):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize(
    "estimator", [LogisticRegression(), make_pipeline(LogisticRegression())]
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
        mapie, ["single_estimator_", "n_features_in_", "n_samples_val_"]
    )
    assert mapie.n_features_in_ == 1


@pytest.mark.parametrize(
    "method", [0.5, 1, "jackknife", "cv", ["base", "plus"]]
)
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
            "n_samples_val_",
            "conformity_scores_"
        ]
    )


@pytest.mark.parametrize("cv", [None, "prefit"])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    model = LogisticRegression(multi_class="multinomial")
    model.fit(X_toy, y_toy)
    mapie = MapieClassifier(estimator=model, cv=cv)
    mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize(
    "cv", [-3.14, 1.5, -2, 0, 1, "cv", DummyClassifier(), [1, 2]]
)
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie = MapieClassifier(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv argument.*"):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize(
    "include_last_label",
    [-3.14, 1.5, -2, 0, 1, "cv", DummyClassifier(), [1, 2]]
)
def test_invalid_include_last_label(include_last_label: Any) -> None:
    """Test that invalid include_last_label raise errors."""
    mapie = MapieClassifier()
    mapie.fit(X_toy, y_toy)
    with pytest.raises(
        ValueError, match=r".*Invalid include_last_label argument.*"
    ):
        mapie.predict(
            X_toy,
            y_toy,
            include_last_label=include_last_label
        )


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_predict_output_shape(
    strategy: str,
    alpha: Any,
    dataset: Tuple[np.ndarray, np.ndarray]
) -> None:
    """Test predict output shape."""
    args_init, args_predict = STRATEGIES[strategy]
    include_last_label = args_predict['include_last_label']
    mapie = MapieClassifier(**args_init)
    X, y = dataset
    mapie.fit(X, y)
    y_pred, y_ps = mapie.predict(
        X,
        include_last_label=include_last_label,
        alpha=alpha
    )
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == (X.shape[0],)
    assert y_ps.shape == (X.shape[0], len(np.unique(y)), n_alpha)


def test_none_alpha_results() -> None:
    """
    Test that alpha set to None in MapieClassifier gives same predictions
    as base Classifier.
    """
    estimator = LogisticRegression()
    estimator.fit(X, y)
    y_pred_est = estimator.predict(X)
    mapie = MapieClassifier(estimator=estimator, cv="prefit")
    mapie.fit(X, y)
    y_pred_mapie = mapie.predict(X)
    np.testing.assert_allclose(y_pred_est, y_pred_mapie)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args_init, args_predict = STRATEGIES[strategy]
    include_last_label = args_predict['include_last_label']
    mapie = MapieClassifier(**args_init)
    mapie.fit(X, y)
    _, y_ps = mapie.predict(
        X,
        include_last_label=include_last_label,
        alpha=[0.1, 0.1]
    )
    np.testing.assert_allclose(y_ps[:, 0, 0], y_ps[:, 0, 1])
    np.testing.assert_allclose(y_ps[:, 1, 0], y_ps[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    args_init, args_predict = STRATEGIES[strategy]
    include_last_label = args_predict['include_last_label']
    mapie = MapieClassifier(**args_init)
    mapie.fit(X, y)
    y_pred_float1, y_ps_float1 = mapie.predict(
        X,
        include_last_label=include_last_label,
        alpha=alpha[0]
    )
    y_pred_float2, y_ps_float2 = mapie.predict(
        X,
        include_last_label=include_last_label,
        alpha=alpha[1]
    )
    y_pred_array, y_ps_array = mapie.predict(
        X,
        include_last_label=include_last_label,
        alpha=alpha
    )
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_ps_float1[:, :, 0], y_ps_array[:, :, 0])
    np.testing.assert_allclose(y_ps_float2[:, :, 0], y_ps_array[:, :, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that MapieRegressor gives equal predictions
    regardless of number of parallel jobs.
    """
    args_init, args_predict = STRATEGIES[strategy]
    include_last_label = args_predict['include_last_label']
    mapie_single = MapieClassifier(n_jobs=1, **args_init)
    mapie_multi = MapieClassifier(n_jobs=-1, **args_init)
    mapie_single.fit(X_toy, y_toy)
    mapie_multi.fit(X_toy, y_toy)
    y_pred_single, y_ps_single = mapie_single.predict(
        X_toy,
        include_last_label=include_last_label,
        alpha=0.2
    )
    y_pred_multi, y_ps_multi = mapie_multi.predict(
        X_toy,
        include_last_label=include_last_label,
        alpha=0.2
    )
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_ps_single, y_ps_multi)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(
    strategy: str
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    args_init, args_predict = STRATEGIES[strategy]
    include_last_label = args_predict['include_last_label']
    n_samples = len(X_toy)
    mapie0 = MapieClassifier(**args_init)
    mapie1 = MapieClassifier(**args_init)
    mapie2 = MapieClassifier(**args_init)
    mapie0.fit(X_toy, y_toy, sample_weight=None)
    mapie1.fit(X_toy, y_toy, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X_toy, y_toy, sample_weight=np.ones(shape=n_samples) * 5)
    y_pred0, y_ps0 = mapie0.predict(
        X_toy,
        include_last_label=include_last_label,
        alpha=0.2
    )
    y_pred1, y_ps1 = mapie1.predict(
        X_toy,
        include_last_label=include_last_label,
        alpha=0.2
    )
    y_pred2, y_ps2 = mapie2.predict(
        X_toy,
        include_last_label=include_last_label,
        alpha=0.2
    )
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred0, y_pred2)
    np.testing.assert_allclose(y_ps0, y_ps1)
    np.testing.assert_allclose(y_ps0, y_ps2)


@pytest.mark.parametrize(
    "alpha", [[0.2, 0.8], (0.2, 0.8), np.array([0.2, 0.8]), None]
)
def test_valid_prediction(alpha: Any) -> None:
    """Test fit and predict."""
    model = LogisticRegression(multi_class="multinomial")
    model.fit(X_toy, y_toy)
    mapie = MapieClassifier(estimator=model, cv="prefit")
    mapie.fit(X_toy, y_toy)
    mapie.predict(X_toy, alpha=alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_toy_dataset_predictions(strategy: str) -> None:
    """Test prediction sets estimated by MapieClassifier on a toy dataset"""
    args_init, args_predict = STRATEGIES[strategy]
    include_last_label = args_predict['include_last_label']
    clf = GaussianNB().fit(X_toy, y_toy)
    mapie = MapieClassifier(estimator=clf, **args_init)
    mapie.fit(X_toy, y_toy)
    _, y_ps = mapie.predict(
        X_toy,
        include_last_label=include_last_label,
        alpha=0.2
    )
    np.testing.assert_allclose(
        classification_coverage_score(y_toy, y_ps[:, :, 0]),
        COVERAGES[strategy],
    )
    np.testing.assert_allclose(y_ps[:, :, 0], y_toy_mapie[strategy])


def test_cumulated_scores() -> None:
    """Test cumulated score method on a tiny dataset."""
    alpha = [0.65]
    quantile = [0.750183952461055]
    # fit
    cumclf = CumulatedScoreClassifier()
    cumclf.fit(cumclf.X_calib, cumclf.y_calib)
    mapie = MapieClassifier(
        cumclf,
        method="cumulated_score",
        cv="prefit",
        random_state=42
    )
    mapie.fit(cumclf.X_calib, cumclf.y_calib)
    np.testing.assert_allclose(mapie.conformity_scores_, cumclf.y_calib_scores)
    # predict
    _, y_ps = mapie.predict(
        cumclf.X_test,
        include_last_label=True,
        alpha=alpha
    )
    np.testing.assert_allclose(mapie.quantiles_, quantile)
    np.testing.assert_allclose(y_ps[:, :, 0], cumclf.y_pred_sets)


@pytest.mark.parametrize("y_pred_proba", Y_PRED_PROBA)
def test_sum_proba_to_one_fit(y_pred_proba: ArrayLike) -> None:
    """
    Test if when the output probabilities of the model do not
    sum to one, return an error in the fit method.
    """
    wrong_model = WrongOutputModel(y_pred_proba)
    wrong_model.fit(X_toy, y_toy)
    mapie = MapieClassifier(wrong_model)
    with pytest.raises(
        AssertionError, match=r".*The sum of the.*"
    ):
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("y_pred_proba", Y_PRED_PROBA)
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
def test_sum_proba_to_one_predict(
    y_pred_proba: ArrayLike,
    alpha: Union[float, Iterable[float]]
) -> None:
    """
    Test if when the output probabilities of the model do not
    sum to one, return an error in the predict method.
    """
    wrong_model = WrongOutputModel(y_pred_proba)
    wrong_model.fit(X_toy, y_toy)
    mapie = MapieClassifier(method='score')
    mapie.fit(X_toy, y_toy)
    mapie.single_estimator_ = wrong_model
    with pytest.raises(
        AssertionError, match=r".*The sum of the.*"
    ):
        mapie.predict(X_toy, alpha=alpha)
