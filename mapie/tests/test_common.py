import inspect
from inspect import signature
from typing import Any, List, Tuple, Optional, Union, Callable, Dict

import numpy as np
import pytest
from numpy._typing import NDArray, ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from mapie.classification import _MapieClassifier
from mapie.regression.regression import _MapieRegressor
from mapie.regression.quantile_regression import _MapieQuantileRegressor

def train_test_split_shuffle(
    X: NDArray,
    y: NDArray,
    test_size: Optional[float] = None,
    random_state: int = 42,
    sample_weight: Optional[NDArray] = None,
) -> Union[Tuple[Any, Any, Any, Any], Tuple[Any, Any, Any, Any, Any, Any]]:
    splitter = ShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if sample_weight is not None:
        sample_weight_train = sample_weight[train_idx]
        sample_weight_test = sample_weight[test_idx]
        return X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test

    return X_train, X_test, y_train, y_test


def filter_params(
    function: Callable,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if params is None:
        return {}

    model_params = inspect.signature(function).parameters
    return {k: v for k, v in params.items() if k in model_params}


class DummyClassifierWithFitAndPredictParams(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self._dummy_fit_param = None

    def fit(self, X: ArrayLike, y: ArrayLike, dummy_fit_param: bool = False) -> Self:
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("Dummy classifier needs at least 3 classes")
        self._dummy_fit_param = dummy_fit_param
        return self

    def predict_proba(self, X: ArrayLike, dummy_predict_param: bool = False) -> NDArray:
        probas = np.zeros((len(X), len(self.classes_)))
        if self._dummy_fit_param & dummy_predict_param:
            probas[:, 0] = 0.1
            probas[:, 1] = 0.9
        elif self._dummy_fit_param:
            probas[:, 1] = 0.1
            probas[:, 2] = 0.9
        elif dummy_predict_param:
            probas[:, 1] = 0.1
            probas[:, 0] = 0.9
        else:
            probas[:, 2] = 0.1
            probas[:, 0] = 0.9
        return probas

    def predict(self, X: ArrayLike, dummy_predict_param: bool = False) -> NDArray:
        y_preds_proba = self.predict_proba(X, dummy_predict_param)
        return np.amax(y_preds_proba, axis=0)


X_toy = np.arange(18).reshape(-1, 1)
y_toy = np.array(
    [0, 0, 1, 0, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 2, 1, 2, 2]
)


def MapieSimpleEstimators() -> List[BaseEstimator]:
    return [_MapieRegressor, _MapieClassifier]


def MapieEstimators() -> List[BaseEstimator]:
    return [_MapieRegressor, _MapieClassifier, _MapieQuantileRegressor]


def MapieDefaultEstimators() -> List[BaseEstimator]:
    return [
        (_MapieRegressor, LinearRegression),
        (_MapieClassifier, LogisticRegression),
    ]


def MapieTestEstimators() -> List[BaseEstimator]:
    return [
        (_MapieRegressor, LinearRegression()),
        (_MapieRegressor, make_pipeline(LinearRegression())),
        (_MapieClassifier, LogisticRegression()),
        (_MapieClassifier, make_pipeline(LogisticRegression())),
    ]


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_initialized(MapieEstimator: BaseEstimator) -> None:
    """Test that initialization does not crash."""
    MapieEstimator()


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_default_parameters(MapieEstimator: BaseEstimator) -> None:
    """Test default values of input parameters."""
    mapie_estimator = MapieEstimator()
    assert mapie_estimator.estimator is None
    assert mapie_estimator.cv is None
    assert mapie_estimator.verbose == 0
    assert mapie_estimator.n_jobs is None


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_fit(MapieEstimator: BaseEstimator) -> None:
    """Test that fit raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that fit-predict raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)
    mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_no_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that predict before fit raises errors."""
    mapie_estimator = MapieEstimator()
    with pytest.raises(NotFittedError):
        mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_default_sample_weight(MapieEstimator: BaseEstimator) -> None:
    """Test default sample weights."""
    mapie_estimator = MapieEstimator()
    assert (
        signature(mapie_estimator.fit).parameters["sample_weight"].default
        is None
    )


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_default_alpha(MapieEstimator: BaseEstimator) -> None:
    """Test default alpha."""
    mapie_estimator = MapieEstimator()
    assert (
        signature(mapie_estimator.predict).parameters["alpha"].default is None
    )


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_estimator(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """Test that None estimator defaults to expected value."""
    MapieEstimator, DefaultEstimator = pack
    mapie_estimator = MapieEstimator(estimator=None)
    mapie_estimator.fit(X_toy, y_toy)
    if isinstance(mapie_estimator, _MapieClassifier):
        assert isinstance(
            mapie_estimator.estimator_.single_estimator_, DefaultEstimator
        )
    if isinstance(mapie_estimator, _MapieRegressor):
        assert isinstance(
            mapie_estimator.estimator_.single_estimator_, DefaultEstimator
        )


@pytest.mark.parametrize("estimator", [0, "a", KFold(), ["a", "b"]])
@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_invalid_estimator(
    MapieEstimator: BaseEstimator, estimator: Any
) -> None:
    """Test that invalid estimators raise errors."""
    mapie_estimator = MapieEstimator(estimator=estimator)
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieTestEstimators())
def test_invalid_prefit_estimator(
    pack: Tuple[BaseEstimator, BaseEstimator]
) -> None:
    """Test that non-fitted estimator with prefit cv raise errors."""
    MapieEstimator, estimator = pack
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    with pytest.raises(NotFittedError):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieTestEstimators())
def test_valid_prefit_estimator(
    pack: Tuple[BaseEstimator, BaseEstimator]
) -> None:
    """Test that fitted estimators with prefit cv raise no errors."""
    MapieEstimator, estimator = pack
    estimator.fit(X_toy, y_toy)
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    mapie_estimator.fit(X_toy, y_toy)
    check_is_fitted(mapie_estimator, mapie_estimator.fit_attributes)
    assert mapie_estimator.n_features_in_ == 1


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
@pytest.mark.parametrize(
    "cv", [-3.14, -2, 0, 1, "cv", LinearRegression(), [1, 2]]
)
def test_invalid_cv(MapieEstimator: BaseEstimator, cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie_estimator = MapieEstimator(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_alpha_results(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """
    Test that alpha set to ``None`` in MapieEstimator gives same predictions
    as base estimator.
    """
    MapieEstimator, DefaultEstimator = pack
    estimator = DefaultEstimator()
    estimator.fit(X_toy, y_toy)
    y_pred_expected = estimator.predict(X_toy)
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    mapie_estimator.fit(X_toy, y_toy)
    y_pred = mapie_estimator.predict(X_toy)
    np.testing.assert_allclose(y_pred_expected, y_pred)
