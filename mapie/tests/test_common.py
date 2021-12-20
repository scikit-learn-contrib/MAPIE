from inspect import signature
from typing import Any, List, Tuple

import pytest
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from mapie.regression import MapieRegressor
from mapie.classification import MapieClassifier


X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.array([0, 0, 1, 0, 1, 2, 1, 2, 2])


def MapieEstimators() -> List[BaseEstimator]:
    return [MapieRegressor, MapieClassifier]


def MapieDefaultEstimators() -> List[BaseEstimator]:
    return [
        (MapieRegressor, LinearRegression),
        (MapieClassifier, LogisticRegression),
    ]


def MapieTestEstimators() -> List[BaseEstimator]:
    return [
        (MapieRegressor, LinearRegression()),
        (MapieRegressor, make_pipeline(LinearRegression())),
        (MapieClassifier, LogisticRegression()),
        (MapieClassifier, make_pipeline(LogisticRegression()))
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
    assert mapie_estimator.verbose == 0
    assert mapie_estimator.n_jobs is None


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_fit(MapieEstimator: BaseEstimator) -> None:
    """Test that fit raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that fit-predict raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)
    mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_no_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that predict before fit raises errors."""
    mapie_estimator = MapieEstimator()
    with pytest.raises(NotFittedError):
        mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_default_sample_weight(MapieEstimator: BaseEstimator) -> None:
    """Test default sample weights."""
    mapie_estimator = MapieEstimator()
    assert signature(
        mapie_estimator.fit
    ).parameters["sample_weight"].default is None


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_default_alpha(MapieEstimator: BaseEstimator) -> None:
    """Test default alpha."""
    mapie_estimator = MapieEstimator()
    assert signature(
        mapie_estimator.predict
    ).parameters["alpha"].default is None


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_estimator(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """Test that None estimator defaults to expected value."""
    MapieEstimator, expected_default_estimator = pack
    mapie_estimator = MapieEstimator(estimator=None)
    mapie_estimator.fit(X_toy, y_toy)
    assert isinstance(
        mapie_estimator.single_estimator_,
        expected_default_estimator
    )


@pytest.mark.parametrize("estimator", [0, "a", KFold(), ["a", "b"]])
@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_invalid_estimator(
    MapieEstimator: BaseEstimator,
    estimator: Any
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


@parametrize_with_checks([MapieRegressor()])  # type: ignore
def test_sklearn_compatible_estimator(
    estimator: BaseEstimator,
    check: Any
) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    check(estimator)
