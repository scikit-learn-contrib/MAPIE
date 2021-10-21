
from inspect import signature
from typing import Any, List

import pytest
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from mapie.regression import MapieRegressor
from mapie.classification import MapieClassifier


X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.array([0, 0, 1, 0, 1, 2, 1, 2, 2])


def MapieEstimators() -> List[BaseEstimator]:
    return [MapieRegressor, MapieClassifier]


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_initialized(MapieEstimator: BaseEstimator) -> None:
    """Test that initialization does not crash."""
    MapieEstimator()


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_fit(MapieEstimator: BaseEstimator) -> None:
    """Test that fit raises no errors."""
    estimator = MapieEstimator()
    estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that fit-predict raises no errors."""
    estimator = MapieEstimator()
    estimator.fit(X_toy, y_toy)
    estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_no_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that predict before fit raises errors."""
    estimator = MapieEstimator()
    with pytest.raises(NotFittedError):
        estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_default_sample_weight(MapieEstimator: BaseEstimator) -> None:
    """Test default sample weights."""
    estimator = MapieEstimator()
    assert signature(estimator.fit).parameters["sample_weight"].default is None


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_default_alpha(MapieEstimator: BaseEstimator) -> None:
    """Test default alpha."""
    estimator = MapieEstimator()
    assert signature(estimator.predict).parameters["alpha"].default is None


@pytest.mark.parametrize("estimator", [0, "a", KFold(), ["a", "b"]])
@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_invalid_estimator(
    MapieEstimator: BaseEstimator,
    estimator: Any
) -> None:
    """Test that invalid estimators raise errors."""
    mapie = MapieEstimator(estimator=estimator)
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie.fit(X_toy, y_toy)


@parametrize_with_checks(
    [
        MapieRegressor(),  # MapieEstimator() for MapieEstimator in MapieEstimators()
    ]
)  # type: ignore
def test_sklearn_compatible_estimator(
    estimator: BaseEstimator,
    check: Any
) -> None:
    """Check compatibility with sklearn, using sklearn estimator checks API."""
    check(estimator)
