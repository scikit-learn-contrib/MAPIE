from typing import Callable, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing_extensions import Self

from mapie._typing import NDArray, ArrayLike
import inspect
from sklearn.model_selection import ShuffleSplit


def train_test_split_shuffle(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:

    splitter = ShuffleSplit(n_splits=1,
                            test_size=test_size,
                            random_state=random_state)
    train_idx, test_idx = next(splitter.split(X))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

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
        self._classes = None
        self._fit_param = None

    def fit(self, X: ArrayLike, y: ArrayLike, fit_param: bool) -> Self:
        self._classes = np.unique(y)
        if len(self._classes) < 3:
            raise ValueError("Dummy classifier needs at least 3 classes")
        self._fit_param = fit_param
        return self

    def predict(self, X: ArrayLike, predict_param: bool):
        if self._fit_param & predict_param:
            return np.array(self._classes[0]) * len(X)
        if self._fit_param:
            return np.array(self._classes[1]) * len(X)
        if predict_param:
            return np.array(self._classes[2]) * len(X)
        return np.random.choice(self._classes, len(X))
