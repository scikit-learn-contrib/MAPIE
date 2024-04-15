##TEST###

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

from sklearn.base import BaseEstimator, ClassifierMixin

from mapie._typing import ArrayLike, NDArray


class EnsembleEstimator(ClassifierMixin, BaseEstimator):
    """
    This class implements methods to handle the training and usage of the
    estimator. This estimator can be unique or composed by cross validated
    estimators.
    """

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params
    ) -> EnsembleEstimator:

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        return_multi_pred: bool = True
    ) -> Union[NDArray, Tuple[NDArray, NDArray, NDArray]]:
