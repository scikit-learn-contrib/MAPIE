from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

from mapie._typing import ArrayLike, NDArray


class EnsembleEstimator(metaclass=ABCMeta):
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
        **kwargs
    ) -> EnsembleEstimator:
        """
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold conformity scores are stored under
        the ``conformity_scores_`` attribute.
        """

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
        **kwargs
    ) -> Union[NDArray, Tuple[NDArray, NDArray, NDArray]]:
        """
        Predict target from X. It also computes the prediction per train sample
        for each test sample according to ``self.method``.
        """
