from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, List

from mapie._typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator


class Calibrator(BaseEstimator, metaclass=ABCMeta):
    """
    Base abstract class for thecalibrators
    
    Attributes
    ----------
    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.
    """

    fit_attributes: List[str]

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> Calibrator:
        """
        Fit the calibrator instance

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y_pred: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """

    @abstractmethod
    def predict(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Predict ``(X, y_pred, z)`` 

        Parameters
        ----------
        X : ArrayLike
            Observed samples

        y_pred : ArrayLike
            Target prediction

        z : ArrayLike
            Exogenous variable

        Returns
        -------
        NDArray
            prediction
        """

    def __call__(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        return self.predict(X, y_pred, z)
