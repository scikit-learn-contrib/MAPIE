from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator

from numpy.typing import NDArray


class BaseConformityScore(metaclass=ABCMeta):
    """
    Base class for conformity scores.

    This class should not be used directly. Use derived classes instead.
    """

    def __init__(self) -> None:
        pass

    def set_external_attributes(
        self,
        **kwargs
    ) -> None:
        """
        Set attributes that are not provided by the user.

        Must be overloaded by subclasses if necessary to add more attributes,
        particularly when the attributes are known after the object has been
        instantiated.
        """

    def set_ref_predictor(
        self,
        predictor: BaseEstimator
    ):
        """
        Set the reference predictor.

        Parameters
        ----------
        predictor: BaseEstimator
            Reference predictor.
        """
        self.predictor = predictor

    def split_data(
        self,
        X: NDArray,
        y: NDArray,
        y_enc: NDArray,
        sample_weight: Optional[NDArray] = None,
        groups: Optional[NDArray] = None,
    ):
        """
        Split data. Keeps part of the data for the calibration estimator
        (separate from the calibration data).

        Parameters
        ----------
        *args: Tuple of NDArray

        Returns
        -------
        Tuple of NDArray
            Split data for training and calibration.
        """
        self.n_samples_ = len(X)
        return X, y, y_enc, sample_weight, groups

    @abstractmethod
    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Placeholder for ``get_conformity_scores``.
        Subclasses should implement this method!

        Compute the sample conformity scores given the predicted and
        observed targets.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """

    @staticmethod
    def get_quantile(
        conformity_scores: NDArray,
        alpha_np: NDArray,
        axis: int = 0,
        reversed: bool = False,
        unbounded: bool = False
    ) -> NDArray:
        """
        Compute the alpha quantile of the conformity scores.

        Parameters
        ----------
        conformity_scores: NDArray of shape (n_samples,)
            Values from which the quantile is computed.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence set.

        axis: int
            The axis from which to compute the quantile.

            By default ``0``.

        reversed: bool
            Boolean specifying whether we take the upper or lower quantile,
            if False, the alpha quantile, otherwise the (1-alpha) quantile.

            By default ``False``.

        unbounded: bool
            Boolean specifying whether infinite prediction sets
            could be produced (when alpha_np is greater than or equal to 1.).

            By default ``False``.

        Returns
        -------
        NDArray of shape (1, n_alpha) or (n_samples, n_alpha)
            The quantiles of the conformity scores.
        """
        n_ref = conformity_scores.shape[1-axis]
        n_calib: int = np.min(np.sum(~np.isnan(conformity_scores), axis=axis))
        signed = 1-2*reversed

        # Adapt alpha w.r.t upper/lower : alpha vs. 1-alpha
        alpha_ref = (1-2*alpha_np)*reversed + alpha_np

        # Adjust alpha w.r.t quantile correction
        alpha_cor = np.ceil(alpha_ref*(n_calib+1))/n_calib
        alpha_cor = np.clip(alpha_cor, a_min=0, a_max=1)

        # Compute the target quantiles:
        # If unbounded is True and alpha is greater than or equal to 1,
        # the quantile is set to infinity.
        # Otherwise, the quantile is calculated as the corrected lower quantile
        # of the signed conformity scores.
        quantile = signed * np.column_stack([
            np.nanquantile(
                signed * conformity_scores, _alpha_cor,
                axis=axis, method="lower"
            ) if not (unbounded and _alpha >= 1) else np.inf * np.ones(n_ref)
            for _alpha, _alpha_cor in zip(alpha_ref, alpha_cor)
        ])
        return quantile

    @abstractmethod
    def predict_set(
        self,
        X: NDArray,
        alpha_np: NDArray,
        **kwargs
    ):
        """
        Compute the prediction sets on new samples based on the uncertainty of
        the target confidence set.

        Parameters:
        -----------
        X: NDArray of shape (n_samples,)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence set to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns:
        --------
        The output structure depend on the subclass.
            The prediction sets for each sample and each alpha level.
        """
