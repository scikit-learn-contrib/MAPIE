from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, List

from mapie._typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator


class Calibrator(BaseEstimator, metaclass=ABCMeta):
    """
    Base abstract class for the calibrators

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
        X_calib: ArrayLike,
        y_pred_calib: Optional[ArrayLike],
        z_calib: Optional[ArrayLike],
        calib_conformity_scores: NDArray,
        alpha: float,
        sym: bool,
        sample_weight_calib: Optional[ArrayLike] = None,
        random_state: Optional[int] = None,
        **optim_kwargs,
    ) -> Calibrator:
        """
        Fit the calibrator instance

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Calibration data.

        y_pred: ArrayLike of shape (n_samples,)
            Calibration target.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

        conformity_scores: ArrayLike of shape (n_samples,)
            Calibration conformity scores

        alpha: float
            Between ``0.0`` and ``1.0``, represents the risk level of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

        sym: bool
            Weather or not, the prediction interval should be symetrical
            or not.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        random_state: Optional[int]
            Integer used to set the numpy seed, to get reproducible calibration
            results.
            If ``None``, the prediction intervals will be stochastics, and will
            change if you refit the calibration
            (even if no arguments have change).

            WARNING: If ``random_state``is not ``None``, ``np.random.seed``
            will be changed, which will reset the seed for all the other random
            number generators. It may have an impact on the rest of your code.

            By default ``None``.

        optim_kwargs: Dict
            Other argument, used in sklear.optimize.minimize
        """

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Predict ``(X, y_pred, z)``

        Parameters
        ----------
        X : ArrayLike
            Observed samples

        y_pred : NDArray
            Target prediction

        z : ArrayLike
            Exogenous variable

        Returns
        -------
        NDArray
            prediction
        """
