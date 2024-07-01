from abc import ABCMeta, abstractmethod

import numpy as np

from mapie._compatibility import np_nanquantile
from mapie._machine_precision import EPSILON
from mapie._typing import NDArray


class BaseConformityScore(metaclass=ABCMeta):
    """
    Base class for conformity scores.

    This class should not be used directly. Use derived classes instead.

    Parameters
    ----------
    consistency_check: bool, optional
        Whether to check the consistency between the methods
        ``get_estimation_distribution`` and ``get_conformity_scores``.
        If ``True``, the following equality must be verified:
        ``self.get_estimation_distribution(
            y_pred, self.get_conformity_scores(y, y_pred, **kwargs), **kwargs
        ) == y``

        By default ``True``.

    eps: float, optional
        Threshold to consider when checking the consistency between
        ``get_estimation_distribution`` and ``get_conformity_scores``.
        It should be specified if ``consistency_check==True``.

        By default, it is defined by the default precision.
    """

    def __init__(
        self,
        consistency_check: bool = True,
        eps: float = float(EPSILON),
    ):
        self.consistency_check = consistency_check
        self.eps = eps

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
        pass

    def check_consistency(
        self,
        y: NDArray,
        y_pred: NDArray,
        conformity_scores: NDArray,
        **kwargs
    ) -> None:
        """
        Check consistency between the following methods:
        ``get_estimation_distribution`` and ``get_signed_conformity_scores``

        The following equality should be verified:
        ``self.get_estimation_distribution(
            y_pred, self.get_conformity_scores(y, y_pred, **kwargs), **kwargs
        ) == y``

        Parameters
        ----------
        y: NDArray of shape (n_samples, ...)
            Observed target values.

        y_pred: NDArray of shape (n_samples, ...)
            Predicted target values.

        conformity_scores: NDArray of shape (n_samples, ...)
            Conformity scores.

        Raises
        ------
        ValueError
            If the two methods are not consistent.
        """
        score_distribution = self.get_estimation_distribution(
            y_pred, conformity_scores, **kwargs
        )
        abs_conformity_scores = np.abs(np.subtract(score_distribution, y))
        max_conf_score = np.max(abs_conformity_scores)
        if max_conf_score > self.eps:
            raise ValueError(
                "The two functions get_conformity_scores and "
                "get_estimation_distribution of the BaseConformityScore class "
                "are not consistent. "
                "The following equation must be verified: "
                "self.get_estimation_distribution(y_pred, "
                "self.get_conformity_scores(y, y_pred)) == y. "
                f"The maximum conformity score is {max_conf_score}. "
                "The eps attribute may need to be increased if you are "
                "sure that the two methods are consistent."
            )

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
        y: NDArray of shape (n_samples, ...)
            Observed target values.

        y_pred: NDArray of shape (n_samples, ...)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples, ...)
            Conformity scores.
        """

    @abstractmethod
    def get_estimation_distribution(
        self,
        y_pred: NDArray,
        conformity_scores: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Placeholder for ``get_estimation_distribution``.
        Subclasses should implement this method!

        Compute samples of the estimation distribution given the predicted
        targets and the conformity scores.

        Parameters
        ----------
        y_pred: NDArray of shape (n_samples, ...)
            Predicted target values.

        conformity_scores: NDArray of shape (n_samples, ...)
            Conformity scores.

        Returns
        -------
        NDArray of shape (n_samples, ...)
            Observed values.
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
        conformity_scores: NDArray of shape (n_samples, ...)
            Values from which the quantile is computed.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        axis: int
            The axis from which to compute the quantile.

            By default ``0``.

        reversed: bool
            Boolean specifying whether we take the upper or lower quantile,
            if False, the alpha quantile, otherwise the (1-alpha) quantile.

            By default ``False``.

        unbounded: bool
            Boolean specifying whether infinite prediction intervals
            could be produced (when alpha_np is greater than or equal to 1.).

            By default ``False``.

        Returns
        -------
        NDArray of shape (1, n_alpha) or (n_samples, n_alpha)
            The quantiles of the conformity scores.
        """
        n_ref = conformity_scores.shape[1-axis]
        n_calib = np.min(np.sum(~np.isnan(conformity_scores), axis=axis))
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
            np_nanquantile(
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
        the target confidence interval.

        Parameters:
        -----------
        X: NDArray of shape (n_samples, ...)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence interval to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns:
        --------
        The output strcture depend on the subclass.
            The prediction sets for each sample and each alpha level.
        """
