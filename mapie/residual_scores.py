import sys
from abc import ABCMeta, abstractmethod

import numpy as np


class ResidualScore(metaclass=ABCMeta):
    """Base class for residual scores.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
        self,
        sym: bool,
        consistency_check: bool = True,
        eps: float = sys.float_info.epsilon,
    ):
        """
        Parameters
        ----------
        sym : bool
            Whether to consider the residual score as symmetrical or not.
        consistency_check : bool, optional
            Whether to check the consistency between the following methods:
            - get_observed_value and 
            - get_signed_residual_scores
            by default True.
        eps : float, optional
            Threshold to consider when checking the consistency between the 
            following methods:
            - get_observed_value and 
            - get_signed_residual_scores
            The following equality must be verified:
            self.get_observed_value(y_pred, self.get_residual_score(y, y_pred)) == y
            It should be specified if consistency_check==True.
            by default sys.float_info.epsilon. 
        """
        self.sym = sym
        self.eps = eps
        self.consistency_check = consistency_check

    @abstractmethod
    def get_signed_residual_scores(
        self, y: np.ndarray, y_pred: np.ndarray,
    ) -> np.ndarray:
        """Placeholder for get_signed_residual_scores.
        Subclasses should implement this method!

        Compute the unsigned residual scores from the predicted values and the
        observed ones.

        Parameters
        ----------
        y : np.ndarray
            Observed values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            Unsigned residual scores.
        """

    def get_residual_scores(self, y: np.ndarray, y_pred: np.ndarray,) -> np.ndarray:
        """Get the residual score considering the symmetrical property if so.

        Parameters
        ----------
        y : np.ndarray
            Observed values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        np.ndarray
            Residual scores.
        """
        residuals = self.get_signed_residual_scores(y, y_pred)
        if self.sym:
            residuals = np.abs(residuals)
        return residuals

    @abstractmethod
    def get_observed_value(
        self, y_pred: np.ndarray, residual_scores: np.ndarray
    ) -> np.ndarray:
        """Placeholder for get_observed_value.
        Subclasses should implement this method!

        Compute the observed values from the predicted values and the
        residual scores.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.
        residual_scores : np.ndarray
            Residual scores.

        Returns
        -------
        np.ndarray
            Observed values.
        """

    def check_consistency(self, y: np.ndarray, y_pred: np.ndarray):
        """Check consistency between the following methods:
        get_observed_value and get_signed_residual_scores

        The following equality should be verified:
        self.get_observed_value(y_pred, self.get_residual_score(y, y_pred)) == y

        Parameters
        ----------
        y : np.ndarray
            Observed values.
        y_pred : np.ndarray
            Predicted values.

        Raises
        ------
        ValueError
            If the two methods are not consistent.
        """
        if self.consistency_check:
            residual_scores = self.get_signed_residual_scores(y, y_pred)
            abs_residuals = np.abs(self.get_observed_value(y_pred, residual_scores) - y)
            max_res = np.max(abs_residuals)
            if (abs_residuals > self.eps).any():
                raise ValueError(
                    "The two functions get_residual_score and get_observed_value "
                    "of the ResidualScore class are not consistent. "
                    "The following equation must be verified: "
                    "self.get_observed_value(y_pred, self.get_residual_score(y, y_pred)) == y. "  # noqa: E501
                    f"The maximum residual is {max_res}."
                    "The eps attribute may need to be increased if you are sure "
                    "that the two methods are consistent."
                )


class AbsoluteResidualScore(ResidualScore):
    """Absolute residual.

    The unsigned residual score = y - y_pred.
    The residual score is symmetrical.

    This is appropriate when the confidence interval is symmetrical and its range
    is approximatively the same over the range of predicted values.
    """

    def __init__(self):
        ResidualScore.__init__(self, True, consistency_check=False)

    def get_signed_residual_scores(
        self, y: np.ndarray, y_pred: np.ndarray,
    ) -> np.ndarray:
        return y - y_pred

    def get_observed_value(
        self, y_pred: np.ndarray, residual_scores: np.ndarray
    ) -> np.ndarray:
        return y_pred + residual_scores


class GammaResidualScore(ResidualScore):
    """Gamma residual.

    The unsigned residual score = (y - y_pred) / y_pred.
    The residual score is not symmetrical.

    This is appropriate when the confidence interval is not symmetrical and its range
    depends on the predicted values.
    """

    def __init__(self):
        ResidualScore.__init__(self, False, consistency_check=False)

    def get_signed_residual_scores(
        self, y: np.ndarray, y_pred: np.ndarray,
    ) -> np.ndarray:
        return (y - y_pred) / y_pred

    def get_observed_value(
        self, y_pred: np.ndarray, residual_scores: np.ndarray
    ) -> np.ndarray:
        return y_pred * (1 + residual_scores)
