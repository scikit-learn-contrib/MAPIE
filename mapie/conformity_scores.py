import sys
from abc import ABCMeta, abstractmethod

import numpy as np

from ._typing import ArrayLike


class ConformityScore(metaclass=ABCMeta):
    """Base class for conformity scores.

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
            Whether to consider the conformity score as symmetrical or not.
        consistency_check : bool, optional
            Whether to check the consistency between the following methods:
            - get_observed_value and
            - get_signed_conformity_scores
            by default True.
        eps : float, optional
            Threshold to consider when checking the consistency between the
            following methods:
            - get_observed_value and
            - get_signed_conformity_scores
            The following equality must be verified:
            self.get_observed_value(
                y_pred, self.get_conformity_scores(y, y_pred)
            ) == y
            It should be specified if consistency_check==True.
            by default sys.float_info.epsilon.
        """
        self.sym = sym
        self.eps = eps
        self.consistency_check = consistency_check

    @abstractmethod
    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> ArrayLike:
        """Placeholder for get_signed_conformity_scores.
        Subclasses should implement this method!

        Compute the signed conformity scores from the predicted values
        and the observed ones.

        Parameters
        ----------
        y : ArrayLike
            Observed values.
        y_pred : ArrayLike
            Predicted values.

        Returns
        -------
        ArrayLike
            Unsigned conformity scores.
        """

    def get_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> ArrayLike:
        """Get the conformity score considering the symmetrical property if so.

        Parameters
        ----------
        y : ArrayLike
            Observed values.
        y_pred : ArrayLike
            Predicted values.

        Returns
        -------
        ArrayLike
            Conformity scores.
        """
        conformity_scores = self.get_signed_conformity_scores(y, y_pred)
        if self.sym:
            conformity_scores = np.abs(conformity_scores)
        return conformity_scores

    @abstractmethod
    def get_observed_value(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> ArrayLike:
        """Placeholder for get_observed_value.
        Subclasses should implement this method!

        Compute the observed values from the predicted values and the
        conformity scores.

        Parameters
        ----------
        y_pred : ArrayLike
            Predicted values.
        conformity_scores : ArrayLike
            Conformity scores.

        Returns
        -------
        ArrayLike
            Observed values.
        """

    def check_consistency(self, y: ArrayLike, y_pred: ArrayLike) -> None:
        """Check consistency between the following methods:
        get_observed_value and get_signed_conformity_scores

        The following equality should be verified:
        self.get_observed_value(
            y_pred, self.get_conformity_scores(y, y_pred)
        ) == y

        Parameters
        ----------
        y : ArrayLike
            Observed values.
        y_pred : ArrayLike
            Predicted values.

        Raises
        ------
        ValueError
            If the two methods are not consistent.
        """
        if self.consistency_check:
            conformity_scores = self.get_signed_conformity_scores(y, y_pred)
            abs_conformity_scores = np.abs(
                self.get_observed_value(y_pred, conformity_scores) - y
            )
            max_conf_score = np.max(abs_conformity_scores)
            if (abs_conformity_scores > self.eps).any():
                raise ValueError(
                    "The two functions get_conformity_scores and "
                    "get_observed_value of the ConformityScore class "
                    "are not consistent. "
                    "The following equation must be verified: "
                    "self.get_observed_value(y_pred, self.get_conformity_scores(y, y_pred)) == y. "  # noqa: E501
                    f"The maximum conformity score is {max_conf_score}."
                    "The eps attribute may need to be increased if you are "
                    "sure that the two methods are consistent."
                )


class AbsoluteConformityScore(ConformityScore):
    """Absolute conformity score.

    The signed conformity score = y - y_pred.
    The conformity score is symmetrical.

    This is appropriate when the confidence interval is symmetrical and
    its range is approximatively the same over the range of predicted values.
    """

    def __init__(self) -> None:
        ConformityScore.__init__(self, True, consistency_check=False)

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> ArrayLike:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the following formula:
        signed conformity score = y - y_pred
        """
        return y - y_pred

    def get_observed_value(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> ArrayLike:
        """
        Compute the observed values from the predicted values and the
        conformity scores, from the following formula:
        signed conformity score = y - y_pred
        <=> y = y_pred + signed conformity score
        """
        return y_pred + conformity_scores


class GammaConformityScore(ConformityScore):
    """Gamma conformity score.

    The signed conformity score = (y - y_pred) / y_pred.
    The conformity score is not symmetrical.

    This is appropriate when the confidence interval is not symmetrical and
    its range depends on the predicted values.
    """

    def __init__(self) -> None:
        ConformityScore.__init__(self, False, consistency_check=False)

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> ArrayLike:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        """
        return (y - y_pred) / y_pred

    def get_observed_value(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> ArrayLike:
        """
        Compute the observed values from the predicted values and the
        conformity scores, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        <=> y = y_pred * (1 + signed conformity score)
        """
        return y_pred * (1 + conformity_scores)
