import numpy as np

from numpy.typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseRegressionScore


class AbsoluteConformityScore(BaseRegressionScore):
    """
    Absolute conformity score.

    The signed conformity score = y - y_pred.
    The conformity score is symmetrical.

    This is appropriate when the confidence interval is symmetrical and
    its range is approximatively the same over the range of predicted values.
    """

    def __init__(
        self,
        sym: bool = True,
    ) -> None:
        super().__init__(sym=sym, consistency_check=True)

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
        **kwargs
    ) -> NDArray:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the following formula:
        signed conformity score = y - y_pred
        """
        return np.subtract(y, y_pred)

    def get_estimation_distribution(
        self,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike,
        **kwargs
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = y - y_pred
        <=> y = y_pred + signed conformity score

        ``conformity_scores`` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        return np.add(y_pred, conformity_scores)
