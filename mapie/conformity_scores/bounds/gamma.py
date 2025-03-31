import numpy as np

from numpy.typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseRegressionScore


class GammaConformityScore(BaseRegressionScore):
    """
    Gamma conformity score.

    The signed conformity score = (y - y_pred) / y_pred.
    The conformity score is not symmetrical.

    This is appropriate when the confidence interval is not symmetrical and
    its range depends on the predicted values. Like the Gamma distribution,
    its support is limited to strictly positive reals.
    """

    def __init__(
        self,
        sym: bool = False,
    ) -> None:
        super().__init__(sym=sym, consistency_check=False)

    def _check_observed_data(
        self,
        y: ArrayLike,
    ) -> None:
        if not self._all_strictly_positive(y):
            raise ValueError(
                f"At least one of the observed target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be strictly positive, "
                "in conformity with the Gamma distribution support."
            )

    def _check_predicted_data(
        self,
        y_pred: ArrayLike,
    ) -> None:
        if not self._all_strictly_positive(y_pred):
            raise ValueError(
                f"At least one of the predicted target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be strictly positive, "
                "in conformity with the Gamma distribution support."
            )

    @staticmethod
    def _all_strictly_positive(
        y: ArrayLike,
    ) -> bool:
        return not np.any(np.less_equal(y, 0))

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
        **kwargs
    ) -> NDArray:
        """
        Compute the signed conformity scores from the observed values
        and the predicted ones, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        """
        self._check_observed_data(y)
        self._check_predicted_data(y_pred)
        return np.divide(np.subtract(y, y_pred), y_pred)

    def get_estimation_distribution(
        self,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike,
        **kwargs
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        <=> y = y_pred * (1 + signed conformity score)

        ``conformity_scores`` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        self._check_predicted_data(y_pred)
        return np.multiply(y_pred, np.add(1, conformity_scores))
