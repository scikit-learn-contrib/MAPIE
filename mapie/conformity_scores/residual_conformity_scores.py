import numpy as np

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray

from mapie.conformity_scores import ConformityScore


class AbsoluteConformityScore(ConformityScore):
    """
    Absolute conformity score.

    The signed conformity score = y - y_pred.
    The conformity score is symmetrical.

    This is appropriate when the confidence interval is symmetrical and
    its range is approximatively the same over the range of predicted values.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(sym=True, consistency_check=True)

    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the observed values
        and the estimator, from the following formula:
        signed conformity score = y - y_pred
        """
        return np.subtract(y, y_pred)

    def get_estimation_distribution(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        values: ArrayLike
    ):
        """
        Compute samples of the estimation distribution from the predicted
        targets and ``values``, from the following formula:
        signed conformity score = y - y_pred
        <=> y = y_pred + signed conformity score

        ``values`` can be either the conformity scores or
        the conformity scores aggregated with the predictions.
        """
        return np.add(y_pred, values)


class GammaConformityScore(ConformityScore):
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
    ) -> None:
        super().__init__(sym=False, consistency_check=False, eps=EPSILON)

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
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the observed values
        and the estimator, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        """
        self._check_observed_data(y)
        self._check_predicted_data(y_pred)
        return np.divide(np.subtract(y, y_pred), y_pred)

    def get_estimation_distribution(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        values: ArrayLike,
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        targets and ``values``, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        <=> y = y_pred * (1 + signed conformity score)

        ``values`` can be either the conformity scores or
        the conformity scores aggregated with the predictions.
        """
        self._check_predicted_data(y_pred)
        return np.multiply(y_pred, np.add(1, values))
