import numpy as np
from numpy.typing import NDArray

from mapie.conformity_scores import BaseRegressionScore


class QuantileRegressionScore(BaseRegressionScore):
    """
    Quantile conformity score for quantile regression.

    Notes
    -----
    `consistency_check` defaults to `False`: the check inherited from
    `BaseRegressionScore` verifies that `y` can be recovered from a single `y_pred` and
    its conformity score, whereas a quantile score is defined against *two* predictions,
    the lower and the upper quantile. The relation it asserts therefore cannot hold here.
    """

    def __init__(self, sym: bool = False, consistency_check: bool = False) -> None:
        super().__init__(sym=sym, consistency_check=consistency_check)

    def get_signed_conformity_scores(
        self, y: NDArray[np.float64], y_pred: NDArray[np.float64], **kwargs
    ) -> NDArray[np.float64]:
        """
        Compute the sample conformity scores given the predicted and
        observed targets.

        Both rows follow the `y - y_pred` orientation shared by the other regression
        scores, so that `get_estimation_distribution` reconstructs `y` from either of
        them: the first row is scored against the lower quantile, the second against the
        upper one.

        Parameters
        ----------
        y: NDArray[float] of shape (2, n_samples)
            Observed target values.

        y_pred: NDArray[float] of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray[float] of shape (n_samples, 2)
            Signed conformity scores.
        """
        return np.vstack((y - y_pred[0], y - y_pred[1]))

    def get_estimation_distribution(
        self,
        y_pred: NDArray[np.float64],
        conformity_scores: NDArray[np.float64],
        **kwargs,
    ) -> NDArray[np.float64]:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = y - y_pred
        <=> y = y_pred + signed conformity score

        `conformity_scores` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        y_pred = np.asarray(y_pred)
        conformity_scores = np.asarray(conformity_scores)

        if y_pred.ndim == 1 and conformity_scores.ndim == 1:
            if y_pred.shape[0] != conformity_scores.shape[0]:
                return np.add(y_pred[:, np.newaxis], conformity_scores)

        return np.add(y_pred, conformity_scores)


class AbsoluteQuantileRegressionScore(QuantileRegressionScore):
    """
    Absolute conformity score for quantile regression.
    """

    def __init__(
        self,
        sym: bool = True,
    ) -> None:
        super().__init__(sym=sym, consistency_check=False)

    def get_conformity_scores(
        self, y: NDArray[np.float64], y_pred: NDArray[np.float64], **kwargs
    ) -> NDArray[np.float64]:
        """
        Compute the conformity scores from the predicted values
        and the observed ones, from the following formula:
        conformity score = max(y_pred_lower - y, y - y_pred_upper)

        Both signed rows follow the `y - y_pred` orientation, so the lower one is
        negated to turn it into a distance above the lower quantile before taking the
        maximum.

        The consistency check inherited from `BaseRegressionScore` is not run here:
        as documented on `QuantileRegressionScore`, the relation it asserts cannot
        hold for a two-sided score, hence `consistency_check` being forced off.
        """
        conformity_scores = self.get_signed_conformity_scores(y, y_pred, **kwargs)
        return np.asarray(np.maximum(-conformity_scores[0], conformity_scores[1]))
