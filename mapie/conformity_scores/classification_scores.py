from mapie.conformity_scores import ConformityScore
from mapie._typing import ArrayLike, NDArray
import numpy as np


class LAC(ConformityScore):
    def __init__(self):
        super().__init__(True, False, None)

    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the ``LAC `` formula.

        Parameters
        ----------
        X : ArrayLike
            Observed values
        y : ArrayLike
            Target class
        y_pred : ArrayLike of shape (n_samples, n_classes)
            Predicted probas of X

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            conformity scors
        """
        y_pred_arr = np.array(y_pred)
        y_arr = np.array(y)
        return 1 - np.array([y_pred_arr[i, yy] for i, yy in enumerate(y_arr)])

    def get_estimation_distribution(
        self,
        X,
        y_pred,
        conformity_scores
    ):
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the ``LAC `` formula.

        Parameters
        ----------
        X : ArrayLike
            Observed values

        y_pred : ArrayLike of shape (n_samples, n_classes)
            Predicted probas of X

        conformity_scores : ArrayLike of shape (n_samples, )
            Correspond to the threshold, used to select the classes of the
            prediction sets

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            Prediction sets
        """
        y_ps = np.zeros_like(y_pred)

        for i in range(len(X)):
            for j, p in enumerate(y_pred[i, :]):
                if p >= 1 - conformity_scores[i]:
                    y_ps[i, j] = 1

        return y_ps

    def check_consistency(self, X, y, y_pred, conformity_scores) -> None:
        return
