from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from mapie._machine_precision import EPSILON

from mapie.conformity_scores.regression import BaseRegressionScore


class StdConformityScore(BaseRegressionScore):
    """
    Standardized non-conformity score

    The conformity score = |y - y_pred|/ y_std.

    This requires a model, such as a Gaussian Process, that can return an
    estimate of the standard deviation of the prediction through
    ``predict(X, return_std=True)``. This non-conformity score is able to give
    adaptive prediction intervals (taking X into account).
    """

    def __init__(
        self,
        power=1,
        sym=True,
        eps: float = float(EPSILON),
    ) -> None:
        super().__init__(sym=sym, consistency_check=False, eps=eps)
        self.pow = power

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
        y_std: Optional[ArrayLike] = None,
        **kwargs,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the following formula:
        signed conformity score = y - y_pred
        """
        if y_std is None:
            raise ValueError("y_std is required for StdConformityScore.")
        y_std = np.maximum(self.eps, y_std) ** self.pow
        return np.subtract(y, y_pred) / y_std

    def get_estimation_distribution(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike, **kwargs
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

    def get_bounds(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: Any,
        conformity_scores: NDArray,
        ensemble: bool = False,
        method: str = "base",
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        **predict_params,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Compute bounds of the prediction intervals from the observed values,
        the estimator of type ``EnsembleEstimator`` and the conformity scores.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Observed feature values.

        estimator: EnsembleEstimator
            Estimator that is fitted to predict y from X.

        conformity_scores: ArrayLike of shape (n_samples,)
            Conformity scores.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.

        method: str
            Method to choose for prediction interval estimates.
            The ``"plus"`` method implies that the quantile is calculated
            after estimating the bounds, whereas the other methods
            (among the ``"naive"``, ``"base"`` or ``"minmax"`` methods,
            for example) do the opposite.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            - The predictions itself. (y_pred) of shape (n_samples,).
            - The lower bounds of the prediction intervals of shape
            (n_samples, n_alpha).
            - The upper bounds of the prediction intervals of shape
            (n_samples, n_alpha).
        """
        y_pred, y_pred_low, y_pred_up, y_std_multi = estimator.predict_with_std(
            X, ensemble, **predict_params
        )
        signed = -1 if self.sym else 1
        conformity_scores = conformity_scores * np.maximum(
            self.eps, y_std_multi**self.pow
        )
        if method == "plus":
            alpha_low = alpha_np if self.sym else alpha_np / 2
            alpha_up = 1 - alpha_np if self.sym else 1 - alpha_np / 2

            conformity_scores_low = self.get_estimation_distribution(
                y_pred_low, signed * conformity_scores
            )
            conformity_scores_up = self.get_estimation_distribution(
                y_pred_up, conformity_scores
            )
            bound_low = self.get_quantile(
                conformity_scores_low,
                alpha_low,
                axis=1,
                reversed=True,
                unbounded=allow_infinite_bounds,
            )
            bound_up = self.get_quantile(
                conformity_scores_up,
                alpha_up,
                axis=1,
                reversed=False,
                unbounded=allow_infinite_bounds,
            )
        else:
            alpha_low = 1 - alpha_np if self.sym else alpha_np / 2
            alpha_up = 1 - alpha_np if self.sym else 1 - alpha_np / 2

            quantile_low = self.get_quantile(
                conformity_scores,
                alpha_low,
                axis=1,
                reversed=True,
                unbounded=allow_infinite_bounds,
            )
            quantile_up = self.get_quantile(
                conformity_scores,
                alpha_up,
                axis=1,
                unbounded=allow_infinite_bounds,
            )
            bound_low = self.get_estimation_distribution(
                y_pred_low, signed * quantile_low
            )
            bound_up = self.get_estimation_distribution(y_pred_up, quantile_up)

        return y_pred, bound_low, bound_up
