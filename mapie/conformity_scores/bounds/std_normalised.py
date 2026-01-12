import warnings
from typing import Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_random_state, indexable

from mapie.conformity_scores import BaseRegressionScore
from mapie.utils import check_sklearn_user_model_is_fitted



class StdConformityScore(BaseRegressionScore):
    """
    Standardized non-conformity score

    The conformity score = |y - y_pred|/ y_std.

    This is appropriate when your model is a Gaussian Process or is able to
    return an estimate of the standard deviation of the prediction. This
    non-conformity score is able to give adaptive prediction intervals
    (taking X into account).
    """

    def __init__(
        self,
        power=1,
        sym=True
    ) -> None:
        self.pow = power
        super().__init__(sym=sym, consistency_check=False)

    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
        y_std: Union[ArrayLike, None],
        **kwargs
    ) -> NDArray:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the following formula:
        signed conformity score = y - y_pred
        """
        y_std = np.maximum(self.eps, y_std) ** self.pow
        return np.subtract(y, y_pred) / y_std

    def get_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
        y_std: Union[ArrayLike, None],
        **kwargs
    ) -> NDArray:
        """
        Get the conformity score considering the symmetrical property if so.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        y_std: Union[NDArray, None]
            Estimated standard deviation of the predictions.
            If None, the conformity scores are not standardised.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """
        conformity_scores = self.get_signed_conformity_scores(
            X, y, y_pred, y_std
        )
        if self.consistency_check:
            self.check_consistency(X, y, y_pred, conformity_scores)
        if self.sym:
            conformity_scores = np.abs(conformity_scores)
        return conformity_scores

    def get_estimation_distribution(
        self,
        X: ArrayLike,
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

    def get_bounds(
        self,
        X: ArrayLike,
        estimator: Optional[RegressorMixin],
        conformity_scores: NDArray,
        alpha_np: NDArray,
        ensemble: bool,
        method: str,
        **kwargs
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
        y_pred, y_pred_low, y_pred_up, y_std_multi = estimator.predict(
            X, ensemble
        )
        signed = -1 if self.sym else 1
        conformity_scores = conformity_scores * np.maximum(
            self.eps, y_std_multi ** self.pow
        )
        if method == "plus":
            alpha_low = alpha_np if self.sym else alpha_np / 2
            alpha_up = 1 - alpha_np if self.sym else 1 - alpha_np / 2

            conformity_scores_low = self.get_estimation_distribution(
                X, y_pred_low, signed * conformity_scores
            )
            conformity_scores_up = self.get_estimation_distribution(
                X, y_pred_up, conformity_scores
            )
            bound_low = self.get_quantile(
                conformity_scores_low, alpha_low, axis=1, reversed=True
            )
            bound_up = self.get_quantile(
                conformity_scores_up, alpha_up, axis=1, reversed=False
            )
        else:
            quantile_search = False if self.sym else True
            alpha_low = 1 - alpha_np if self.sym else alpha_np / 2
            alpha_up = 1 - alpha_np if self.sym else 1 - alpha_np / 2

            quantile_low = self.get_quantile(
                conformity_scores, alpha_low, axis=1, method=quantile_search
            )
            quantile_up = self.get_quantile(
                conformity_scores, alpha_up, axis=1, method=False
            )
            bound_low = self.get_estimation_distribution(
                X, y_pred_low, signed * quantile_low
            )
            bound_up = self.get_estimation_distribution(
                X, y_pred_up, quantile_up
            )

        return y_pred, bound_low, bound_up