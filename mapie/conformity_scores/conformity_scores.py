from abc import ABCMeta, abstractmethod

import numpy as np
from typing import Tuple
from sklearn.base import RegressorMixin

from mapie._compatibility import np_nanquantile
from mapie._typing import ArrayLike, NDArray


class ConformityScore(metaclass=ABCMeta):
    """
    Base class for conformity scores.

    Warning: This class should not be used directly.
    Use derived classes instead.

    Parameters
    ----------
    sym: bool
        Whether to consider the conformity score as symmetrical or not.

    consistency_check: bool, optional
        Whether to check the consistency between the following methods:
        - ``get_estimation_distribution`` and
        - ``get_signed_conformity_scores``

        By default ``True``.

    eps: float, optional
        Threshold to consider when checking the consistency between the
        following methods:
        - ``get_estimation_distribution`` and
        - ``get_signed_conformity_scores``
        The following equality must be verified:
        ``self.get_estimation_distribution(
            X,
            y_pred,
            self.get_conformity_scores(X, y, y_pred)
        ) == y``
        It should be specified if ``consistency_check==True``.

        By default ``np.float64(1e-8)``.
    """

    def __init__(
        self,
        sym: bool,
        consistency_check: bool = True,
        eps: np.float64 = np.float64(1e-8),
    ):
        self.sym = sym
        self.consistency_check = consistency_check
        self.eps = eps

    @abstractmethod
    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Placeholder for ``get_signed_conformity_scores``.
        Subclasses should implement this method!

        Compute the signed conformity scores from the predicted values
        and the observed ones.

        Parameters
        ----------
        X: ArrayLike
            Observed feature values.

        y: ArrayLike
            Observed target values.

        y_pred: ArrayLike
            Predicted target values.

        Returns
        -------
        NDArray
            Signed conformity scores.
        """

    @abstractmethod
    def get_estimation_distribution(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        values: ArrayLike
    ) -> NDArray:
        """
        Placeholder for ``get_estimation_distribution``.
        Subclasses should implement this method!

        Compute samples of the estimation distribution from the predicted
        targets and ``values``that can be either the conformity scores or
        the conformity scores aggregated with the predictions.

        Parameters
        ----------
        X: ArrayLike
            Observed feature values.

        y_pred: ArrayLike
            Predicted reference values of shape (n_samples, ...).
            The last dimension is the reference of the prediction.

        values: ArrayLike
            Either the conformity scores or the conformity scores aggregated
            with the predictions according to the subclass formula.

        Returns
        -------
        NDArray
            Observed values.
        """

    def check_consistency(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike,
    ) -> None:
        """
        Check consistency between the following methods:
        ``get_estimation_distribution`` and ``get_signed_conformity_scores``

        The following equality should be verified:
        ``self.get_estimation_distribution(
            X,
            y_pred,
            self.get_conformity_scores(X, y, y_pred)
        ) == y``

        Parameters
        ----------
        X: ArrayLike
            Observed feature values.

        y: ArrayLike
            Observed target values.

        y_pred: ArrayLike
            Predicted target values.

        conformity_scores: ArrayLike
            Conformity scores.

        Raises
        ------
        ValueError
            If the two methods are not consistent.
        """
        score_distribution = self.get_estimation_distribution(
            X, y_pred, conformity_scores
        )
        abs_conformity_scores = np.abs(np.subtract(score_distribution, y))
        max_conf_score = np.max(abs_conformity_scores)
        if max_conf_score > self.eps:
            raise ValueError(
                "The two functions get_conformity_scores and "
                "get_estimation_distribution of the ConformityScore class "
                "are not consistent. "
                "The following equation must be verified: "
                "self.get_estimation_distribution(X, y_pred, "
                "self.get_conformity_scores(X, y, y_pred)) == y"  # noqa: E501
                f"The maximum conformity score is {max_conf_score}."
                "The eps attribute may need to be increased if you are "
                "sure that the two methods are consistent."
            )

    def get_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Get the conformity score considering the symmetrical property if so.

        Parameters
        ----------
        X: NDArray
            Observed feature values.

        y: NDArray
            Observed target values.

        y_pred: NDArray
            Predicted target values.

        Returns
        -------
        NDArray
            Conformity scores.
        """
        conformity_scores = self.get_signed_conformity_scores(X, y, y_pred)
        if self.consistency_check:
            self.check_consistency(X, y, y_pred, conformity_scores)
        if self.sym:
            conformity_scores = np.abs(conformity_scores)
        return conformity_scores

    @staticmethod
    def get_quantile(
        values: NDArray,
        alpha_np: NDArray,
        axis: int,
        method: str
    ) -> NDArray:
        """
        Compute the alpha quantile of the conformity scores considering
        the symmetrical property if so.

        Parameters
        ----------
        values: NDArray
            Values from which the quantile is computed, it can be the
            conformity scores or the conformity scores aggregated with
            the predictions.

        alpha_np: NDArray
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        axis: int
            The axis from which to compute the quantile.

        method: str
            ``"higher"`` or ``"lower"`` the method to compute the quantile.

        Returns
        -------
        NDArray
            Lower and upper quantile of the prediction intervals.
            These quantiles are identical if the score is not symmetrical.
        """
        quantile = np.column_stack([
            np_nanquantile(
                values.astype(float),
                _alpha,
                axis=axis,
                method=method
            )
            for _alpha in alpha_np
        ])
        return quantile

    def get_bounds(
        self,
        X: ArrayLike,
        estimator: RegressorMixin,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        ensemble: bool,
        method: str
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Compute bounds of the prediction intervals from the observed values,
        the estimator of MapieRegressor and the conformity scores.

        Parameters
        ----------
        X: ArrayLike
            Observed feature values.

        estimator: RegressorMixin
            Estimator that is fitted to predict y from X.

        conformity_scores: ArrayLike
            Conformity scores.

        alpha_np: NDArray
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.

        method: str
            The method parameter of MapieRegressor.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            - The predictions itself. (y_pred)
            - The lower bounds of the prediction intervals.
            - The upper bounds of the prediction intervals.
        """
        y_pred, y_pred_low, y_pred_up = estimator.predict(X, ensemble)

        signed = -1 if self.sym else 1

        if method == "plus":
            alpha_low = alpha_np if self.sym else alpha_np / 2
            alpha_up = 1 - alpha_np if self.sym else 1 - alpha_np / 2

            values_low = self.get_estimation_distribution(
                X, y_pred_low, signed * conformity_scores
            )
            values_up = self.get_estimation_distribution(
                X, y_pred_up, conformity_scores
            )
            bound_low = self.get_quantile(
                values_low, alpha_low, axis=1, method="lower"
            )
            bound_up = self.get_quantile(
                values_up, alpha_up, axis=1, method="higher"
            )
        else:
            quantile_search = "higher" if self.sym else "lower"
            alpha_low = 1 - alpha_np if self.sym else alpha_np / 2
            alpha_up = 1 - alpha_np if self.sym else 1 - alpha_np / 2

            quantile_low = self.get_quantile(conformity_scores, alpha_low,
                                             axis=0, method=quantile_search)
            quantile_up = self.get_quantile(conformity_scores, alpha_up,
                                            axis=0, method="higher")
            bound_low = self.get_estimation_distribution(
                X, y_pred_low, signed * quantile_low
            )
            bound_up = self.get_estimation_distribution(
                X, y_pred_up, quantile_up
            )

        return y_pred, bound_low, bound_up
