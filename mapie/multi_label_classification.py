from __future__ import annotations
from typing import Optional, Union, Tuple, Iterable, List, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    _num_samples,
    _check_y,
)

from ._typing import ArrayLike, NDArray
from .utils import (
    check_null_weight,
    check_alpha,
    check_n_jobs,
    check_verbose,
    get_r_hat_plus
)


class MapieMultiLabelClassifier(BaseEstimator, ClassifierMixin):

    valid_methods_ = ["crc", "rcps"]
    n_lambdas = 100
    fit_attributes = [
        "single_estimator_",
        "risks"
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "crc",
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        if self.method not in self.valid_methods_:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'crc' or 'rcps"
            )
        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
    ) -> ClassifierMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LogisticRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        estimator : Optional[ClassifierMixin], optional
            Estimator to check, by default ``None``

        Returns
        -------
        ClassifierMixin
            The estimator itself or a default ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit, predict, nor predict_proba methods.

        NotFittedError
            If the estimator is not fitted and ``cv`` attribute is "prefit".
        """
        if isinstance(estimator, Pipeline):
            est = estimator[-1]
        else:
            est = estimator
        if (
            not hasattr(est, "fit")
            and not hasattr(est, "predict")
            and not hasattr(est, "predict_proba")
        ):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )
        check_is_fitted(est)
        if not hasattr(est, "classes_"):
            raise AttributeError(
                "Invalid classifier. "
                "Fitted classifier does not contain "
                "'classes_' attribute."
            )
        return estimator

    def _transform_pred_proba(self, y_pred_proba):
        y_pred_proba_stacked = np.stack(y_pred_proba, 0)[:, :, 1]
        y_pred_proba_array = np.moveaxis(y_pred_proba_stacked, 0, -1)

        return y_pred_proba_array[..., np.newaxis]

    def _compute_risks(self, y_pred_proba, y):
        if type(y_pred_proba) == list:
            y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
        else:
            y_pred_proba_array = y_pred_proba

        lambdas = np.arange(
            0, 1,
            1 / self.n_lambdas
        )[:, np.newaxis, np.newaxis]

        y_pred_proba_repeat = np.repeat(
            y_pred_proba_array,
            self.n_lambdas,
            axis=2
        )

        y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

        y_repeat = np.repeat(y[..., np.newaxis], self.n_lambdas, axis=2)
        risks = 1 - y_pred_th * y_repeat / y.sum(axis=1)[:, np.newaxis]

        return risks

    def find_lambda_star(self, r_hat_plus, alphas):
        """Find the optimal lambda for each of the bound in
        r_hat_plus

        Parameters
        ----------
        r_hat_plus : Dict[str, np.ndarray]
            Upper bounds computed in the get_r_hat_plus function.
        alphas : Union[float, List[float]]
            Risk levels.

        Returns
        -------
        Dict[str, float]
            Optimal lambdas which controls the risks.
        """

        alphas_np = np.zeros((len(alphas), 1))
        alphas_np[:, 0] = alphas

        bound_rep = np.repeat(
            np.expand_dims(r_hat_plus, axis=0),
            len(alphas),
            axis=0
        )

        lambdas_star = np.argmin(
                - np.greater_equal(
                    bound_rep,
                    alphas_np
                ).astype(int),
                axis=1
            ) / len(r_hat_plus)

        return lambdas_star

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieMultiLabelClassifier:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : NDArray of shape (n_samples, n_classes)
            Training labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no prediction sets.

            By default None.

        Returns
        -------
        MapieMultiLabelClassifier
            The model itself.
        """
        # Checks
        self._check_parameters()
        estimator = self._check_estimator(X, y, self.estimator)

        X, y = indexable(X, y)
        y = _check_y(y)
        assert type_of_target(y) == "multilabel-indicator"
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)

        self.n_classes_ = len(y.shape[1])
        self.n_samples_ = _num_samples(X)

        # Work
        self.single_estimator_ = estimator
        y_pred_proba = self.single_estimator_.predict_proba(X)

        self.risks = self._compute_risks(y_pred_proba, y)

        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        delta: Optional[Union[float, Iterable[float]]] = None,
        bound: Optional[Union[str, List[str]]] = "wsr"
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:

        alpha = cast(Optional[NDArray], check_alpha(alpha))
        check_is_fitted(self, self.fit_attributes)

        # Estimate prediction sets
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)

        # Estimate of probabilities from estimator(s)
        # In all cases : len(y_pred_proba.shape) == 3
        # with  (n_test, n_classes, n_alpha or n_train_samples)
        alpha_np = cast(NDArray, alpha)

        y_pred_proba = self.single_estimator_.predict_proba(X)
        y_pred_proba = np.repeat(
            y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
        )

        r_hat_plus = get_r_hat_plus(self.risks, bound, delta)

        lambdas_star = self.find_lambda_star(r_hat_plus, alpha_np)

        return y_pred, lambdas_star
