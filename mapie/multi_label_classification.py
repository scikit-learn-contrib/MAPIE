from __future__ import annotations
from typing import Optional, Union, Tuple, Iterable, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    _num_samples,
)

from ._typing import ArrayLike, NDArray
from .utils import (
    check_null_weight,
    check_alpha,
    check_n_jobs,
    check_verbose,
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
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
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
        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_method(self) -> None:
        if self.method not in self.valid_methods_:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'crc' or 'rcps"
            )

    def _check_delta(self, delta):
        if (self.method == "rcps") and (delta is None):
            raise ValueError(
                "Invalid delta. "
                "delta can not be None when using "
                "a RCPS method."
            )

        if ((delta <= 0) or (delta >= 1)) and self.method == "rcps":
            raise ValueError(
                "Invalid delta. "
                "delta must be in the ]0, 1[ interval"
            )

    def _check_y(self, y) -> None:
        if not type_of_target(y) == "multilabel-indicator":
            raise ValueError(
                "Invalid target type. "
                "The target should be an array of shape "
                "(n_samples, n_labels)"
            )

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
        if type(y_pred_proba) == np.ndarray:
            y_pred_proba_array = y_pred_proba
        else:
            y_pred_proba_stacked = np.stack(y_pred_proba, 0)[:, :, 1]
            y_pred_proba_array = np.moveaxis(y_pred_proba_stacked, 0, -1)

        return np.expand_dims(y_pred_proba_array, axis=2)

    def _compute_risks(self, y_pred_proba, y):
        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)

        lambdas = np.arange(
            0, 1,
            1 / self.n_lambdas
        )[np.newaxis, np.newaxis, :]

        y_pred_proba_repeat = np.repeat(
            y_pred_proba_array,
            self.n_lambdas,
            axis=2
        )

        y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

        y_repeat = np.repeat(y[..., np.newaxis], self.n_lambdas, axis=2)
        risks = 1 - (
            (y_pred_th * y_repeat).sum(axis=1) /
            y.sum(axis=1)[:, np.newaxis]
        )

        return risks

    def _find_lambda_star(self, r_hat_plus, alpha_np):
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

        if len(alpha_np) > 1:
            alphas_np = alpha_np[:, np.newaxis]
        else:
            alphas_np = alpha_np

        bound_rep = np.repeat(
            np.expand_dims(r_hat_plus, axis=0),
            len(alphas_np),
            axis=0
        )
        bound_rep[:, np.argmax(bound_rep, axis=1)] = np.maximum(
            alphas_np,
            bound_rep[:, np.argmax(bound_rep, axis=1)]
        )
        lambdas_star = np.argmin(
                - np.greater_equal(
                    bound_rep,
                    alphas_np
                ).astype(int),
                axis=1
            ) / len(r_hat_plus)

        return lambdas_star

    def _get_r_hat_plus(self, bound, delta, sigma_init=.25):
        """Compute the upper bound of the loss for each lambda.

        Parameters
        ----------
        losses : np.ndarray of shape (n_samples, n_lambdas)
            Loss computed with the get_losses function.
        bound : str
            Bounds to compute. Either hoeffding, bernstein or wsr.
        delta : float
            Level of confidence.
        sigma_init : float, optional
            First variance in the sigma_hat array. The default
            value is the same as in the paper implmentation.

            By default .25

        Returns
        -------
        Dict[str, np.ndarray]
            Upper bound of the loss for each technique.
        """

        assert (
            bound in ["hoeffding", "bernstein", "wsr", None]
        ), 'bounds must be in ["hoeffding", "bernstein", "wsr", None]'
        r_hat = self.risks.mean(axis=0)
        n_obs = len(self.risks)
        n_lambdas = len(r_hat)

        if self.method == "rcps":
            if bound == "hoeffding":
                r_hat_plus = (
                    r_hat +
                    np.sqrt((1 / (2 * n_obs)) * np.log(1 / delta))
                )

            elif bound == "bernstein":
                sigma_hat_bern = np.var(r_hat, axis=0, ddof=1)
                r_hat_plus = (
                    r_hat +
                    np.sqrt((sigma_hat_bern * 2 * np.log(2 / delta)) / n_obs) +
                    (7 * np.log(2 / delta)) / (3 * (n_obs - 1))
                )

            elif bound == "wsr":
                mu_hat = (
                    (.5 + np.cumsum(self.risks, axis=0)) /
                    (np.repeat([range(1, n_obs + 1)], n_lambdas, axis=0).T + 1)
                )
                sigma_hat = (
                    (.25 + np.cumsum((self.risks - mu_hat)**2, axis=0)) /
                    (np.repeat([range(1, n_obs + 1)], n_lambdas, axis=0).T + 1)
                )
                sigma_hat = np.concatenate(
                    [np.ones((1, n_lambdas)) * sigma_init, sigma_hat[:-1]]
                )
                nu = np.minimum(
                    1,
                    np.sqrt((2 * np.log(1 / delta)) / (n_obs * sigma_hat))
                )
                batches = {
                    "1": int(n_obs / 2),
                    "2": n_obs - int(n_obs / 2)
                }  # Split the calculation in two to prevent memory issues
                K_R_max = np.zeros((n_lambdas, n_lambdas))
                for batch, n_batch in batches.items():
                    if int(batch) == 1:
                        nu_batch = nu[:n_batch]
                        losses_batch = self.risks[:n_batch]
                    else:
                        nu_batch = nu[n_batch:]
                        losses_batch = self.risks[n_batch:]

                    nu_batch = np.repeat(
                        np.expand_dims(nu_batch, axis=2),
                        n_lambdas,
                        axis=2
                    )
                    losses_batch = np.repeat(
                        np.expand_dims(losses_batch, axis=2),
                        n_lambdas,
                        axis=2
                    )

                    R = np.arange(n_lambdas) / n_lambdas
                    K_R = np.cumsum(
                        np.log(
                            (
                                1 -
                                nu_batch *
                                (losses_batch - R)
                            ) +
                            np.finfo(np.float64).eps
                        ),
                        axis=0
                    )
                    K_R = np.max(K_R, axis=0)
                    K_R_max += K_R

                r_hat_plus_tronc = (np.argwhere(
                    np.cumsum(K_R_max > -np.log(delta), axis=1) == 1
                )[:, 1] / n_lambdas)
                r_hat_plus = np.ones(n_lambdas)
                r_hat_plus[:len(r_hat_plus_tronc)] = r_hat_plus_tronc

        elif self.method == "crc":
            r_hat_plus = (n_obs / (n_obs + 1)) * r_hat + (1 / (n_obs + 1))

        return r_hat, r_hat_plus

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        partial: bool = False
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
        self._check_y(y)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)

        self.n_classes_ = y.shape[1]
        self.n_samples_ = _num_samples(X)

        # Work
        self.single_estimator_ = estimator
        y_pred_proba = self.single_estimator_.predict_proba(X)
        if partial:
            if hasattr(self, "risks"):
                partial_risk = self._compute_risks(y_pred_proba, y)
                self.risks = np.concatenate([self.risks, partial_risk], axis=0)
            else:
                self.risks = self._compute_risks(y_pred_proba, y)
        else:
            self.risks = self._compute_risks(y_pred_proba, y)

        return self

    def predict(
        self,
        X: ArrayLike,
        method: Optional[str] = "crc",
        alpha: Optional[Union[float, Iterable[float]]] = None,
        delta: Optional[float] = None,
        bound: Optional[Union[str, None]] = "wsr"
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:

        self.method = method
        self._check_method()
        self._check_delta(delta)
        alpha = cast(Optional[NDArray], check_alpha(alpha))
        check_is_fitted(self, self.fit_attributes)

        # Estimate prediction sets
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)

        alpha_np = cast(NDArray, alpha)

        y_pred_proba = self.single_estimator_.predict_proba(X)

        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
        y_pred_proba_array = np.repeat(
            y_pred_proba_array,
            len(alpha_np),
            axis=2
        )
        self.r_hat, self.r_hat_plus = self._get_r_hat_plus(
            bound,
            delta
        )
        self.lambdas_star = self._find_lambda_star(self.r_hat_plus, alpha_np)
        y_pred_proba_array = (
            y_pred_proba_array >
            self.lambdas_star[np.newaxis, np.newaxis, :]
        )
        return y_pred, y_pred_proba_array
