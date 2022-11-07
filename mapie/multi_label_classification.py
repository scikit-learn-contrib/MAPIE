from __future__ import annotations
from typing import Optional, Union, Tuple, Iterable, cast, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    _num_samples,
    _check_y
)

from ._typing import ArrayLike, NDArray
from .utils import (
    check_null_weight,
    check_alpha,
    check_n_jobs,
    check_verbose,
)


class MapieMultiLabelClassifier(BaseEstimator, ClassifierMixin):
    """
    Prediction sets for multilabel-classification.

    This class implements two conformal prediction strategies for
    estimating prediction sets for multi-classification. It guarantees
    (under the hypothesis of exchangeability) that the recall is at
    leat 1 - alpha (alpha being a user-specified parameter).

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any fitted multi-label classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default None.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        At this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        for evaluation quantiles and prediction sets in cumulated_score.
        Pass an int for reproducible output across multiple function calls.

        By default ```1``.

    verbose : int, optional
        The verbosity level, used with joblib for multiprocessing.
        At this moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    single_estimator_ : sklearn.ClassifierMixin
        Estimator fitted on the whole training set.

    n_lambdas: int
        Number of thresolds on which we compute the risk.

    risks : ArrayLike of shape (n_samples_cal, n_lambdas)
        The risk for each observation for each threshold

    r_hat : ArrayLike of shape (n_lambdas)
        Average risk for each lambda

    r_hat_plus: ArrayLike of shape (n_lambdas)
        Upper confidence bound for each lambda, computed
        with different bounds (see predict). Only relevant when
        method="rcps".

    lambdas_star: ArrayLike of shape (n_lambdas)
        Optimal threshold according to alpha.

    References
    ----------
    [1] Lihua Lei Jitendra Malik Stephen Bates, Anastasios Angelopoulos
    and Michael I. Jordan. Distribution-free, risk-controlling prediction
    sets. CoRR, abs/2101.02703, 2021.
    URL https://arxiv.org/abs/2101.02703.39

    [2] Angelopoulos, Anastasios N., Stephen, Bates, Adam, Fisch, Lihua,
    Lei, and Tal, Schuster. "Conformal Risk Control." (2022).

    Examples
    --------
    import numpy as np
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mapie.multi_label_classification import MapieMultiLabelClassifier
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack(
        [[1, 0, 1], [1, 0, 0], [0, 1, 1],
        [0, 1, 0], [0, 0, 1], [1, 1, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    )
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    >>> mapie = MapieMultiLabelClassifier(estimator=clf).fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.2)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False  True]
    [ True False  True]
    [ True False  True]
    [ True False  True]
    [ True  True  True]
    [ True  True  True]
    [ True  True  True]
    [ True  True  True]
    [False  True  True]]
    """
    valid_methods_ = ["crc", "rcps"]
    n_lambdas = 100
    fit_attributes = [
        "single_estimator_",
        "risks"
    ]
    sigma_init = .25

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

    def _check_delta(self, delta: Optional[float]):
        """Check that delta is not ``None`` when the
        method is RCPS and that it has values between
        0 and 1.

        Parameters
        ----------
        delta : float
            Probability with wchi we control the risk.

        Raises
        ------
        ValueError
            If delta is ``None`` and method is RCSP
        ValueError
            If delta value is not in [0, 1] and method
            is RCPS.
        """
        if (self.method == "rcps") and (delta is None):
            raise ValueError(
                "Invalid delta. "
                "delta can not be None when using "
                "a RCPS method."
            )

        if (delta is not None) and (
            ((delta <= 0) or (delta >= 1)) and
            self.method == "rcps"
        ):
            raise ValueError(
                "Invalid delta. "
                "delta must be in the ]0, 1[ interval"
            )

    def _check_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
    ) -> ClassifierMixin:
        """
        Check if estimator is ``None``,
        and returns a multi-output ``LogisticRegression``
        instance if necessary.

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
            The estimator itself or a default multi-output
            ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit, predict, nor predict_proba methods.

        NotFittedError
            If the estimator is not fitted.
        """
        if estimator is None:
            return MultiOutputClassifier(
                LogisticRegression(multi_class="multinomial")
            ).fit(X, y)

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

    def _check_partial_fit_first_call(self) -> bool:
        """Check that this is the first time partial_fit
        or fit is called.

        Returns
        -------
        bool
            True if it is the first time, else False.
        """
        return not hasattr(self, "risks")

    def _check_bound(self, bound: Optional[str]):
        """Check the value on the bound.

        Parameters
        ----------
        bound : Optional[str]
            Bound defined in the predict.

        Raises
        ------
        AttributeError
            If bound in not in ["hoeffding", "bernstein", "wsr", None]
        """
        if bound not in ["hoeffding", "bernstein", "wsr", None]:
            raise AttributeError(
                'bound must be in ["hoeffding", "bernstein", "wsr", None]'
            )

    def _transform_pred_proba(
        self,
        y_pred_proba: Union[Sequence[NDArray], NDArray]
    ) -> NDArray:
        """If the output of the predict_proba is a list of arrays (output of
        the ``predict_proba`` of ``MultiOutputClassifier``) we transform it
        into an array of shape (n_samples, n_classes)

        Parameters
        ----------
        y_pred_proba : Union[List, NDArray]
            Output of the multi-label classifier.

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            Output of the model ready for risk computation.
        """
        if type(y_pred_proba) == np.ndarray:
            y_pred_proba_array = y_pred_proba
        else:
            y_pred_proba_stacked = np.stack(
                y_pred_proba,  # type: ignore
                axis=0
            )[:, :, 1]
            y_pred_proba_array = np.moveaxis(y_pred_proba_stacked, 0, -1)

        return np.expand_dims(y_pred_proba_array, axis=2)

    def _compute_risks(self, y_pred_proba: NDArray, y: NDArray) -> NDArray:
        """Compute the risk

        Parameters
        ----------
        y_pred_proba : NDArray
            Predicted probabilities for each label and each observation
        y : NDArray
            Labels.

        Returns
        -------
        NDArray of shape (n_samples, n_lambdas)
            Risk for each observation and each value of lambda.
        """
        lambdas = np.arange(
            0, 1,
            1 / self.n_lambdas
        )[np.newaxis, np.newaxis, :]

        y_pred_proba_repeat = np.repeat(
            y_pred_proba,
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

    def _find_lambda_star(
        self,
        r_hat_plus: NDArray,
        alpha_np: NDArray
    ) -> NDArray:
        """Find the higher value of lambda such that for
        all smaller lambda, the risk is smaller, for each value
        of alpha.

        Parameters
        ----------
        r_hat_plus : NDArray
            Upper bounds computed in the get_r_hat_plus function.
        alphas : NDArray
            Risk levels.

        Returns
        -------
        NDArray of shape (n_alphas, )
            Optimal lambdas which controls the risks for each value
            of alpha.
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

    def _get_r_hat_plus(
        self, bound: Optional[str],
        delta: Optional[float],
        sigma_init: float = .25
    ) -> Tuple[NDArray, NDArray]:
        """Compute the upper bound of the loss for each lambda.

        Parameters
        ----------
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
        Tuple[NDArray, NDArray] of shape (n_lambdas, ) and (n_lambdas)
            Average risk over all the obervations and upper bound of the risk.
        """
        r_hat = self.risks.mean(axis=0)
        n_obs = len(self.risks)
        n_lambdas = len(r_hat)

        if (self.method == "rcps") and (delta is not None):
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

                # Split the calculation in two to prevent memory issues
                batches = [
                    range(int(n_obs / 2)),
                    range(n_obs - int(n_obs / 2), n_obs)
                ]
                K_R_max = np.zeros((n_lambdas, n_lambdas))
                for batch in batches:
                    nu_batch = nu[batch]
                    losses_batch = self.risks[batch]

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

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieMultiLabelClassifier:
        """
        Fit the base estimator or use the fitted base estimator on
        batch data. All the computed risks will be concatenated each
        time the partial_fit method is called.

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
        first_call = self._check_partial_fit_first_call()
        self._check_parameters()
        estimator = self._check_estimator(X, y, self.estimator)

        X, y = indexable(X, y)
        _check_y(y, multi_output=True)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)
        X = cast(NDArray, X)

        self.n_classes_ = y.shape[1]
        self.n_samples_ = _num_samples(X)

        # Work
        self.single_estimator_ = estimator
        y_pred_proba = self.single_estimator_.predict_proba(X)
        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
        if first_call:
            self.theta_ = np.zeros(X.shape[1])
            self.risks = self._compute_risks(y_pred_proba_array, y)
        else:
            if X.shape[1] != self.theta_.shape:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_.shape[1]))
            partial_risk = self._compute_risks(y_pred_proba_array, y)
            self.risks = np.concatenate([self.risks, partial_risk], axis=0)

        return self

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
        return self.partial_fit(X, y, sample_weight)

    def predict(
        self,
        X: ArrayLike,
        method: Optional[str] = "crc",
        alpha: Optional[Union[float, Iterable[float]]] = None,
        delta: Optional[float] = None,
        bound: Optional[Union[str, None]] = "wsr"
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Prediction prediction sets on new samples based on target confidence
        interval.
        Prediction sets for a given ``alpha`` are deduced from the computed
        risks.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.
        method : Optional[str], optional
            Method to choose for prediction interval estimates.
            Choose among:

            - "crc", for Conformal Risk Control. See [1] fot
            more details.

            - "rcps", based on computation of the upper bound
            of the risk. See [2] for more details.

            By default "crc".

        alpha : Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            sets.
            ``alpha`` is the complement of the target coverage level.
            By default ``None``.
        delta : Optional[float]
            Can be a float, or ``None``. If using method="rcps", then it
            can not be set to ``None``.
            Between 0 and 1, the level of certainty with which we compute
            the Upper Confidence Bound of the average risk.
            Lower ``delta`` produce larger (more conservative) prediction
            sets.
            By default ``None``.
        bound : Optional[Union[str, None]]
            Method used to compute the Upper Confience Bound of the
            average risk. Only necessary with the RCPS method.
            By default "wsr"

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is None.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples, n_classes) and (n_samples, n_classes, n_alpha)
        if alpha is not None.
        """

        self.method = method
        self._check_method()
        self._check_delta(delta)
        self._check_bound(bound)
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
            delta,
            self.sigma_init
        )
        self.lambdas_star = self._find_lambda_star(self.r_hat_plus, alpha_np)
        y_pred_proba_array = (
            y_pred_proba_array >
            self.lambdas_star[np.newaxis, np.newaxis, :]
        )
        return y_pred, y_pred_proba_array
