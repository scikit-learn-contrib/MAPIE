from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from ._typing import ArrayLike, NDArray
from .utils import check_alpha, check_n_jobs, check_verbose


class MapieMultiLabelClassifier(BaseEstimator, ClassifierMixin):
    """
    Prediction sets for multilabel-classification.

    This class implements two conformal prediction methods for
    estimating prediction sets for multilabel-classification.
    It guarantees (under the hypothesis of exchangeability) that
    a risk is at least 1 - alpha (alpha is a user-specified parameter).
    For now, we consider the recall as risk.

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any fitted multi-label classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods).
        If ``None``, estimator by default is a sklearn LogisticRegression
        instance.

         by default ``None``

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        For this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        "None" is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        to evaluate quantiles and prediction sets in cumulated_score method.
        Pass an int for reproducible output across multiple function calls.

        By default ``1``.

    verbose : int, optional
        The verbosity level, used with joblib for parallel processing.
        For the moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods. Either CRC or RCPS
    valid_methods: List[Union[str, ``None``]]
        List of all valid bounds computation for RCPS only.
    single_estimator_ : sklearn.ClassifierMixin
        Estimator fitted on the whole training set.

    n_lambdas: int
        Number of thresholds on which we compute the risk.

    lambdas: NDArray
        Array with all the values of lambda.

    risks : ArrayLike of shape (n_samples_cal, n_lambdas)
        The risk for each observation for each threshold

    r_hat : ArrayLike of shape (n_lambdas)
        Average risk for each lambda

    r_hat_plus: ArrayLike of shape (n_lambdas)
        Upper confidence bound for each lambda, computed
        with different bounds (see predict). Only relevant when
        method="rcps".

    lambdas_star: ArrayLike of shape (n_lambdas)
        Optimal threshold for a given alpha.

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
    >>> import numpy as np
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mapie.multi_label_classification import MapieMultiLabelClassifier
    >>> X_toy = np.arange(4).reshape(-1, 1)
    >>> y_toy = np.stack([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]])
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    >>> mapie = MapieMultiLabelClassifier(estimator=clf).fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.3)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False  True]
     [ True False  True]
     [False  True False]
     [False  True False]]
    """
    valid_methods_ = ["crc", "rcps"]
    valid_bounds_ = ["hoeffding", "bernstein", "wsr", None]
    lambdas = np.arange(0, 1, 0.01)
    n_lambdas = len(lambdas)
    fit_attributes = [
        "single_estimator_",
        "risks"
    ]
    sigma_init = .25  # Value given in the paper.
    cal_size = .3

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
        Check n_jobs, verbose and random_states.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_method(self) -> None:
        """
        Check that the specified method is valid

        Raises
        ------
        ValueError
            Raise error if the name of the method is not
            in self.valid_methods_
        """
        if self.method not in self.valid_methods_:
            raise ValueError(
                "Invalid method. "
                "Allowed values are" + " or ".join(self.valid_methods_)
            )

    def _check_all_labelled(self, y: NDArray) -> None:
        """
        Check that all observations have at least
        one label

        Parameters
        ----------
        y : NDArray of shape (n_samples, n_labels)
            Labels of the observations.

        Raises
        ------
        ValueError
            Raise error if at least one observation
            has no label.
        """
        if not (y.sum(axis=1) > 0).all():
            raise ValueError(
                "Invalid y. "
                "All observations should contain at "
                "least one label."
            )

    def _check_delta(self, delta: Optional[float]):
        """
        Check that delta is not ``None`` when the
        method is RCPS and that it is between 0 and 1.

        Parameters
        ----------
        delta : float
            Probability with which we control the risk. The higher
            the probability, the more conservative the algorithm will be.

        Raises
        ------
        ValueError
            If delta is ``None`` and method is RCSP or
            if delta is not in [0, 1] and method
            is RCPS.
        Warning
            If delta is not ``None`` and method is CRC
        """
        if (self.method == "rcps") and (delta is None):
            raise ValueError(
                "Invalid delta. "
                "delta cannot be ``None`` when using "
                "RCPS method."
            )
        if (not isinstance(delta, float)) and (delta is not None):
            raise ValueError(
                "Invalid delta. "
                f"delta must be a float, not a {type(delta)}"
            )
        if (delta is not None) and (
            ((delta <= 0) or (delta >= 1)) and
            (self.method == "rcps")
        ):
            raise ValueError(
                "Invalid delta. "
                "delta must be in ]0, 1["
            )
        if (self.method == "crc") and (delta is not None):
            warnings.warn(
                "WARNING: you are using crc method, hence "
                + "even if the delta is not ``None``, it won't be"
                + "taken into account"
            )

    def _check_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
        _refit: Optional[bool] = False,
    ) -> ClassifierMixin:
        """
        Check the estimator value. If it is ``None``,
        it returns a multi-output ``LogisticRegression``
        instance if necessary.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        estimator : Optional[ClassifierMixin], optional
            Estimator to check, by default ``None``

        _refit : Optional[bool]
            Whether or not the user is using fit (True) or
            partial_fit (False).

            By default False


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

        Warning
            If estimator is then to warn about the split of the
            data between train and calibration
        """
        if (estimator is None) and (not _refit):
            raise ValueError(
                "Invalid estimator with partial_fit. "
                "If the estimator is ``None`` you can not "
                "use partial_fit."
            )
        if (estimator is None) and (_refit):
            estimator = MultiOutputClassifier(
                LogisticRegression(multi_class="multinomial")
            )
            X_train, X_calib, y_train, y_calib = train_test_split(
                    X,
                    y,
                    test_size=self.calib_size,
                    random_state=self.random_state,
            )
            estimator.fit(X_train, y_train)
            warnings.warn(
                "WARNING: To avoid overffiting, X has been splitted"
                + "into X_train and X_calib. The calibraiton will only"
                + "be done on X_calib"
            )
            return estimator, X_calib, y_calib

        if isinstance(estimator, Pipeline):
            est = estimator[-1]
        else:
            est = estimator
        if (
            not hasattr(est, "fit")
            or not hasattr(est, "predict")
            or not hasattr(est, "predict_proba")
        ):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )
        check_is_fitted(est)
        return estimator, X, y

    def _check_partial_fit_first_call(self) -> bool:
        """
        Check that this is the first time partial_fit
        or fit is called.

        Returns
        -------
        bool
            True if it is the first time, else False.
        """
        return not hasattr(self, "risks")

    def _check_bound(self, bound: Optional[str]):
        """
        Check the value of the bound.

        Parameters
        ----------
        bound : Optional[str]
            Bound defined in the predict.

        Raises
        ------
        AttributeError
            If bound is not in ["hoeffding", "bernstein", "wsr", ``None``]

        Warning
            If bound is not ``None``and method is CRC
        """
        if bound not in self.valid_bounds_:
            raise ValueError(
                "bound must be in ['hoeffding', 'bernstein', 'wsr', ``None``]"
            )
        elif (bound is not None) and (self.method == "crc"):
            warnings.warn(
                "WARNING: you are using crc method, hence "
                + "even if the bound is not ``None``, it won't be"
                + "taken into account"
            )

    def _transform_pred_proba(
        self,
        y_pred_proba: Union[Sequence[NDArray], NDArray]
    ) -> NDArray:
        """If the output of the predict_proba is a list of arrays (output of
        the ``predict_proba`` of ``MultiOutputClassifier``) we transform it
        into an array of shape (n_samples, n_classes, 1), otherwise, we add
        one dimension at the end.

        Parameters
        ----------
        y_pred_proba : Union[List, NDArray]
            Output of the multi-label classifier.

        Returns
        -------
        NDArray of shape (n_samples, n_classe, 1)
            Output of the model ready for risk computation.
        """
        if isinstance(y_pred_proba, np.ndarray):
            y_pred_proba_array = y_pred_proba
        else:
            y_pred_proba_stacked = np.stack(
                y_pred_proba,  # type: ignore
                axis=0
            )[:, :, 1]
            y_pred_proba_array = np.moveaxis(y_pred_proba_stacked, 0, -1)

        return np.expand_dims(y_pred_proba_array, axis=2)

    def _compute_risks(self, y_pred_proba: NDArray, y: NDArray) -> NDArray:
        """
        Compute the risks

        Parameters
        ----------
        y_pred_proba : NDArray of shape (n_samples, n_labels)
            Predicted probabilities for each label and each observation
        y : NDArray of shape (n_samples, n_labels)
            True labels.

        Returns
        -------
        NDArray of shape (n_samples, n_lambdas)
            Risks for each observation and each value of lambda.
        """
        y_pred_proba_repeat = np.repeat(
            y_pred_proba,
            self.n_lambdas,
            axis=2
        )

        y_pred_th = (y_pred_proba_repeat > self.lambdas).astype(int)

        y_repeat = np.repeat(y[..., np.newaxis], self.n_lambdas, axis=2)
        risks = 1 - (
            (y_pred_th * y_repeat).sum(axis=1) /
            y.sum(axis=1)[:, np.newaxis]
        )

        return risks

    def _get_r_hat_plus(
        self,
        bound: Optional[str],
        delta: Optional[float],
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
            value is the same as in the paper implementation.

            By default .25

        Returns
        -------
        Tuple[NDArray, NDArray] of shape (n_lambdas, ) and (n_lambdas)
            Average risk over all the obervations and upper bound of the risk.
        """
        r_hat = self.risks.mean(axis=0)
        n_obs = len(self.risks)

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

            else:
                mu_hat = (
                    (.5 + np.cumsum(self.risks, axis=0)) /
                    (np.repeat(
                        [range(1, n_obs + 1)],
                        self.n_lambdas,
                        axis=0
                    ).T + 1)
                )
                sigma_hat = (
                    (.25 + np.cumsum((self.risks - mu_hat)**2, axis=0)) /
                    (np.repeat(
                        [range(1, n_obs + 1)],
                        self.n_lambdas,
                        axis=0
                    ).T + 1)
                )
                sigma_hat = np.concatenate(
                    [
                        np.full(
                            (1, self.n_lambdas), fill_value=self.sigma_init
                        ), sigma_hat[:-1]
                    ]
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
                K_R_max = np.zeros((self.n_lambdas, self.n_lambdas))
                for batch in batches:
                    nu_batch = nu[batch]
                    losses_batch = self.risks[batch]

                    nu_batch = np.repeat(
                        np.expand_dims(nu_batch, axis=2),
                        self.n_lambdas,
                        axis=2
                    )
                    losses_batch = np.repeat(
                        np.expand_dims(losses_batch, axis=2),
                        self.n_lambdas,
                        axis=2
                    )

                    R = self.lambdas
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

                r_hat_plus_tronc = self.lambdas[np.argwhere(
                    np.cumsum(K_R_max > -np.log(delta), axis=1) == 1
                )[:, 1]]
                r_hat_plus = np.ones(self.n_lambdas)
                r_hat_plus[:len(r_hat_plus_tronc)] = r_hat_plus_tronc

        else:
            r_hat_plus = (n_obs / (n_obs + 1)) * r_hat + (1 / (n_obs + 1))

        return r_hat, r_hat_plus

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
        r_hat_plus : NDArray of shape (n_lambdas, )
            Upper bounds computed in the `get_r_hat_plus` method.
        alphas : NDArray of shape (n_alphas, )
            Risk levels.

        Returns
        -------
        NDArray of shape (n_alphas, )
            Optimal lambdas which control the risks for each value
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
        )  # to avoid an error if the risk is always higher than alpha
        lambdas_star = self.lambdas[np.argmin(
                - np.greater_equal(
                    bound_rep,
                    alphas_np
                ).astype(int),
                axis=1
            )]

        return lambdas_star

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        _refit: Optional[bool] = False,
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

        _refit: bool
            Whether or not refit from scratch.

            By default False

        Returns
        -------
        MapieMultiLabelClassifier
            The model itself.
        """
        # Checks
        first_call = self._check_partial_fit_first_call()
        self._check_parameters()

        X, y = indexable(X, y)
        _check_y(y, multi_output=True)
        estimator, X, y = self._check_estimator(
            X, y, self.estimator,
            _refit
        )

        y = cast(NDArray, y)
        X = cast(NDArray, X)

        self._check_all_labelled(y)
        self.n_samples_ = _num_samples(X)

        # Work
        if first_call or _refit:
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
            self.theta_ = X.shape[1]
            self.risks = self._compute_risks(y_pred_proba_array, y)
        else:
            if X.shape[1] != self.theta_:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_))
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
            partial_risk = self._compute_risks(y_pred_proba_array, y)
            self.risks = np.concatenate([self.risks, partial_risk], axis=0)

        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        calib_size: Optional[float] = .3
    ) -> MapieMultiLabelClassifier:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : NDArray of shape (n_samples, n_classes)
            Training labels.

        calib_size: Optional[float]
            Size of the calibration dataset with respect to X if the
            given model is ``None`` need to fit a LogisticRegression.

            By default .3

        Returns
        -------
        MapieMultiLabelClassifier
            The model itself.
        """
        self.calib_size = calib_size
        return self.partial_fit(X, y, _refit=True)

    def predict(
        self,
        X: ArrayLike,
        method: Optional[str] = "crc",
        alpha: Optional[Union[float, Iterable[float]]] = None,
        delta: Optional[float] = None,
        bound: Optional[Union[str, None]] = "wsr"
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Prediction sets on new samples based on target confidence
        interval.
        Prediction sets for a given ``alpha`` are deduced from the computed
        risks.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.
        method : Optional[str]
            Method to choose for prediction interval estimates.
            Choose among:

            - "crc", for Conformal Risk Control. See [1] for
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
            Between 0 and 1, the level of certainty at which we compute
            the Upper Confidence Bound of the average risk.
            Lower ``delta`` produce larger (more conservative) prediction
            sets.
            By default ``None``.
        bound : Optional[Union[str, ``None``]]
            Method used to compute the Upper Confidence Bound of the
            average risk. Only necessary with the RCPS method.
            By default "wsr"

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is ``None``.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples, n_classes) and (n_samples, n_classes, n_alpha)
        if alpha is not ``None``.
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
        )
        self.lambdas_star = self._find_lambda_star(self.r_hat_plus, alpha_np)
        y_pred_proba_array = (
            y_pred_proba_array >
            self.lambdas_star[np.newaxis, np.newaxis, :]
        )
        return y_pred, y_pred_proba_array
