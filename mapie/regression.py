from __future__ import annotations
from typing import Optional, Union, Iterable, Tuple, List, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from ._typing import ArrayLike
from .utils import (
    check_null_weight,
    fit_estimator,
    check_n_features_in,
    check_alpha,
    check_alpha_and_n_samples,
    check_n_jobs,
    check_verbose
)


class MapieRegressor(BaseEstimator, RegressorMixin):  # type: ignore
    """
    Prediction interval with out-of-fold residuals.

    This class implements the jackknife+ strategy and its variations
    for estimating prediction intervals on single-output data. The
    idea is to evaluate out-of-fold residuals on hold-out validation
    sets and to deduce valid confidence intervals with strong theoretical
    guarantees.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with fit and predict methods), by default None.
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "naive", based on training set residuals,
        - "base", based on validation sets residuals,
        - "plus", based on validation residuals and testing predictions,
        - "minmax", based on validation residuals and testing predictions
          (min/max among cross-validation clones).

        By default "plus".

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing residuals.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: ``sklearn.model_selection.LeaveOneOut()``
          (jackknife variants) or ``sklearn.model_selection.KFold()``
          (cross-validation variants)
        - ``"prefit"``, assumes that ``estimator`` has been fitted already,
          and the ``method`` parameter is ignored.
          All data provided in the ``fit`` method is then used
          for computing residuals only.
          At prediction time, quantiles of these residuals are used to provide
          a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          residual estimate are disjoint.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for ‘unset’ that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    ensemble: bool, optional
        Determines how to return the predictions.
        If ``False``, returns the predictions from the single estimator
        trained on the full training dataset.
        If ``True``, returns the median of the prediction intervals computed
        from the out-of-folds models.
        The Jackknife+ interval can be interpreted as an interval
        around the median prediction,
        and is guaranteed to lie inside the interval,
        unlike the single estimator predictions.

        By default ``False``.

    verbose : int, optional
        The verbosity level, used with joblib for multiprocessing.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    single_estimator_ : sklearn.RegressorMixin
        Estimator fitted on the whole training set.

    estimators_ : list
        List of out-of-folds estimators.

    residuals_ : np.ndarray of shape (n_samples_train,)
        Residuals between ``y_train`` and ``y_pred``.

    k_: np.ndarray of shape(n_samples_train,)
        Id of the fold containing each trainig sample.

    n_features_in_: int
        Number of features passed to the fit method.

    n_samples_val_: List[int]
        Number of samples passed to the fit method.

    References
    ----------
    Rina Foygel Barber, Emmanuel J. Candès,
    Aaditya Ramdas, and Ryan J. Tibshirani.
    "Predictive inference with the jackknife+."
    Ann. Statist., 49(1):486–507, February 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import MapieRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    >>> y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> pireg = MapieRegressor(LinearRegression()).fit(X_toy, y_toy)
    >>> y_pred, y_pis = pireg.predict(X_toy, alpha=0.5)
    >>> print(y_pis[:, :, 0])
    [[ 4.7972973   5.8       ]
     [ 6.69767442  7.65540541]
     [ 8.59883721  9.58108108]
     [10.5        11.40116279]
     [12.4        13.30232558]
     [14.25       15.20348837]]
    >>> print(y_pred)
    [ 5.28571429  7.17142857  9.05714286 10.94285714 12.82857143 14.71428571]
    """

    valid_methods_ = [
        "naive",
        "base",
        "plus",
        "minmax"
    ]

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "plus",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        ensemble: bool = False,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble
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
                "Allowed values are 'naive', 'base', 'plus' and 'minmax'."
            )

        if not isinstance(self.ensemble, bool):
            raise ValueError(
                "Invalid ensemble argument. Must be a boolean."
            )

        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)

    def _check_estimator(
        self,
        estimator: Optional[RegressorMixin] = None
    ) -> RegressorMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LinearRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator : Optional[RegressorMixin], optional
            Estimator to check, by default ``None``

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``LinearRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit nor predict methods.

        NotFittedError
            If the estimator is not fitted and ``cv`` attribute is "prefit".
        """
        if estimator is None:
            return LinearRegression()
        if not hasattr(estimator, "fit") and not hasattr(estimator, "predict"):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a regressor with fit and predict methods."
            )
        if self.cv == "prefit":
            if isinstance(self.estimator, Pipeline):
                check_is_fitted(self.estimator[-1])
            else:
                check_is_fitted(self.estimator)
        return estimator

    def _check_cv(
        self,
        cv: Optional[Union[int, str, BaseCrossValidator]] = None
    ) -> Union[str, BaseCrossValidator]:
        """
        Check if cross-validator is
        ``None``, ``int``, ``"prefit"``, ``KFold`` or ``LeaveOneOut``.
        Return a ``LeaveOneOut`` instance if integer equal to -1.
        Return a ``KFold`` instance if integer superior or equal to 2.
        Return a ``KFold`` instance if ``None``.
        Else raise error.

        Parameters
        ----------
        cv : Optional[Union[int, str, BaseCrossValidator]], optional
            Cross-validator to check, by default ``None``.

        Returns
        -------
        Union[str, BaseCrossValidator]
            The cross-validator itself or a default ``KFold`` instance.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv is None:
            return KFold(n_splits=5)
        if isinstance(cv, int):
            if cv == -1:
                return LeaveOneOut()
            if cv >= 2:
                return KFold(n_splits=cv)
        if (
            isinstance(cv, KFold)
            or isinstance(cv, LeaveOneOut)
            or cv == "prefit"
        ):
            return cv
        raise ValueError(
            "Invalid cv argument. "
            "Allowed values are None, -1, int >= 2, 'prefit', "
            "KFold or LeaveOneOut."
        )

    def _fit_and_predict_oof_model(
        self,
        estimator: RegressorMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        val_index: ArrayLike,
        k: int,
        sample_weight: Optional[ArrayLike] = None
    ) -> Tuple[RegressorMixin, ArrayLike, ArrayLike, ArrayLike]:
        """
        Fit a single out-of-fold model on a given training set and
        perform predictions on a test set.

        Parameters
        ----------
        estimator : RegressorMixin
            Estimator to train.

        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
            Input labels.

        train_index : np.ndarray of shape (n_samples_train)
            Training data indices.

        val_index : np.ndarray of shape (n_samples_val)
            Validation data indices.

        k : int
            Split identification number.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default None.

        Returns
        -------
        Tuple[RegressorMixin, ArrayLike, ArrayLike, ArrayLike]

        - [0]: Fitted estimator
        - [1]: Estimator predictions on the validation fold,
          of shape (n_samples_val,)
        - [2]: Identification number of the validation fold,
          of shape (n_samples_val,)
        - [3]: Validation data indices,
          of shape (n_samples_val,)
        """
        X_train, y_train, X_val = X[train_index], y[train_index], X[val_index]
        if sample_weight is None:
            estimator = fit_estimator(estimator, X_train, y_train)
        else:
            estimator = fit_estimator(
                estimator, X_train, y_train, sample_weight[train_index]
            )
        y_pred = estimator.predict(X_val)
        val_id = np.full_like(y_pred, k)
        return estimator, y_pred, val_id, val_index

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None
    ) -> MapieRegressor:
        """
        Fit estimator and compute residuals used for prediction intervals.
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold residuals are stored under the ``residuals_`` attribute.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.

            By default None.

        Returns
        -------
        MapieRegressor
            The model itself.
        """
        # Checks
        self._check_parameters()
        cv = self._check_cv(self.cv)
        estimator = self._check_estimator(self.estimator)
        X, y = check_X_y(
            X, y, force_all_finite=False, dtype=["float64", "int", "object"]
        )
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        # Initialization
        self.estimators_: List[RegressorMixin] = []
        self.k_ = np.empty_like(y, dtype=int)
        y_pred = np.empty_like(y, dtype=float)

        # Work
        if cv == "prefit":
            self.single_estimator_ = estimator
            y_pred = self.single_estimator_.predict(X)
            self.n_samples_val_ = [X.shape[0]]
        else:
            self.single_estimator_ = fit_estimator(
                clone(estimator), X, y, sample_weight
            )
            if self.method == "naive":
                y_pred = self.single_estimator_.predict(X)
                self.n_samples_val_ = [X.shape[0]]
            else:
                outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._fit_and_predict_oof_model)(
                        clone(estimator),
                        X, y, train_index, val_index, k, sample_weight
                    ) for k, (train_index, val_index) in enumerate(cv.split(X))
                )
                self.estimators_, predictions, val_ids, val_indices = map(
                    list, zip(*outputs)
                )
                self.n_samples_val_ = [
                    np.array(pred).shape[0] for pred in predictions
                ]
                predictions, val_ids, val_indices = map(
                    np.concatenate, (predictions, val_ids, val_indices)
                )
                self.k_[val_indices] = val_ids
                y_pred[val_indices] = predictions
        self.residuals_ = np.abs(y - y_pred)
        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of residuals (naive and base methods)
        - quantiles of (predictions +/- residuals) (plus methods)
        - quantiles of (max/min(predictions) +/- residuals) (minmax methods)

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``np.ndarray`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.
            By default ``None``.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]

        - np.ndarray of shape (n_samples,) if alpha is None

        - Tuple[np.ndarray, np.ndarray] of shapes
        (n_samples,) and (n_samples, 2, n_alpha) if alpha is not None

            - [:, 0, :]: Lower bound of the prediction interval
            - [:, 1, :]: Upper bound of the prediction interval
        """
        # Checks
        check_is_fitted(
            self,
            [
                "single_estimator_",
                "estimators_",
                "k_",
                "residuals_",
                "n_features_in_",
                "n_samples_val_",
            ]
        )
        alpha_ = check_alpha(alpha)
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)
        else:
            # SHAPES:
            # (n_samples_train,) : self.residuals_
            # (n_alpha,) : alpha
            # (n_samples_test, n_alpha) : y_pred_low, y_pred_up
            # (n_samples_test, n_samples_train) : y_pred_multi, low/up_bounds
            alpha_ = cast(np.ndarray, alpha_)
            check_alpha_and_n_samples(alpha_, self.residuals_.shape[0])
            if self.method in ["naive", "base"] or self.cv == "prefit":
                quantile = np.quantile(
                    self.residuals_, 1 - alpha_, interpolation="higher"
                )
                y_pred_low = y_pred[:, np.newaxis] - quantile
                y_pred_up = y_pred[:, np.newaxis] + quantile
            else:
                y_pred_multi = np.column_stack(
                    [e.predict(X) for e in self.estimators_]
                )
                if self.method == "plus":
                    # At this point,
                    # y_pred_multi is of shape (n_samples_test, n_estimators_)
                    # We thus enforce y_pred_multi
                    # to be of shape (n_samples_test, n_samples_train)
                    if len(self.estimators_) < len(self.k_):
                        y_pred_multi = y_pred_multi[:, self.k_]
                    lower_bounds = y_pred_multi - self.residuals_
                    upper_bounds = y_pred_multi + self.residuals_
                if self.method == "minmax":
                    lower_bounds = np.min(y_pred_multi, axis=1, keepdims=True)
                    upper_bounds = np.max(y_pred_multi, axis=1, keepdims=True)
                    lower_bounds = lower_bounds - self.residuals_
                    upper_bounds = upper_bounds + self.residuals_
                y_pred_low = np.column_stack([
                    np.quantile(
                        lower_bounds,
                        _alpha,
                        axis=1,
                        interpolation="lower"
                    ) for _alpha in alpha_
                ])
                y_pred_up = np.column_stack([
                    np.quantile(
                        upper_bounds,
                        1 - _alpha,
                        axis=1,
                        interpolation="higher"
                    ) for _alpha in alpha_
                ])
                if self.ensemble:
                    y_pred = np.median(y_pred_multi, axis=1)
            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
