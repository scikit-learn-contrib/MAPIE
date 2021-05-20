from __future__ import annotations
from typing import Optional, Union, Iterable, Tuple, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut

from ._typing import ArrayLike


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
        Any regressor with scikit-learn API (i.e. with fit and predict methods), by default None.
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

    alpha: Union[float, Iterable[float]], optional
        Can be a float, a list of floats, or a np.ndarray of floats.
        Between 0 and 1, represent the uncertainty of the confidence interval.
        Lower alpha produce larger (more conservative) prediction intervals.
        alpha is the complement of the target coverage level.
        Only used at prediction time. By default 0.1.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "naive", based on training set residuals,
        - "base", based on cross-validation sets residuals,
        - "plus", based on cross-validation sets residuals and testing set predictions,
        - "minmax", based on cross-validation sets residuals and testing set predictions
          (min/max among cross-validation clones).

        By default "plus".

    cv: Optional[Union[int, BaseCrossValidator]]
        The cross-validation strategy for computing residuals. It directly drives the
        distinction between jackknife and cv variants. Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: ``sklearn.model_selection.LeaveOneOut()`` (jackknife variants) or
          ``sklearn.model_selection.KFold()`` (cross-validation variants)

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all, which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for ‘unset’ that will be interpreted as ``n_jobs=1`` (sequential execution).

        By default ``None``.

    ensemble: bool, optional
        Determines how to return the predictions.
        If ``False``, returns the predictions from the single estimator trained on the full training dataset.
        If ``True``, returns the median of the prediction intervals computed from the out-of-folds models.
        The Jackknife+ interval can be interpreted as an interval around the median prediction,
        and is guaranteed to lie inside the interval, unlike the single estimator predictions.

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
        Estimator fit on the whole training set.

    estimators_ : list
        List of out-of-folds estimators.

    residuals_ : np.ndarray of shape (n_samples_train,)
        Residuals between ``y_train`` and ``y_pred``.

    k_: np.ndarray of shape(n_samples_train,)
        Id of the fold containing each trainig sample.

    n_features_in_: int
        Number of features passed to the fit method.

    References
    ----------
    Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani.
    Predictive inference with the jackknife+. Ann. Statist., 49(1):486–507, 022021

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.estimators import MapieRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    >>> y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> pireg = MapieRegressor(LinearRegression())
    >>> print(pireg.fit(X_toy, y_toy).predict(X_toy)[:, :, 0])
    [[ 5.28571429  4.61627907  6.        ]
     [ 7.17142857  6.51744186  7.8       ]
     [ 9.05714286  8.4         9.68023256]
     [10.94285714 10.2        11.58139535]
     [12.82857143 12.         13.48255814]
     [14.71428571 13.8        15.38372093]]
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
        alpha: Union[float, Iterable[float]] = 0.1,
        method: str = "plus",
        cv: Optional[Union[int, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        ensemble: bool = False,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
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
            Is parameters are not valid.
        """
        if self.method not in self.valid_methods_:
            raise ValueError("Invalid method. Allowed values are 'naive', 'base', 'plus' and 'minmax'.")

        if not isinstance(self.ensemble, bool):
            raise ValueError("Invalid ensemble argument. Must be a boolean.")

        if not isinstance(self.n_jobs, (int, type(None))):
            raise ValueError("Invalid n_jobs argument. Must be an integer.")

        if self.n_jobs == 0:
            raise ValueError("Invalid n_jobs argument. Must be different than 0.")

        if not isinstance(self.verbose, int):
            raise ValueError("Invalid verbose argument. Must be an integer.")

        if self.verbose < 0:
            raise ValueError("Invalid verbose argument. Must be non-negative.")

    def _check_estimator(self, estimator: Optional[RegressorMixin] = None) -> RegressorMixin:
        """
        Check if estimator is ``None``, and returns a ``LinearRegression`` instance if necessary.

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
            If the estimator is not ``None`` and has no fit nor predict methods.
        """
        if estimator is None:
            return LinearRegression()
        if not hasattr(estimator, "fit") and not hasattr(estimator, "predict"):
            raise ValueError("Invalid estimator. Please provide a regressor with fit and predict methods.")
        return estimator

    def _check_cv(self, cv: Optional[Union[int, BaseCrossValidator]] = None) -> BaseCrossValidator:
        """
        Check if cross-validator is ``None``, ``int``, ``KFold`` or ``LeaveOneOut``.
        Return a ``LeaveOneOut`` instance if integer equal to -1.
        Return a ``KFold`` instance if integer superior or equal to 2.
        Return a ``KFold`` instance if ``None``.
        Else raise error.

        Parameters
        ----------
        cv : Optional[Union[int, BaseCrossValidator]], optional
            Cross-validator to check, by default ``None``

        Returns
        -------
        BaseCrossValidator
            The cross-validator itself or a default ``KFold`` instance.

        Raises
        ------
        ValueError
            If the cross-validator is not ``None``, not a valid ``int``, nor a valid cross validator.
        """
        if cv is None:
            return KFold(n_splits=5)
        if isinstance(self.cv, int):
            if self.cv == -1:
                return LeaveOneOut()
            if self.cv >= 2:
                return KFold(n_splits=self.cv)
        if isinstance(self.cv, KFold) or isinstance(self.cv, LeaveOneOut):
            return cv
        raise ValueError("Invalid cv argument. Allowed values are None, -1, int >= 2, KFold or LeaveOneOut.")

    def _check_alpha(self, alpha: Union[float, Iterable[float]]) -> np.ndarray:
        """
        Check alpha and prepare it as a np.ndarray

        Parameters
        ----------
        alpha : Union[float, Iterable[float]]
        Can be a float, a list of floats, or a np.ndarray of floats.
        Between 0 and 1, represent the uncertainty of the confidence interval.
        Lower alpha produce larger (more conservative) prediction intervals.
        alpha is the complement of the target coverage level.
        Only used at prediction time. By default 0.1.

        Returns
        -------
        np.ndarray
            Prepared alpha.

        Raises
        ------
        ValueError
            If alpha is not a float or an Iterable of floats between 0 and 1.
        """
        if isinstance(alpha, float):
            alpha_np = np.array([alpha])
        elif isinstance(alpha, Iterable):
            alpha_np = np.array(alpha)
        else:
            raise ValueError("Invalid alpha. Allowed values are float or Iterable.")
        if len(alpha_np.shape) != 1:
            raise ValueError("Invalid alpha. Please provide a one-dimensional list of values.")
        if alpha_np.dtype.type not in [np.float64, np.float32]:
            raise ValueError("Invalid alpha. Allowed values are Iterable of floats.")
        if np.any((alpha_np <= 0) | (alpha_np >= 1)):
            raise ValueError("Invalid alpha. Allowed values are between 0 and 1.")
        return alpha_np

    def _fit_and_predict_oof_model(
        self,
        estimator: RegressorMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        val_index: ArrayLike,
        k: int
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

        Returns
        -------
        Tuple[RegressorMixin, ArrayLike, ArrayLike, ArrayLike]

            - [0]: Fitted estimator
            - [1]: Estimator predictions on the validation fold, of shape (n_samples_val,)
            - [2]: Identification number of the validation fold, of shape (n_samples_val,)
            - [3]: Validation data indices, of shapes (n_samples_val,)
        """
        X_train, y_train, X_val = X[train_index], y[train_index], X[val_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        val_id = np.full_like(y_pred, k)
        return estimator, y_pred, val_id, val_index

    def fit(self, X: ArrayLike, y: ArrayLike) -> MapieRegressor:
        """
        Fit estimator and compute residuals used for prediction intervals.
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold residuals are stored under the ``residuals_`` attribute.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        Returns
        -------
        MapieRegressor
            The model itself.
        """
        self._check_parameters()
        cv = self._check_cv(self.cv)
        estimator = self._check_estimator(self.estimator)
        X, y = check_X_y(X, y, force_all_finite=False, dtype=["float64", "object"])
        y_pred = np.empty_like(y, dtype=float)
        self.estimators_: List[RegressorMixin] = []
        self.n_features_in_ = X.shape[1]
        self.k_ = np.empty_like(y, dtype=int)
        self.single_estimator_ = clone(estimator).fit(X, y)
        if self.method == "naive":
            y_pred = self.single_estimator_.predict(X)
        else:
            cv_outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_and_predict_oof_model)(
                    clone(estimator), X, y, train_index, val_index, k
                ) for k, (train_index, val_index) in enumerate(cv.split(X))
            )
            self.estimators_, predictions, val_ids, val_indices = map(list, zip(*cv_outputs))
            predictions, val_ids, val_indices = map(np.concatenate, (predictions, val_ids, val_indices))
            self.k_[val_indices] = val_ids
            y_pred[val_indices] = predictions
        self.residuals_ = np.abs(y - y_pred)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation. Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of residuals (naive and base methods)
        - quantiles of (predictions +/- residuals) (plus methods)
        - quantiles of (max/min(predictions) +/- residuals) (minmax methods)

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        np.ndarray of shape (n_samples, 3, len(alpha))

            - [:, 0, :]: Center of the prediction interval
            - [:, 1, :]: Lower bound of the prediction interval
            - [:, 2, :]: Upper bound of the prediction interval
        """
        check_is_fitted(self, ["single_estimator_", "estimators_", "k_", "residuals_"])
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)
        alpha = self._check_alpha(self.alpha)
        if self.method in ["naive", "base"]:
            quantile = np.quantile(self.residuals_, 1 - alpha, interpolation="higher")
            # broadcast y_pred to get y_pred_low/up of shape (n_samples_test, len(alpha))
            y_pred_low = y_pred[:, np.newaxis] - quantile
            y_pred_up = y_pred[:, np.newaxis] + quantile
        else:
            y_pred_multi = np.stack([e.predict(X) for e in self.estimators_], axis=1)
            if self.method == "plus":
                if len(self.estimators_) < len(self.k_):
                    y_pred_multi = y_pred_multi[:, self.k_]
                lower_bounds = y_pred_multi - self.residuals_
                upper_bounds = y_pred_multi + self.residuals_
            if self.method == "minmax":
                lower_bounds = np.min(y_pred_multi, axis=1, keepdims=True) - self.residuals_
                upper_bounds = np.max(y_pred_multi, axis=1, keepdims=True) + self.residuals_
            y_pred_low = np.stack([
                np.quantile(lower_bounds, _alpha, axis=1, interpolation="lower") for _alpha in alpha
            ], axis=1)
            y_pred_up = np.stack([
                np.quantile(upper_bounds, 1 - _alpha, axis=1, interpolation="higher") for _alpha in alpha
            ], axis=1)
            if self.ensemble:
                y_pred = np.median(y_pred_multi, axis=1)
        # tile y_pred to get same shape as y_pred_low/up
        y_pred = np.tile(y_pred, (alpha.shape[0], 1)).T
        return np.stack([y_pred, y_pred_low, y_pred_up], axis=1)
