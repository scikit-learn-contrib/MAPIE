from __future__ import annotations
from typing import Optional, Union, Iterable

import numpy as np
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

    alpha: Union[float, List[float], np.ndarray], optional
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

    ensemble: bool, optional
        Determines how to return the predictions.
        If ``False``, returns the predictions from the single estimator trained on the full training dataset.
        If ``True``, returns the median of the prediction intervals computed from the out-of-folds models.
        The Jackknife+ interval can be interpreted as an interval around the median prediction,
        and is guaranteed to lie inside the interval, unlike the single estimator predictions.

        By default ``False``.

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
        alpha: Iterable[float] = 0.1,
        method: str = "plus",
        cv: Optional[Union[int, BaseCrossValidator]] = None,
        ensemble: bool = False
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.method = method
        self.cv = cv
        self.ensemble = ensemble

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

    def _check_alpha(self, alpha: Iterable[float]) -> np.ndarray:
        """
        Check alpha and prepare it as a np.ndarray

        Parameters
        ----------
        alpha : Iterable[float]
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
            [description]
        """
        flag = 0
        if isinstance(alpha, Iterable):
            for aa in alpha:
                if not isinstance(aa, float) or not 0 < aa < 1:
                    flag += 1
        elif isinstance(alpha, float):
            if not 0 < alpha < 1:
                flag += 1
        else:
            flag += 1
        if flag > 0:
            raise ValueError("Invalid alpha. Allowed values are between 0 and 1.")

        if isinstance(alpha, float):
            alpha = np.stack([alpha])
        elif isinstance(self.alpha, list) or isinstance(self.alpha, tuple):
            alpha = np.stack(alpha)
        return alpha

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
        estimator = self._check_estimator(self.estimator)
        cv = self._check_cv(self.cv)
        X, y = check_X_y(X, y, force_all_finite=False, dtype=["float64", "object"])
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.k_ = np.empty_like(y, dtype=int)
        self.single_estimator_ = clone(estimator).fit(X, y)
        if self.method == "naive":
            y_pred = self.single_estimator_.predict(X)
        else:
            y_pred = np.empty_like(y, dtype=float)
            for k, (train_fold, val_fold) in enumerate(cv.split(X)):
                self.k_[val_fold] = k
                e = clone(estimator).fit(X[train_fold], y[train_fold])
                y_pred[val_fold] = e.predict(X[val_fold])
                self.estimators_.append(e)
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
        self.alpha = self._check_alpha(self.alpha)
        if self.method in ["naive", "base"]:
            quantile = np.quantile(self.residuals_, 1 - self.alpha, interpolation="higher")
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
            # transpose to get array of shape (n_test_samples, len(alpha))
            y_pred_low = np.quantile(lower_bounds, self.alpha, axis=1, interpolation="lower").T
            y_pred_up = np.quantile(upper_bounds, 1 - self.alpha, axis=1, interpolation="higher").T
            if self.ensemble:
                y_pred = np.median(y_pred_multi, axis=1)
        # tile y_pred to get same shape as y_pred_low/up
        y_pred = np.tile(y_pred, (self.alpha.shape[0], 1)).T
        return np.stack([y_pred, y_pred_low, y_pred_up], axis=1)
