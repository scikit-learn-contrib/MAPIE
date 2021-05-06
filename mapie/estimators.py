from __future__ import annotations
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.linear_model import LinearRegression

from ._typing import ArrayLike


class MapieRegressor(BaseEstimator, RegressorMixin):  # type: ignore
    """
    Prediction interval with out-of-fold residuals.

    This class implements the jackknife+ method and its variations
    for estimating prediction intervals on single-output data. The
    idea is to evaluate out-of-fold residuals on hold-out validation
    sets and to deduce valid confidence intervals with strong theoretical
    guarantees.

    Parameters
    ----------
    estimator : sklearn.RegressorMixin, optional
        Any scikit-learn regressor, by default None.

    alpha: float, optional
        Between 0 and 1, represent the uncertainty of the confidence interval.
        Lower alpha produce larger (more conservative) prediction intervals.
        alpha is the complement of the target coverage level.
        Only used at prediction time. By default 0.1.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:
        - "naive"
        - "jackknife"
        - "jackknife_plus"
        - "jackknife_minmax"
        - "cv"
        - "cv_plus"
        - "cv_minmax"

        By default "cv_plus".

    n_splits: int, optional
        Number of splits for cross-validation, by default 5.

    shuffle: bool, optional
        Whether to shuffle the data before splitting into batches, by default True.

    ensemble: bool, optional
        Determines how to return the predictions.
        If False, returns the predictions from the single estimator trained on the full training dataset.
        If True, returns the median of the prediction intervals computed from the out-of-folds models.
        The Jackknife+ interval can be interpreted as an interval around the median prediction,
        and is guaranteed to lie inside the interval, unlike the single estimator predictions.

        By default `False`.

    random_state : int, optional
        Control randomness of cross-validation if relevant, by default None.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    valid_return_preds: List[str]
        List of all valid return_pred values..

    single_estimator_ : sklearn.RegressorMixin
        Estimator fit on the whole training set.

    estimators_ : list
        List of leave-one-out estimators.

    residuals_ : np.ndarray of shape (n_samples_train,)
        Residuals between y_train and y_pred.

    k_: np.ndarray of shape(n_samples_train,)
        Id of the fold containing each trainig sample.

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
    >>> pireg = MapieRegressor(LinearRegression(), method="jackknife_plus")
    >>> print(pireg.fit(X_toy, y_toy).predict(X_toy))
    [[ 5.28571429  4.61627907  6.2       ]
     [ 7.17142857  6.51744186  8.        ]
     [ 9.05714286  8.4         9.8       ]
     [10.94285714 10.2        11.6       ]
     [12.82857143 12.         13.48255814]
     [14.71428571 13.8        15.38372093]]
    """

    valid_methods = [
        "naive",
        "jackknife",
        "jackknife_plus",
        "jackknife_minmax",
        "cv",
        "cv_plus",
        "cv_minmax"
    ]

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        alpha: float = 0.1,
        method: str = "cv_plus",
        n_splits: int = 5,
        shuffle: bool = True,
        ensemble: bool = False,
        random_state: Optional[int] = None
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.method = method
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.ensemble = ensemble
        self.random_state = random_state

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters and estimator.
        """
        if not 0 < self.alpha < 1:
            raise ValueError("Invalid alpha. Please choose an alpha value between 0 and 1.")
        if self.method not in self.valid_methods:
            raise ValueError("Invalid method.")
        if not isinstance(self.ensemble, bool):
            raise ValueError("Invalid ensemble argument. Must be a boolean.")
        if self.estimator is None:
            self.estimator = LinearRegression()

    def _select_cv(self) -> Union[KFold, LeaveOneOut]:
        """
        Define the object that splits the dataset into training
        and validation folds depending on the method:
            - LeaveOneOut for jackknife methods
            - KFold for CV methods

        Returns
        -------
        Union[KFold, LeaveOneOut]
            Either a KFold or a LeaveOneOut object.
        """
        if self.method.startswith("cv"):
            if not self.shuffle:
                self.random_state = None
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        elif self.method.startswith("jackknife"):
            cv = LeaveOneOut()
        else:
            raise ValueError("Invalid method.")
        return cv

    def fit(self, X: ArrayLike, y: ArrayLike) -> MapieRegressor:
        """
        Fit estimator and compute residuals used for prediction intervals.
        Fit the base estimator under the single_estimator_ attribute.
        Fit all cross-validated estimator clones and rearrange them into a list, the estimators_ attribute.
        Out-of-fold residuals are stored under the residuals_ attribute.

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
        X, y = check_X_y(X, y, force_all_finite=False, dtype=["float64", "object"])
        self.single_estimator_ = clone(self.estimator).fit(X, y)
        if self.method == "naive":
            y_pred = self.single_estimator_.predict(X)
        else:
            cv = self._select_cv()
            self.estimators_ = []
            y_pred = np.empty_like(y, dtype=float)
            self.k_ = np.empty_like(y, dtype=int)
            for k, (train_fold, val_fold) in enumerate(cv.split(X)):
                self.k_[val_fold] = k
                e = clone(self.estimator).fit(X[train_fold], y[train_fold])
                self.estimators_.append(e)
                y_pred[val_fold] = e.predict(X[val_fold])
        self.residuals_ = np.abs(y - y_pred)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation. Prediction Intervals for a given alpha are deduced from either
        - quantiles of residuals (naive, jackknife, cv)
        - quantiles of (predictions +/- residuals) (jackknife_plus, cv_plus)
        - quantiles of (max/min(predictions) +/- residuals) (jackknife_minmax, cv_minmax)

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        np.ndarray of shape (n_samples, 3)
        [0]: Center of the prediction interval
        [1]: Lower bound of the prediction interval
        [2]: Upper bound of the prediction interval
        """
        check_is_fitted(self, ["single_estimator_"])
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)
        if self.method in ["naive", "jackknife", "cv"]:
            quantile = np.quantile(self.residuals_, 1 - self.alpha, interpolation="higher")
            y_pred_low = y_pred - quantile
            y_pred_up = y_pred + quantile
        else:
            y_pred_multi = np.stack([e.predict(X) for e in self.estimators_], axis=1)
            if self.ensemble:
                y_pred = np.median(y_pred_multi, axis=1)
            if self.method == "cv_plus":
                y_pred_multi = y_pred_multi[:, self.k_]
            if self.method.endswith("plus"):
                lower_bounds = y_pred_multi - self.residuals_
                upper_bounds = y_pred_multi + self.residuals_
            elif self.method.endswith("minmax"):
                lower_bounds = np.min(y_pred_multi, axis=1, keepdims=True) - self.residuals_
                upper_bounds = np.max(y_pred_multi, axis=1, keepdims=True) + self.residuals_
            else:
                raise ValueError("Invalid method.")
            y_pred_low = np.quantile(lower_bounds, self.alpha, axis=1, interpolation="lower")
            y_pred_up = np.quantile(upper_bounds, 1 - self.alpha, axis=1, interpolation="higher")
        return np.stack([y_pred, y_pred_low, y_pred_up], axis=1)
