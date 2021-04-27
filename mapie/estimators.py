from __future__ import annotations
from typing import Optional, Union

import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, LeaveOneOut

from ._typing import ArrayLike


def check_not_none(estimator: Optional[RegressorMixin]) -> None:
    """
    Check that estimator is not None.

    Parameters
    ----------
    estimator : RegressorMixin
        Any scikit-learn regressor.

    Raises
    ------
    ValueError
        If the estimator is None.
    """
    if estimator is None:
        raise ValueError("Invalid none estimator.")


class MapieRegressor(BaseEstimator, RegressorMixin):  # type: ignore
    """
    Estimator implementing the jackknife+ method and its variations
    for estimating prediction intervals from leave-one-out regressors on
    single-output data.

    Parameters
    ----------
    estimator : sklearn.RegressorMixin, optional
        Any scikit-learn regressor, by default None.

    alpha: float, optional
        1 - (target coverage level), by default 0.1.

    method: str, optional
        Method to choose for prediction interval estimates.
        By default, "cv_plus".
        Choose among:
        - "naive"
        - "jackknife"
        - "jackknife_plus"
        - "jackknife_minmax"
        - "cv"
        - "cv_plus"
        - "cv_minmax"

    n_splits: int, optional
        Number of splits for cross-validation, by default 5.

    shuffle: bool, default=True
        Whether to shuffle the data before splitting into batches.

    return_pred: str, optional
        Return the predictions from either
        - the single estimator trained on the full training dataset ("single")
        - the median of the prediction intervals computed from the leave-one-out or out-of-folds models ("ensemble")

        Valid for the jackknife_plus, jackknife_minmax, cv_plus, or cv_minmax methods.
        By default, returns "single".

    random_state : int, optional
        Control randomness of cross-validation if relevant.

    Attributes
    ----------
    valid_methods_: List[str]
        List of all valid methods.

    single_estimator_ : sklearn.RegressorMixin
        Estimator fit on the whole training set.

    estimators_ : list
        List of leave-one-out estimators.

    quantile_: float
        Quantile of the naive, jackknife, or CV residuals.

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

    valid_methods_ = [
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
        return_pred: str = "single",
        random_state: Optional[int] = None
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.method = method
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.return_pred = return_pred
        self.random_state = random_state

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters and estimator
        """
        if not 0 < self.alpha < 1:
            raise ValueError("Invalid alpha. Please choose an alpha value between 0 and <1.")
        if self.method not in self.valid_methods_:
            raise ValueError("Invalid method.")
        check_not_none(self.estimator)

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
        Fit all jackknife clones and rearrange them into a list.
        The initial estimator is fit apart.

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
        self.single_estimator_ = clone(self.estimator)
        self.single_estimator_.fit(X, y)
        if self.method == "naive":
            y_pred = self.single_estimator_.predict(X)
            residuals = np.abs(y - y_pred)
            self.quantile_ = np.quantile(residuals, 1 - self.alpha, interpolation="higher")
        else:
            cv = self._select_cv()
            n_samples = len(y)
            self.estimators_ = []
            y_pred = np.empty(n_samples, dtype=float)
            if self.method.startswith("cv"):
                self.k_ = np.empty(n_samples, dtype=int)
            for k, (train_fold, val_fold) in enumerate(cv.split(X)):
                if self.method.startswith("cv"):
                    self.k_[val_fold] = k
                e = clone(self.estimator)
                e.fit(X[train_fold], y[train_fold])
                self.estimators_.append(e)
                y_pred[val_fold] = e.predict(X[val_fold])
            self.residuals_ = np.abs(y - y_pred)
            if self.method in ["cv", "jackknife"]:
                self.quantile_ = np.quantile(self.residuals_, 1 - self.alpha, interpolation="higher")
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Compute Prediction Intervals (PIs) for a given alpha from the trained
        leave-one-out or out-of-fold models using the chosen method.

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
            y_pred_low = y_pred - self.quantile_
            y_pred_up = y_pred + self.quantile_
        else:
            y_pred_multi = np.stack([e.predict(X) for e in self.estimators_], axis=1)
            if self.return_pred == "ensemble":
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
