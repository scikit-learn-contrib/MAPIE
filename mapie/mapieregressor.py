from __future__ import annotations
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, LeaveOneOut


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
    for estimating prediction intervals from leave-one-out models.

    Parameters
    ----------
    estimator : sklearn.RegressorMixin, optional
        Any scikit-learn regressor, by default None.
    alpha: float, optional
        1 - (target coverage level), by default 0.1.
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
        By default, returns "jackknife_plus" method.
    n_splits: int, optional
        Number of splits for cross-validation, by default 10.
    shuffle: bool, default=True
        Whether to shuffle the data before splitting into batches.
    return_pred: str, optional
        Return the predictions from either
            - the single estimator trained on the full training
            dataset ("single")
            - the median of the prediction intervals computed from
            the leave-one-out or out-of-folds models ("ensemble")
        Valid for the jackknife_plus, jackknife_minmax, cv_plus, or cv_minmax methods.
        By  default, returns "single"
    random_state : int, optional

    Attributes
    ----------
    single_estimator_ : sklearn.RegressorMixin
        Estimator fit on the whole training set.
    estimators_ : list
        List of leave-one-out estimators.
    y_train_pred_split_ : np.ndarray of shape (n_samples,) or (n_splits,)
        Training label predictions by leave-one-out or out-of-fold estimators.
    quantile_: float
        Quantile of the naive, jackknife, or CV residuals.
    residuals_split_ : np.ndarray of shape (n_samples,)
        Residuals between y_train and y_train_pred_split_.
    val_fold_ids_: np.ndarray of shape(n_samples,)
        Attributes the corresponding out-of-folds model to each training point.

    Sources
    -------
    Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani.
    Predictive inference with the jackknife+. Ann. Statist., 49(1):486–507, 022021

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.mapieregressor import MapieRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    >>> y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> pireg = MapieRegressor(LinearRegression())
    >>> print(pireg.fit(X_toy, y_toy).predict(X_toy))
    [[ 5.28571429  4.61627907  6.2       ]
     [ 7.17142857  6.51744186  8.        ]
     [ 9.05714286  8.4         9.8       ]
     [10.94285714 10.2        11.6       ]
     [12.82857143 12.         13.48255814]
     [14.71428571 13.8        15.38372093]]
    """

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        alpha: float = 0.1,
        method: str = "jackknife_plus",
        n_splits: int = 10,
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
        valid_methods = [
            "naive", "jackknife", "jackknife_plus", "jackknife_minmax", "cv", "cv_plus", "cv_minmax"
        ]
        if self.method not in valid_methods:
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
        Either a KFold or a LeaveOneOut object.
        """
        if self.method.startswith("cv"):
            if not self.shuffle:
                self.random_state = None
            cv = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
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
        X, y = check_X_y(
            X, y, force_all_finite=False, dtype=["float64", "object"]
        )
        self.single_estimator_ = clone(self.estimator)
        self.single_estimator_.fit(X, y)
        if self.method == "naive":
            y_train_pred = self.single_estimator_.predict(X)
            residuals = np.abs(y_train_pred - y)
            self.quantile_ = np.quantile(
                residuals, 1 - self.alpha, interpolation="higher"
            )
        else:
            cv = self._select_cv()
            self.estimators_ = []
            self.y_train_pred_split_ = np.array([], dtype=float)
            if self.method.startswith("cv"):
                self.val_fold_ids_ = np.array([], dtype=int)
                index_cv = np.array([], dtype=int)
            for val_fold_id, (train_fold, val_fold) in enumerate(cv.split(X)):
                if self.method.startswith("cv"):
                    self.val_fold_ids_ = np.concatenate(
                        (
                            self.val_fold_ids_,
                            np.full(len(val_fold), val_fold_id)
                        )
                    )
                    index_cv = np.concatenate((index_cv, val_fold))
                e = clone(self.estimator)
                e.fit(X[train_fold], y[train_fold])  # type: ignore
                self.estimators_.append(e)
                self.y_train_pred_split_ = np.concatenate(
                    (
                        self.y_train_pred_split_,
                        e.predict(X[val_fold])  # type: ignore
                    )
                )
            if self.method.startswith("cv"):
                order = np.argsort(index_cv)
                self.val_fold_ids_ = self.val_fold_ids_[order]
                self.y_train_pred_split_ = self.y_train_pred_split_[order]
            self.residuals_split_ = np.abs(self.y_train_pred_split_ - y)
            if self.method in ["cv", "jackknife"]:
                self.quantile_ = np.quantile(
                    self.residuals_split_,
                    1 - self.alpha,
                    interpolation="higher"
                )
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
        y_test_pred = self.single_estimator_.predict(X)
        if self.method in ["naive", "jackknife", "cv"]:
            y_test_pred_low = y_test_pred - self.quantile_
            y_test_pred_up = y_test_pred + self.quantile_
        else:
            y_test_pred_split = np.stack(
                [e.predict(X) for e in self.estimators_], axis=1
            )
            if self.return_pred == "ensemble":
                y_test_pred = np.median(y_test_pred_split, axis=1)
            if self.method == "cv_plus":
                y_test_pred_split = y_test_pred_split[:, self.val_fold_ids_]
            if self.method.endswith("plus"):
                lower_bounds = y_test_pred_split - self.residuals_split_
                upper_bounds = y_test_pred_split + self.residuals_split_
            elif self.method.endswith("minmax"):
                lower_bounds = np.min(
                    y_test_pred_split, axis=1, keepdims=True
                ) - self.residuals_split_
                upper_bounds = np.max(
                    y_test_pred_split, axis=1, keepdims=True
                ) + self.residuals_split_
            else:
                raise ValueError("Invalid method.")
            y_test_pred_low = np.quantile(
                lower_bounds, self.alpha, axis=1, interpolation="lower"
            )
            y_test_pred_up = np.quantile(
                upper_bounds, 1 - self.alpha, axis=1, interpolation="higher"
            )
        return np.stack([y_test_pred, y_test_pred_low, y_test_pred_up], axis=1)
