from abs import ABC, abstractmethod
from inspect import signature
from typing import cast, Union, Optional, Self, Tuple
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.base import RegressorMixin, ClassifierMixin, clone
from warnings import warn

import numpy as np


# TODO: add checkings from _MapieRegressor
from mapie.utils import (
    _raise_error_if_fit_called_in_prefit_mode,
    _raise_error_if_method_already_called,
)

Estimator = Union[RegressorMixin, ClassifierMixin]


# Base Mixin for fitting behavior
class _FitterMixin:
    def _fit_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        **fit_params,
    ) -> Estimator:
        """
        Fit an estimator on training data by distinguishing two cases:
        - the estimator supports sample weights and sample weights are provided.
        - the estimator does not support samples weights or
          samples weights are not provided.

        Parameters
        ----------

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples,)
            Input labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default None.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        RegressorMixin
            Fitted estimator.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> from mapie.utils import check_sklearn_user_model_is_fitted
        >>> X = np.array([[0], [1], [2], [3], [4], [5]])
        >>> y = np.array([5, 7, 9, 11, 13, 15])
        >>> estimator = LinearRegression()
        >>> estimator = _fit_estimator(estimator, X, y)
        >>> check_sklearn_user_model_is_fitted(estimator)
        True
        """
        estimator = clone(self.__base_estimator)
        fit_parameters = signature(estimator.fit).parameters
        supports_sw = "sample_weight" in fit_parameters
        if supports_sw and sample_weight is not None:
            estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
        elif sample_weight is not None:
            warn("Sample weight couldn't be passed to model for fitting.")
        else:
            estimator.fit(X, y, **fit_params)
        return estimator

    @property
    def is_fitted(self):
        """Returns True if the estimator is fitted"""
        return self._is_fitted


class _RegressorFitterMixin(_FitterMixin):
    estimator_type = RegressorMixin

    def _estimator_predict(self, X: ArrayLike, **predict_params):
        return self._estimator_.predict(X, **predict_params)


class _ClassifierFitterMixin(_FitterMixin):
    estimator_type = ClassifierMixin

    def _fix_number_of_classes(
        self, n_classes_training: NDArray, y_proba: NDArray
    ) -> NDArray:
        """
        Fix shape of y_proba of validation set if number of classes
        of the training set used for cross-validation is different than
        number of classes of the original dataset y.

        Parameters
        ----------
        n_classes_training: NDArray
            Classes of the training set.
        y_proba: NDArray
            Probabilities of the validation set.

        Returns
        -------
        NDArray
            Probabilities with the right number of classes.
        """
        y_pred_full = np.zeros(shape=(len(y_proba), self.n_classes))
        y_index = np.tile(n_classes_training, (len(y_proba), 1))
        np.put_along_axis(y_pred_full, y_index, y_proba, axis=1)
        return y_pred_full

    def _estimator_predict(self, X: ArrayLike, **predict_params):
        """
        Predict probabilities of a test set from a fitted estimator.

        Parameters
        ----------
        X: ArrayLike
            Test set.

        Returns
        -------
        ArrayLike
            Predicted probabilities.
        """
        y_pred = self.estimator_.predict_proba(X, **predict_params)
        if len(self.estimator_.classes_) != self.n_classes:
            y_pred = self._fix_number_of_classes(self.estimator_.classes_, y_pred)
        return y_pred


class _Conformalizer(ABC):
    @abstractmethod
    def _fit_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        **fit_params,
    ) -> Estimator:
        pass

    @abstractmethod
    def _estimator_predict(self, X: ArrayLike, **predict_params):
        pass

    def _safe_predict_oof(self, X: ArrayLike, **predict_param):
        if _num_samples(X) < 0:
            return np.array([])
        return self._estimator_predict(X, **predict_param)

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ):
        pass

    @abstractmethod
    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ):
        pass


class _SplitConformalizer(ABC, _Conformalizer):
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> Self:
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)
        if not self.prefit:
            # TODO back-end: avoid using private utilities from sklearn like
            # _safe_indexing (may break anytime without notice)
            train_index, _ = self.cv.split(X, y, groups)
            X_train = _safe_indexing(X, train_index)
            y_train = _safe_indexing(y, train_index)
            if sample_weight is not None:
                sample_weight = _safe_indexing(sample_weight, train_index)
                sample_weight = cast(NDArray, sample_weight)
            self.estimator = self._fit_estimator(
                X_train, y_train, sample_weight=sample_weight, **fit_params
            )
        self._is_fitted = True
        return self

        def _get_val_samples(
            self, X: ArrayLike, y: ArrayLike, groups: Optional[ArrayLike] = None
        ) -> Tuple[ArrayLike, ArrayLike]:
            X_val = X
            y_val = y
            if not self.prefit:
                _, val_indices = self.cv.split(X, y, groups)
                X_val = _safe_indexing(X, val_indices)
                y_val = _safe_indexing(y, val_indices)

            return X_val, y_val

        def conformalize(
            self,
            X: ArrayLike,
            y: ArrayLike,
            groups: Optional[ArrayLike] = None,
            **predict_params,
        ) -> Self:
            X_val, y_val = self._get_val_samples(X, y, groups)
            y_pred = self._safe_predict_oof(X_val, **predict_params)

            self.conformity_scores_ = self.conformity_score.get_conformity_scores(
                y_val, y_pred, X=X
            )
            return self


class _CrossConformalizer(ABC, _Conformalizer):
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> Self:
        self.k_ = np.full(
            shape=(_num_samples(X), self.cv.get_n_splits(X, y, groups)),
            fill_value=np.nan,
            dtype=float,
        )

        self.estimators_ = Parallel(self.n_jobs, verbose=self.verbose)(
            delayed(self._fit_estimator)(
                _safe_indexing(X, train_index),
                _safe_indexing(y, train_index),
                sample_weight,
                **fit_params,
            )
            for train_index, _ in self.cv.split(X, y, groups)
        )
        self._is_fitted = True
        return self
