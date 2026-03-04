from abs import ABC, abstractmethod
from inspect import signature
from typing import cast, Union, Optional, Self, Tuple, Callable
from joblib import Parallel, delayed
from functools import wraps
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.preprocessing import LabelEncoder
from sklearn.base import RegressorMixin, ClassifierMixin, clone
from warnings import warn

from mapie.aggregation_functions import aggregate_all
import numpy as np


# TODO: add checkings from _MapieRegressor
from mapie.utils import (
    _raise_error_if_fit_called_in_prefit_mode,
    _raise_error_if_method_already_called,
    check_is_fitted,
    _check_nan_in_aposteriori_prediction,
)

Estimator = Union[RegressorMixin, ClassifierMixin]


# Base Mixin for fitting behavior
# In constructor estimator_ is set from __base_estimator if cv is prefit
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
        return self.estimator_.predict(X, **predict_params)

    def _process_cross_conformal(
        self, preds: ArrayLike, indices: ArrayLike, n_samples: int, n_splits: int
    ) -> ArrayLike:
        pred_matrix = np.full(
            shape=(n_samples, n_splits), fill_value=np.nan, dtype=float
        )

        col = 0
        for ind in indices:
            pred_matrix[ind, col] = np.array(preds[col], dtype=float)
            col += 1

        _check_nan_in_aposteriori_prediction(pred_matrix)
        y_pred = aggregate_all(self.agg_function, pred_matrix)

        return y_pred

    def _aggregate(self, preds):
        if self.aggregation == "median":
            return phi2D(A=preds, B=self.k, fun=lambda: np.nanmedian(preds, axis=1))
        if self.aggregation == "mean":
            K = np.nan_to_num(
                self.k,
            )
            return np.matmul(preds, (K / K.sum(axis=1, keepdims=True)).T)

    def predict_intervalle(self, X: ArrayLike, **predict_params) -> ArrayLike:
        return self.conformity_score.predict_set(
            X,
            self.alpha,
            self.conformity_score,
            ensemble=True,
            method=self.method,
            optimize_beta=self.optimize_beta,
            allowinfinite_bounds=self.allow_infinite_bounds,
        )


class _ClassifierFitterMixin(_FitterMixin):
    estimator_type = ClassifierMixin

    def _set_classes_(self, y) -> Self:
        """
        Extracte the number of classes and the classes values
        from the pre-trained model or the values in y.

        Parameters
        ----------

        y: NDArray
            Values to predict.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            If `cv="prefit"` and that classes in `y` are not included into
            `estimator.classes_`.

        Warning
            If number of calibration labels is lower than number of labels
            for training (in prefit setting)
        """
        n_unique_y_labels = len(np.unique(y))
        if self.is_fitted:
            classes = self.estimator_.classes_
            n_classes = len(np.unique(classes))
            if not set(np.unique(y)).issubset(classes):
                raise ValueError(
                    "Values in y do not matched values in estimator.classes_."
                    + " Check that you are not adding any new label"
                )
            if n_classes > n_unique_y_labels:
                warn(
                    "WARNING: your conformalization dataset has less labels"
                    + " than your training dataset (training"
                    + f" has {n_classes} unique labels while"
                    + f" conformalization have {n_unique_y_labels} unique labels"
                )

        else:
            n_classes = n_unique_y_labels
            classes = np.unique(y)

        self.n_classes = n_classes
        self.classes = classes

    def _set_label_encoder(self) -> Self:
        """
        Construct the label encoder with respect to the classes values.

        Returns
        -------
        LabelEncoder
        """
        self.label_encoder = LabelEncoder().fit(self.classes_)

    # TODO: upgrade typing using ParamSpecs and TypeVar
    # use to decorate fit and conformalize when instanciated with a Conformalizer
    @staticmethod
    def _fit_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, X: ArrayLike, y: ArrayLike, **kwargs):
            self._set_classes(y)
            self._set_label_encoder()
            return func(X, y, **kwargs)

        return wrapper

    # use to decorate fit and conformalize when instanciated with a Conformalizer
    @staticmethod
    def _encode_labels(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, X: ArrayLike, y: ArrayLike, **kwargs):
            y_enc = self.label_encoder.transform(y)
            return func(X, y_enc, **kwargs)

        return wrapper

    def _encode_labels(self, y):
        return self.label_encoder.transform(y)

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

    @staticmethod
    def _check_proba_normalized(y_pred_proba: ArrayLike, axis: int = 1) -> ArrayLike:
        """
        Check if, for all the observations, the sum of
        the probabilities is equal to one.

        Parameters
        ----------
        y_pred_proba: ArrayLike of shape
            (n_samples, n_classes) or (n_samples, n_train_samples, n_classes)
            Softmax output of a model.

        Returns
        -------
        ArrayLike of shape (n_samples, n_classes)
            Softmax output of a model if the scores all sum to one.

        Raises
        ------
        ValueError
            If the sum of the scores is not equal to one.
        """
        np.testing.assert_allclose(
            np.sum(y_pred_proba, axis=axis),
            1,
            err_msg="The sum of the scores is not equal to one.",
            rtol=1e-5,
        )
        return y_pred_proba

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
        self._check_proba_normalized(y_pred)
        return y_pred

    def _process_cross_conformal(
        self, preds: ArrayLike, indices: ArrayLike, n_samples: int, n_splits: int
    ) -> ArrayLike:
        y_pred = np.empty((n_samples, self.n_classes), dtype=float)

        for ind, pred in zip(indices, preds):
            y_pred[ind] = preds

        return y_pred

    # use to decorate predict functions when instanciated with a Conformalizer
    @staticmethod
    def _decode_labels(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, X: ArrayLike, **kwargs):
            y_pred = func(X, **kwargs)
            return self.label_encoder.inverse_transform(y_pred, axis=1)

        return wrapper

    # TODO: rework signature
    def predict_intervalle(self, X: ArrayLike, predict_param) -> ArrayLike:
        y_pred = self._estimator_predict(X, predict_params)

        prediction_sets = self.conforomity_score.predict_set(
            X,
            self.alpha,
            y_pred,
            self.cv,
            conforomity_scores=self.conformity_scores,
            include_last_label=self.include_last_label,
        )

        return y_pred, prediction_sets

    def _aggregate(self, preds: ArrayLike) -> ArrayLike:
        if self.aggregation == "mean":
            return np.mean(preds, axis=0)


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

    @abstractmethod
    def predict(self, X: ArrayLike, **predict_param) -> ArrayLike:
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
            check_is_fitted(self)
            X_val, y_val = self._get_val_samples(X, y, groups)
            y_pred = self._safe_predict_oof(X_val, **predict_params)

            self.conformity_scores_ = self.conformity_score.get_conformity_scores(
                y_val, y_pred, X=X
            )
            return self

        # To be decorated for classifier
        def predict(self, X: ArrayLike, **predict_param) -> ArrayLike:
            return self._estimator_predict(X, **predict_param)


class _CrossConformalizer(ABC, _Conformalizer):
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> Self:
        X, y, sample_weight, groups = self.conformity_scores.split(
            X, y, sample_weight, groups
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

    def _predict_val(self, estimator: Estimator, X: ArrayLike, **predict_params):
        self.estimator_ = estimator
        return self._safe_predict_oof(X, **predict_params)

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **predict_params,
    ) -> Self:
        check_is_fitted(self)
        val_indices = [val_index for (_, val_index) in self.cv.split(X, y, groups)]
        preds = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(
                self._predict_val(
                    estimator,
                    _safe_indexing(X, val_indices, **predict_params),
                    **predict_params,
                )
                for val_index, estimator in zip(val_indices, self.estimators_)
            )
        )

        y_pred = self._process_cross_conformal(
            preds,
            val_indices,
            _num_samples(X),
            self.cv.get_n_splits(),
        )

        self.conformity_scores_ = self.conformity_score.get_conformity_scores(
            y,
            y_pred,
            X=X,
            sample_weight=sample_weight,
            groups=groups,
            predict_params=predict_params,
        )
        return self

    def predict(self, X: ArrayLike, **predict_params) -> ArrayLike:
        preds = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(
                self._predict_val(
                    estimator,
                    X,
                    **predict_params,
                )
                for estimator in self.estimators_
            )
        )
        # TODO : not sure it is necessary
        preds = np.column_stack(preds)

        return self._aggregate(preds)
