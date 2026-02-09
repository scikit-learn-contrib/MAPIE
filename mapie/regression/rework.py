from abs import ABC, abstractmethod
from sklearn.utils import _safe_indexing
from sklearn.base import RegressorMixin, ClassifierMixin
from warnings import warn


# Base Mixin for fitting behavior
class _FitterMixin:
    def _fit_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        **fit_params,
    ) -> Union[RegressorMixin, ClassifierMixin]:
        """
        Fit an estimator on training data by distinguishing two cases:
        - the estimator supports sample weights and sample weights are provided.
        - the estimator does not support samples weights or
          samples weights are not provided.

        Parameters
        ----------
        estimator: Union[RegressorMixin, ClassifierMixin]
            Estimator to train.

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
        fit_parameters = signature(self.estimator.fit).parameters
        supports_sw = "sample_weight" in fit_parameters
        if supports_sw and sample_weight is not None:
            self.estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
        elif sample_weight is not None:
            warn("Sample weight couldn't be passed to model for fitting.")
        else:
            self.estimator.fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    @property
    def is_fitted(self):
        """Returns True if the estimator is fitted"""
        return self._is_fitted


class _RegressorFitterMixin(_FitterMixin):
    estimator_type = RegressorMixin


class _SplitConformalizer(ABC):
    @abstractmethod
    def _fit_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        **fit_params,
    ) -> Union[RegressorMixin, ClassifierMixin]:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> _SplitConformalizer:
        if not self.prefit:
            # TODO back-end: avoid using private utilities from sklearn like
            # _safe_indexing (may break anytime without notice)
            train_index, _ = self.cv.split(X, y, groups)
            X_train = _safe_indexing(X, tran_index)
            y_train = _safe_indexing(y, train_index)
            if sample_weight is not None:
                sample_weight = _safe_indexing(sample_weight, train_index)
                sample_weight = cast(NDArray, sample_weight)
            self._fit_estimator(
                X_train, y_train, sample_weight=sample_weight, **fit_params
            )


class _CrossConformalizerMixin(ABC):
    pass
