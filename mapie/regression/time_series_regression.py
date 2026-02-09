from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union, cast
from warnings import warn
from abs import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from mapie.conformity_scores import BaseRegressionScore
from mapie.regression.regression import _MapieRegressor
from mapie.utils import (
    _check_alpha,
    _transform_confidence_level_to_alpha_list,
    check_is_fitted,
)


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
            train_index, _ = self.cv.split(X, y, groups)

            self._fit_estimator(X[train_index])


class _CrossConformalizerMixin(ABC):
    pass


class TimeSeriesRegressor(_MapieRegressor):
    """
    Prediction intervals with out-of-fold residuals for time series.
    This class only has two valid ``method`` : ``"enbpi"`` or ``"aci"``

    The prediction intervals are calibrated on a split of the trained data.
    Both strategies are estimating prediction intervals
    on single-output time series.

    EnbPI allows you to update conformal scores using the ``update``
    function. It will replace the oldest one with the newest scores.
    It will keep the same amount of total scores

    Actually, EnbPI only corresponds to ``TimeSeriesRegressor`` if the
    ``cv`` argument is of type ``BlockBootstrap``.

    The ACI strategy allows you to adapt the conformal inference
    (i.e the quantile). If the real values are not in the coverage,
    the size of the intervals will grow. Conversely, if the real values are in
    the coverage, the size of the intervals will decrease. You can use a gamma
    coefficient to adjust the strength of the correction. If the quantile is
    equal to zero, the method will produce an infinite set size.

    References
    ----------
    Chen Xu, and Yao Xie.
    "Conformal prediction for dynamic time-series."
    https://arxiv.org/abs/2010.09107

    Isaac Gibbs, Emmanuel Candes
    "Adaptive conformal inference under distribution shift"
    https://proceedings.neurips.cc/paper/2021/file/\
0d441de75945e5acbc865406fc9a2559-Paper.pdf

    Margaux Zaffran et al.
    "Adaptive Conformal Predictions for Time Series"
    https://arxiv.org/pdf/2202.07282.pdf
    """

    cv_need_agg_function_ = _MapieRegressor.cv_need_agg_function_ + ["BlockBootstrap"]
    valid_methods_ = ["enbpi", "aci"]
    default_sym_ = False

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "enbpi",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[BaseRegressionScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
            conformity_score=conformity_score,
            random_state=random_state,
        )

    def _relative_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
    ) -> NDArray:
        """
        Compute the conformity scores on a data set.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
                Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.
            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        Returns
        -------
            The conformity scores corresponding to the input data set.
        """
        y_pred = super().predict(X, ensemble=ensemble)
        scores = np.array(
            self.conformity_score_function_.get_conformity_scores(y, y_pred, X=X)
        )
        return scores

    def _update_conformity_scores_with_ensemble(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
    ) -> TimeSeriesRegressor:
        """
        Update the ``conformity_scores_`` attribute when new data with known
        labels are available.
        Note: Don't use ``_update_conformity_scores_with_ensemble`` with samples of the training set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.
            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        Returns
        -------
        TimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of ``y`` is greater than
            the length of the training set.
        """
        check_is_fitted(self)
        X, y = cast(NDArray, X), cast(NDArray, y)
        m, n = len(X), len(self.conformity_scores_)
        if m > n:
            raise ValueError(
                "The number of observations to update is higher than the"
                "number of training instances."
            )
        new_conformity_scores_ = self._relative_conformity_scores(
            X, y, ensemble=ensemble
        )
        self.conformity_scores_ = np.roll(
            self.conformity_scores_, -len(new_conformity_scores_)
        )
        self.conformity_scores_[-len(new_conformity_scores_) :] = new_conformity_scores_
        return self

    @staticmethod
    def _check_gamma(gamma: float) -> None:
        """
        Check if gamma is between 0 and 1.

        Parameters
        ----------
        gamma: float

        Raises
        ------
        ValueError
            If gamma is lower than 0 or higher than 1.
        """
        if (gamma < 0) or (gamma > 1):
            raise ValueError("Invalid gamma. Allowed values are between 0 and 1.")

    def _get_alpha(
        self, alpha: Optional[Union[float, Iterable[float]]] = None, reset: bool = False
    ) -> Optional[NDArray]:
        """
        Get and set the current alpha (or confidence_level) value(s) given the
        initial alpha (or confidence_level) value(s) for ACI method.

        This method retrieves the alpha value(s) used for confidence intervals.
        If the alpha value(s) is provided, it returns the current alpha
        value(s) stored in the object. Else, nothing. If the reset flag is set
        to True, it resets the current alpha value(s).

        Parameters
        ----------
        alpha: Optional[NDArray]
            Between ``0`` and ``1``, represents the uncertainty of the
            confidence interval.

            By default ``None``.

        reset: bool
            Flag indicating whether to reset the current alpha value(s).

        Returns
        -------
        Optional[Union[float, Iterable[float]]]
            The current alpha value(s) for confidence intervals.
        """
        if "current_alpha" not in self.__dict__ or reset:
            self.current_alpha: dict[float, float] = {}

        if alpha is not None:
            alpha_np = cast(NDArray, _check_alpha(alpha))
            alpha_np = np.round(alpha_np, 2)
            for ix, alpha_checked in enumerate(alpha_np):
                alpha_np[ix] = self.current_alpha.setdefault(
                    alpha_checked, alpha_checked
                )
            alpha = alpha_np
        return alpha

    def adapt_conformal_inference(
        self,
        X: ArrayLike,
        y: ArrayLike,
        gamma: float,
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
        ensemble: bool = False,
        optimize_beta: bool = False,
    ) -> TimeSeriesRegressor:
        """
        Adapt the ``alpha_t`` attribute when new data with known
        labels are available.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.
            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        gamma: float
            Coefficient that decides the correction of the conformal inference.
            If it equals 0, there are no corrections.

        confidence_level: Optional[Union[float, Iterable[float]]]
            Between ``0`` and ``1``, represents the confidence level of the interval.

            By default ``None``.

        optimize_beta: bool
            Whether to optimize the PIs' width or not.

            By default ``False``.

        Returns
        -------
        TimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of ``y`` is greater than
            the length of the training set.
        """
        if self.method != "aci":
            raise AttributeError(
                "This method can be called only with method='aci', "
                f"not with '{self.method}'."
            )

        check_is_fitted(self)
        self._check_gamma(gamma)
        X, y = cast(NDArray, X), cast(NDArray, y)

        self._get_alpha()
        alpha = self._transform_confidence_level_to_alpha_array(confidence_level)
        if alpha is None:
            alpha = np.array(list(self.current_alpha.keys()))
        alpha_np = cast(NDArray, alpha)

        for x_row, y_row in zip(X, y):
            x = np.expand_dims(x_row, axis=0)
            _, y_pred_bounds = self.predict(
                x,
                ensemble=ensemble,
                confidence_level=1 - alpha_np,
                optimize_beta=optimize_beta,
                allow_infinite_bounds=True,
            )

            for alpha_ix, alpha_0 in enumerate(alpha_np):
                alpha_t = self.current_alpha[alpha_0]

                is_lower_bounded = y_row > y_pred_bounds[:, 0, alpha_ix]
                is_upper_bounded = y_row < y_pred_bounds[:, 1, alpha_ix]
                is_not_bounded = not (is_lower_bounded and is_upper_bounded)

                new_alpha_t = alpha_t + gamma * (alpha_0 - is_not_bounded)
                self.current_alpha[alpha_0] = np.clip(new_alpha_t, 0, 1)

        return self

    def update(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
        gamma: float = 0.0,
        optimize_beta: bool = False,
    ) -> TimeSeriesRegressor:
        """
        Update conformity scores

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.
            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        confidence_level: Optional[Union[float, Iterable[float]]]
            (deprecated)
            Between ``0`` and ``1``, represents the confidence level of the interval.

            By default ``None``.

        gamma: float
            (deprecated)
            Coefficient that decides the correction of the conformal inference.
            If it equals 0, there are no corrections.

            By default ``0.``.

        optimize_beta: bool
            (deprecated)
            Whether to optimize the PIs' width or not.

            By default ``False``.

        Returns
        -------
        TimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of ``y`` is greater than
            the length of the training set.
        """
        warn("""
        This function behavior has been changed to allow updating the scores even when using ACI.
        Currently the parameters confidence_level and optimize_beta have no effect. They are kept
        for API stability and will be removed in a future release.
        If you want to adapt confidence level, use adapt_conformal_inference instead.
        """)
        self._check_method(self.method)
        if self.method not in ["enbpi", "aci"]:
            raise ValueError(
                f"Invalid method. Allowed values are {self.valid_methods_}."
            )

        return self._update_conformity_scores_with_ensemble(X, y, ensemble=ensemble)

    # Overriding _MapieRegressor .predict method here. Bad practise, but this
    # inheritance is questionable and will probably be reconsidered anyway.
    def predict(  # type: ignore[override]
        self,
        X: ArrayLike,
        ensemble: bool = False,
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        **predict_params,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.
            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        confidence_level: Optional[Union[float, Iterable[float]]]
            Between ``0`` and ``1``, represents the confidence level of the interval.

            By default ``None``.

        optimize_beta: bool
            Whether to optimize the PIs' width or not.

            By default ``False``.

        allow_infinite_bounds: bool
            Allow infinite prediction intervals to be produced.

        predict_params : dict
            Additional predict parameters.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if ``alpha`` is ``None``.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if ``alpha`` is not ``None``.
              - [:, 0, :]: Lower bound of the prediction interval.
              - [:, 1, :]: Upper bound of the prediction interval.
        """
        alpha = self._transform_confidence_level_to_alpha_array(confidence_level)
        if alpha is None:
            super().predict(
                X,
                ensemble=ensemble,
                alpha=alpha,
                optimize_beta=optimize_beta,
                **predict_params,
            )
        if self.method == "aci":
            alpha = self._get_alpha(alpha)

        return super().predict(
            X,
            ensemble=ensemble,
            alpha=alpha,
            optimize_beta=optimize_beta,
            allow_infinite_bounds=allow_infinite_bounds,
            **predict_params,
        )

    # The public API changed from alpha to confidence_level.
    # TODO: refactor this class to use confidence_level everywhere
    @staticmethod
    def _transform_confidence_level_to_alpha_array(
        confidence_level: Optional[Union[float, Iterable[float]]] = None,
    ) -> Optional[NDArray]:
        confidence_level = cast(Optional[NDArray], _check_alpha(confidence_level))
        if confidence_level is None:
            alpha = None
        else:
            alpha = np.array(
                _transform_confidence_level_to_alpha_list(confidence_level)
            )
        return alpha
