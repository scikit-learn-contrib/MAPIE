from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast

import warnings
import numpy as np
from abc import ABC
from functools import lru_cache
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split, BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils.validation import _check_y, _num_samples, indexable

from mapie.utils import (
    _cast_predictions_to_ndarray_tuple,
    _check_alpha_and_n_samples,
    _check_cv,
    _check_cv_not_string,
    _check_cv_not_subsample,
    _check_estimator_fit_predict,
    _check_if_param_in_allowed_values,
    _check_lower_upper_bounds,
    _check_null_weight,
    _fit_estimator,
    _prepare_params,
    _raise_error_if_fit_called_in_prefit_mode,
    _raise_error_if_method_already_called,
    _raise_error_if_previous_method_not_called,
    _transform_confidence_level_to_alpha_list,
    _transform_confidence_level_to_alpha,
    check_is_fitted,
    check_sklearn_user_model_is_fitted,
    _check_nan_in_aposteriori_prediction,
)

from mapie.aggregation_functions import aggregate_all
from .regression import _MapieRegressor
from mapie.estimator.regressor import _Conformalizer
from mapie.conformity_scores import (
    AbsoluteQuantileRegressionScore,
    QuantileRegressionScore,
)

REGRESSOR_TYPE = Union[RegressorMixin, Pipeline]

# Stands in for `typing.Self`, which is only available from Python 3.11 on: it keeps
# `reset` returning the concrete subclass rather than the abstract conformalizer.
_QuantileConformalizerT = TypeVar(
    "_QuantileConformalizerT", bound="_QuantileConformalizer"
)

# Default of `aggregate_point_predictions`: no single value suits every method, since
# `"base"` predicts points without aggregating the folds while `"plus"` and `"minmax"`
# require an aggregation. This sentinel resolves to the one the method supports.
AGGREGATE_POINT_PREDICTIONS_AUTO = "auto"


class _QuantileConformalizer(_Conformalizer, ABC):
    quantile_estimator_params = {
        "GradientBoostingRegressor": {"loss_name": "loss", "alpha_name": "alpha"},
        "QuantileRegressor": {"loss_name": "quantile", "alpha_name": "quantile"},
        "HistGradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "quantile",
        },
        "LGBMRegressor": {"loss_name": "objective", "alpha_name": "alpha"},
    }

    ALLOWED_SCORES = QuantileRegressionScore
    ALLOWED_AGG = ["mean", "median", "pinball_weighted_mean"]

    _central_estimator: Optional[RegressorMixin]
    fit_central_estimator: Optional[bool]
    estimator: RegressorMixin
    score: QuantileRegressionScore

    # One entry per confidence level, keyed by `str(alpha)`.
    alpha: List[float]
    quantiles: dict[str, NDArray[np.float64]]
    conformity_scores: dict[str, NDArray[np.float64]]
    pinball_losses: dict[str, ArrayLike]

    # The lower and upper estimators are fitted per confidence level, whereas the
    # central one is shared across levels and therefore stored apart.
    estimators_: dict[str, dict[str, List[RegressorMixin]]]
    central_estimators_: List[RegressorMixin]
    _base_estimator_: dict[str, dict[str, List[RegressorMixin]]]
    _base_central_estimator_: List[RegressorMixin]

    key_mapping: dict[str, int]
    _predict_params: dict
    __central_fitted: bool = False

    def __init__(self) -> None:
        # to run tests
        self.is_fitted = False
        self.is_conformalized = False

    def _check_alpha(
        self,
        alpha: float = 0.1,
    ) -> NDArray:
        """
        Perform several checks on the alpha value and changes it from
        a float to an ArrayLike.

        Parameters
        ----------
        alpha : float
            Can only be a float value between `0.0` and `1.0`.
            Represent the risk level of the confidence interval.
            Lower alpha produce larger (more conservative) prediction
            intervals. Alpha is the complement of the target coverage level.

            By default `0.1`.

        Returns
        -------
        ArrayLike
            An ArrayLike of three values:

            - [0]: alpha value of alpha/2
            - [1]: alpha value of of 1 - alpha/2
            - [2]: alpha value of 0.5

        Raises
        ------
        ValueError
            If alpha is not a float.

        ValueError
            If the value of `alpha` is not between `0.0` and `1.0`.
        """
        if isinstance(alpha, float):
            if np.any(np.logical_or(alpha <= 0, alpha >= 1.0)):
                raise ValueError(
                    "Invalid confidence_level. Allowed values are between 0.0 and 1.0."
                )
            else:
                alpha_values = [alpha / 2, 1 - alpha / 2]
                if self._central_estimator is None:
                    alpha_values.append(0.5)
                alpha_np = np.array(alpha_values)
        else:
            raise ValueError("Invalid confidence_level. Allowed values are float.")
        return alpha_np

    def pinball_loss(self, y_true: ArrayLike, y_pred: ArrayLike, level: str) -> NDArray:
        """
        Compute the pinball loss for quantile regression.

        Parameters
        ----------
        y_true : ArrayLike
            True target values.
        y_pred : ArrayLike
            Predicted target values.
        level : str
            Quantile level as a string key to access the corresponding alpha value.

        Returns
        -------
        NDArray
            Pinball loss values.
        """
        alpha = np.atleast_2d(self.quantiles[level]).T
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.array(
            np.maximum(alpha * (y_true - y_pred), (alpha - 1) * (y_true - y_pred)).mean(
                axis=1
            )
        )

    def _check_score(self, score):
        """
        Check if the score is a subclass of QuantileRegressionScore.

        Both a class and an already instantiated score are accepted, so that options
        such as `sym` can be set by the caller.
        """
        score_class = score if isinstance(score, type) else type(score)
        if not issubclass(score_class, self.ALLOWED_SCORES):
            raise ValueError(
                "Invalid score. Allowed values are subclasses of QuantileRegressionScore."
            )

    def _check_quantile_estimator(
        self,
        estimator: Optional[REGRESSOR_TYPE] = None,
    ) -> REGRESSOR_TYPE:
        """
        Perform several checks on the estimator to check if it has
        all the required specifications to be used with this methodology.
        The estimators that can be used in _QuantileConformalizer need to
        have a `fit` and `predict` attribute, but also need to allow
        a quantile loss and therefore also setting a quantile value.
        Note that there is a `TypedDict` to check which methods allow for
        quantile regression.

        Parameters
        ----------
        estimator : Optional[RegressorMixin], optional
            Estimator to check, by default `None`.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default `QuantileRegressor` instance
            with `solver` set to "highs".

        Raises
        ------
        ValueError
            If the estimator implements `fit` or `predict` methods.

        ValueError
            We check if it's a known estimator that does quantile regression
            according to the dictionnary set quantile_estimator_params.
            This dictionnary will need to be updated with the latest new
            available estimators.

        ValueError
            The estimator does not have the `"loss_name"` in its parameters
            and therefore can not be used as an estimator.

        ValueError
            There is no quantile `"loss_name"` and therefore this estimator
            can not be used as a `_MapieQuantileRegressor`.

        ValueError
            The parameter to set the alpha value does not exist in this
            estimator and therefore we cannot use it.
        """
        if estimator is None:
            return QuantileRegressor(
                solver="highs-ds",
                alpha=0.0,
            )
        _check_estimator_fit_predict(estimator)
        if isinstance(estimator, Pipeline):
            self._check_quantile_estimator(estimator[-1])
            return estimator
        else:
            name_estimator = estimator.__class__.__name__
            if name_estimator == "QuantileRegressor":
                return estimator
            else:
                if name_estimator in self.quantile_estimator_params:
                    param_estimator = estimator.get_params()
                    loss_name, alpha_name = self.quantile_estimator_params[
                        name_estimator
                    ].values()
                    if loss_name in param_estimator:
                        if param_estimator[loss_name] != "quantile":
                            raise ValueError(
                                "You need to set the loss/objective argument"
                                + " of your base model to `quantile`."
                            )
                        else:
                            if alpha_name in param_estimator:
                                return estimator
                            else:
                                raise ValueError(
                                    "The matching parameter `alpha_name` for"
                                    " estimator does not exist. "
                                    "Make sure you set it when initializing "
                                    "your estimator."
                                )
                    else:
                        raise ValueError(
                            "The matching parameter `loss_name` for"
                            + " estimator does not exist."
                        )
                else:
                    raise ValueError(
                        "The base model is not supported. \n"
                        "Give a base model among: \n"
                        f"{self.quantile_estimator_params.keys()} "
                        "Or, add your base model to" + " `quantile_estimator_params`."
                    )

    # Potential caching here
    def get_estimator_name(self) -> str:
        """
        Get the name of the estimator class.

        Returns
        -------
        str
            The name of the estimator class.
        """
        if isinstance(self.estimator, Pipeline):
            return str(self.estimator[-1].__class__.__name__)
        return str(self.estimator.__class__.__name__)

    def _set_estimator_params(
        self, estimator: REGRESSOR_TYPE, **params
    ) -> REGRESSOR_TYPE:
        """
        Set the parameters of the estimator to the given alpha value.

        Parameters
        ----------
        estimator : RegressorMixin
            The estimator to set the parameters for.

        params : dict
            The parameters to set for the estimator.

        Returns
        -------
        RegressorMixin
            The estimator with updated parameters.
        """
        if isinstance(estimator, Pipeline):
            estimator[-1].set_params(**params)
        else:
            estimator.set_params(**params)
        return estimator

    def _initialize_fit_conformalize(self) -> None:
        self.quantiles = {str(alpha): self._check_alpha(alpha) for alpha in self.alpha}
        self.estimators_ = {
            str(alpha): {
                "lower": [],
                "upper": [],
            }
            for alpha in self.alpha
        }
        self.central_estimators_ = []
        self.n_calib_samples = []
        self.conformity_scores = {str(alpha): np.array([]) for alpha in self.alpha}
        self.pinball_losses = {str(alpha): [] for alpha in self.alpha}
        self.key_mapping = {"lower": 0, "upper": 1, "central": 2}
        self.__central_fitted = False

        if self.method == "base":
            self._base_estimator_ = {
                str(alpha): {
                    "lower": [],
                    "upper": [],
                }
                for alpha in self.alpha
            }
            self._base_central_estimator_ = []

    @property
    @lru_cache(maxsize=None)
    def reverse_key_mapping(self) -> dict[int, str]:
        """
        Get the reverse mapping of key_mapping.

        Returns
        -------
        dict[int, str]
            The reverse mapping of key_mapping.
        """
        return {v: k for k, v in self.key_mapping.items()}

    def _set_quantile_estimator_params(
        self, estimator: REGRESSOR_TYPE, alpha: float, alpha_name: str, **params
    ) -> REGRESSOR_TYPE:
        """
        Set the parameters of the estimator to the given alpha value.

        Parameters
        ----------
        estimator : RegressorMixin
            The estimator to set the parameters for.
        alpha : float
            The quantile level to set for the estimator.
        **params : dict
            Additional parameters to set for the estimator.

        Returns
        -------
        RegressorMixin
            The estimator with updated parameters.
        """
        cloned_estimator_ = clone(estimator)
        params = {alpha_name: alpha}
        return self._set_estimator_params(cloned_estimator_, **params)

    # -------------------------------------- Fit
    # TODO: Nearly duplicated from EnsemblRegressor _fit_oof_estimator -> should be factorize in next refacto
    def _fit_cv_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        level: str,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params,
    ) -> dict[str, List[RegressorMixin]]:
        """Fit the cross-validated estimator.
        Parameters
        ----------
        estimator: RegressorMixin
            Estimator to train.

        level: str
            The quantile level to fit.

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples,)
            Input labels.

        train_index: ArrayLike of shape (n_samples_train)
            Training data indices.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default `None`.

        **fit_params : dict
            Additional fit parameters.
        """
        # The interest of this method is the safe indexing, can be converted into a checking function
        # TODO back-end: avoid using private utilities from sklearn like
        # _safe_indexing (may break anytime without notice)
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        if sample_weight is not None:
            sample_weight = _safe_indexing(sample_weight, train_index)
            sample_weight = cast(NDArray, sample_weight)
        return self._fit_quantiles(
            X_train,
            y_train,
            level,
            sample_weight=sample_weight,
            **fit_params,
        )

    # TODO: A structure can handle quantiles fitting and prediction to avoid code duplication between conformalizer
    def _fit_quantiles(
        self,
        X: ArrayLike,
        y: ArrayLike,
        level: str,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params,
    ) -> dict[str, List[RegressorMixin]]:
        """
        Fits the estimators with provided training data
        and stores them in self.estimators_.
        """
        checked_estimator = self._check_quantile_estimator(self.estimator)

        X, y = indexable(X, y)
        y = _check_y(y)

        estimators_: dict[str, List[RegressorMixin]] = {
            "lower": [],
            "upper": [],
            "central": [],
        }
        sample_weight, X, y = _check_null_weight(sample_weight, X, y)
        estimator_name = self.get_estimator_name()
        alpha_name = self.quantile_estimator_params[estimator_name]["alpha_name"]
        for i, alpha in enumerate(self.quantiles[level]):
            cloned_estimator_ = self._set_quantile_estimator_params(
                checked_estimator,
                alpha,
                alpha_name=alpha_name,
            )
            estimators_[self.reverse_key_mapping[i]].append(
                _fit_estimator(
                    cloned_estimator_,
                    X,
                    y,
                    sample_weight,
                    **fit_params,
                )
            )

        # The central estimator predicts the median directly: it only needs `fit` and `predict`
        if self._central_estimator is not None and self.fit_central_estimator:
            _check_estimator_fit_predict(self._central_estimator)
            cloned_estimator = clone(self._central_estimator)
            estimators_["central"].append(
                _fit_estimator(
                    cloned_estimator,
                    X,
                    y,
                    sample_weight,
                    **fit_params,
                )
            )
        elif self._central_estimator is not None:
            estimators_["central"].append(self._central_estimator)

        return estimators_

    # TODO: Duplicated from CrossConformalRegressor -> should be factorize in next refacto
    def reset(self: _QuantileConformalizerT) -> _QuantileConformalizerT:
        """
        Discard previously computed conformity scores so that
        `fit_conformalize` can be called again with new data.

        Returns
        -------
        Self
            This conformalizer instance, reset to its pre-fit state.
        """
        self.is_fitted = False
        self.is_conformalized = False
        self.estimators_ = {
            str(alpha): {
                "lower": [],
                "upper": [],
            }
            for alpha in self.alpha
        }
        self.central_estimators_ = []
        self.n_calib_samples = []
        self.conformity_scores = {str(alpha): np.array([]) for alpha in self.alpha}
        self.pinball_losses = {str(alpha): np.array([]) for alpha in self.alpha}
        self._predict_params = {}
        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> _Conformalizer:
        """
        Fits the estimators with provided training data
        and stores them in self.estimators_.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
            Input labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        groups: Optional[ArrayLike] of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into folds.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        Self
            This _QuantileConformalizer instance, fitted.
        """
        if self.is_fitted:
            warnings.warn(
                "The fit method has already been called. "
                "Calling it again will overwrite the previous fitted estimators."
            )
            self.reset()

        self._initialize_fit_conformalize()

        n_samples = _num_samples(y)

        if self.cv == "prefit":
            # Create a placeholder attribute 'k_' filled with NaN values
            # This attribute is defined for consistency but
            # is not used in prefit mode
            self.k_ = np.full(shape=(n_samples, 1), fill_value=np.nan, dtype=float)

        else:
            for alpha in self.alpha:
                level = str(alpha)
                cv = cast(BaseCrossValidator, self.cv)
                self.k_ = np.full(
                    shape=(n_samples, cv.get_n_splits(X, y, groups)),
                    fill_value=np.nan,
                    dtype=float,
                )
                list_estimators = Parallel(self.n_jobs, verbose=self.verbose)(
                    delayed(self._fit_cv_estimator)(
                        X,
                        y,
                        train_index,
                        level,
                        sample_weight,
                        **fit_params,
                    )
                    for train_index, _ in cv.split(X, y, groups)
                )
                for dict_estimator in list_estimators:
                    for key, fitted_estimators in dict_estimator.items():
                        if key != "central":
                            self.estimators_[level][key].extend(fitted_estimators)
                        elif not self.__central_fitted:
                            self.central_estimators_.extend(fitted_estimators)

                if self.method == "base":
                    base_estimators_ = self._fit_quantiles(
                        X, y, level, sample_weight=sample_weight, **fit_params
                    )
                    for key, fitted_estimators in base_estimators_.items():
                        if key != "central":
                            self._base_estimator_[level][key].extend(fitted_estimators)
                        elif not self.__central_fitted:
                            self._base_central_estimator_.extend(fitted_estimators)

                self.__central_fitted = True

        self.is_fitted = True
        return self

    # ---------------------Conformalizer
    # TODO: Nearly duplicated from EnsemblRegressor _predict_oof_estimator -> should be factorize in next refacto
    def _predict_oof(
        self,
        X: ArrayLike,
        val_index: ArrayLike,
        index: int,
        level: str,
        **predict_params,
    ) -> NDArray:
        """
        Perform predictions on a single out-of-fold model on a validation set.

        Parameters
        ----------
        index: int
            Index of the estimator to use.

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        val_index: ArrayLike of shape (n_samples_val)
            Validation data indices.

        level: str
            The quantile level to predict.

        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        NDArray
            Predictions of estimator from val_index of X.
        """
        X_val = _safe_indexing(X, val_index)
        if _num_samples(X_val) > 0:
            y_pred = self._predict_quantiles(
                X_val, index=index, level=level, **predict_params
            )
        else:
            y_pred = np.array([])
        return y_pred

    # TODO: Nearly duplicated from EnsemblRegressor predict_calib -> should be factorize in next refacto
    def _predict_calib(
        self,
        X: ArrayLike,
        level: str,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **predict_params,
    ) -> NDArray:
        """
        Perform predictions on X : the calibration set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data

        level: str
            The quantile level to predict.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

            By default `None`.

        groups: Optional[ArrayLike] of shape (n_samples_test,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default `None`.

        **predict_params : dict
            Additional predict parameters.

        Returns
        -------
        NDArray of shape (n_samples_test, 1)
            The predictions.
        """
        check_is_fitted(self)

        n_samples = _num_samples(X)
        n_splits = self.cv.get_n_splits(X, y, groups)
        indices = [calib_index for _, calib_index in self.cv.split(X, y, groups)]
        if len(self.n_calib_samples) < n_splits:
            self.n_calib_samples = [len(calib_index) for calib_index in indices]

        if self.cv == "prefit":
            y_pred = self._predict_quantiles(X, level=level, index=0, **predict_params)
        else:
            pred_matrix = np.full(
                shape=(n_samples, n_splits, len(self.quantiles[level])),
                fill_value=np.nan,
                dtype=np.float64,
            )
            outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._predict_oof)(
                    X, calib_index, level=level, index=model_index, **predict_params
                )
                for calib_index, model_index in zip(indices, range(n_splits))
            )
            self.pinball_losses[level] = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose
            )(
                delayed(self.pinball_loss)(
                    _safe_indexing(y, calib_index), output, level
                )
                for calib_index, output in zip(indices, outputs)
            )

            for i, ind in enumerate(indices):
                pred_matrix[ind, i, :] = np.array(outputs[i], dtype=float).T
                self.k_[ind, i] = 1

            _check_nan_in_aposteriori_prediction(pred_matrix)
            y_pred = aggregate_all(self.agg_function, pred_matrix)

        return cast(NDArray, y_pred)

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        **predict_params: Any,
    ) -> _QuantileConformalizer:
        """
        Conformalize the model using the provided calibration data.

        Parameters
        ----------
        X : ArrayLike
            Calibration features.
        y : ArrayLike
            Calibration targets.
        groups : Optional[ArrayLike], optional
            Group labels for the samples, by default None
        **predict_params : Any
            Additional parameters to pass to the prediction method.

        Returns
        -------
        _QuantileConformalizer
            The conformalized quantile regressor.
        """
        X_calib, y_calib = cast(ArrayLike, X), cast(ArrayLike, y)
        X_calib, y_calib = indexable(X_calib, y_calib)
        y_calib = cast(NDArray, _check_y(y_calib))

        _raise_error_if_previous_method_not_called(
            "conformalize",
            "fit",
            self.is_fitted,
        )
        if self.is_conformalized:
            warnings.warn(
                "The conformalize method has already been called. "
                "Calling it again will overwrite the previous conformity scores."
            )
            self.conformity_scores = {str(alpha): np.array([]) for alpha in self.alpha}

        for alpha in self.alpha:
            level = str(alpha)
            pred = self._predict_calib(
                X_calib, level, y_calib, groups, **predict_params
            )
            self.conformity_scores[level] = self.score.get_conformity_scores(
                y_calib, pred.T, X=X_calib
            )

        self.is_conformalized = True
        return self

    # ------------------------------ Predict
    def _pinball_weighted_mean(self, y_preds: ArrayLike, level: str) -> NDArray:
        """
        Computes the weighted mean of the predicted values using the pinball losses as weights.

        The weight of a fold is the reciprocal of its pinball loss, so that folds
        predicting a quantile accurately weigh more. Weights are normalized over the
        folds, independently for each quantile.

        Parameters
        ----------
        y_preds : ArrayLike of shape (n_samples, n_split, n_quantiles)
            Predictions of the estimator of each cross-validation fold, one column
            per quantile.

        level : str
            The confidence level whose pinball losses are used as weights.

        Returns
        -------
        NDArray of shape (n_samples, n_quantiles)
            Fold predictions aggregated for each sample and each quantile.
        """
        pinball_weights = 1 / (
            np.asarray(self.pinball_losses[level], dtype=float) + 1e-8
        )
        y_preds = np.asarray(y_preds, dtype=float)

        # Losses are of shape (n_split, n_quantiles), one per fold and per quantile.
        # A single loss per fold is broadcast to every quantile.
        if pinball_weights.ndim == 1:
            pinball_weights = pinball_weights[:, np.newaxis]
        weights = pinball_weights / pinball_weights.sum(axis=0, keepdims=True)

        # Reduce the fold axis, keeping one prediction per sample and per quantile.
        weighted_values = np.sum(weights[np.newaxis, :, :] * y_preds, axis=1)

        return np.asarray(weighted_values)

    # TODO: A structure can handle quantiles fitting and prediction to avoid code duplication between conformalizer
    def _predict_quantiles(
        self, X: ArrayLike, index: int, level: str, **predict_params
    ) -> NDArray:
        """
        Predicts the lower and upper quantiles for the given input data X using the specified index.

        Parameters
        ----------
        X : ArrayLike
            Input data for which to predict quantiles.
        index : int
            Index of the estimator to use for prediction (for split method this is always 0)
        level: str
            The quantile level to predict.
        **predict_params : Any
            Additional parameters to pass to the predict method of the estimators.

        Returns
        -------
        NDArray
            Predicted lower and upper quantiles for the input data X as distinct lines.
        """
        preds = [
            self.estimators_[level]["lower"][index]
            .predict(X, **predict_params)
            .ravel(),
            self.estimators_[level]["upper"][index]
            .predict(X, **predict_params)
            .ravel(),
        ]
        if self.quantiles[level].size == 3:
            preds.append(
                self.central_estimators_[index].predict(X, **predict_params).ravel()
            )
        return np.vstack(preds)

    def _predict_center(self, X: ArrayLike, index: int, **predict_params) -> NDArray:
        """
        Predicts the central quantile for the given input data X using the specified index.

        Parameters
        ----------
        X : ArrayLike
            Input data for which to predict the central quantile.
        index : int
            Index of the estimator to use for prediction (for split method this is always 0)
        **predict_params : Any
            Additional parameters to pass to the predict method of the estimators.

        Returns
        -------
        NDArray
            Predicted central quantile for the input data X.
        """
        return np.asarray(
            self.central_estimators_[index].predict(X, **predict_params).ravel()
        )

    def _predict_base(self, X: ArrayLike, level: str, **predict_params):
        """
        Predicts for base strategy the lower and upper quantiles for the given input data X using the base estimator.
        """
        y_pred_low = self._base_estimator_[level]["lower"][0].predict(
            X, **predict_params
        )
        y_pred_up = self._base_estimator_[level]["upper"][0].predict(
            X, **predict_params
        )
        y_pred_center = self._base_central_estimator_[0].predict(X, **predict_params)
        return np.column_stack((y_pred_low, y_pred_up, y_pred_center))

    def _predict_aggregate(self, X: ArrayLike, level: str, **predict_params):
        """
        Predicts the lower and upper quantiles for the given input data X using the specified level and aggregation function.
        """
        n_split = len(self.estimators_[level]["lower"])
        n_quantiles = len(self.quantiles[level])
        pred_matrix = np.full(
            shape=(_num_samples(X), n_split, 3),
            fill_value=np.nan,
            dtype=np.float64,
        )
        for i in range(n_split):
            pred_matrix[:, i, :n_quantiles] = self._predict_quantiles(
                X, index=i, level=level, **predict_params
            ).T
            if n_quantiles < 3:
                pred_matrix[:, i, 2] = self._predict_center(
                    X, index=i, **predict_params
                )

        if self.agg_function == "pinball_weighted_mean":
            y_pred = self._pinball_weighted_mean(pred_matrix[:, :, :n_quantiles], level)
            if n_quantiles < 3:
                y_pred_multi_center = np.mean(
                    pred_matrix[:, :, 2], axis=1, keepdims=True
                )
                y_pred = np.hstack((y_pred, y_pred_multi_center))
        else:
            y_pred = aggregate_all(self.agg_function, pred_matrix)

        # `"minmax"` only changes how the bounds are aggregated: the central
        # prediction keeps the aggregation requested by the user, so that it matches
        # the point predicted by `predict`.
        if self.method == "minmax":
            y_pred[:, 0] = np.min(pred_matrix[:, :, 0], axis=1)
            y_pred[:, 1] = np.max(pred_matrix[:, :, 1], axis=1)

        return y_pred

    def _predict(
        self, X: ArrayLike, ensemble: bool = False, **predict_params
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Predicts the lower and upper quantiles for the given input data X.

        Parameters
        ----------
        X : ArrayLike
            Input data for which to predict quantiles.
        ensemble : Optional[bool], default=None
            Whether to use the ensemble of estimators for prediction (for compatibility with scores).
        **predict_params : Any
            Additional parameters to pass to the predict method of the estimators.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            Predicted lower, upper, and central quantiles for the input data X,
            the bounds holding one column per confidence level of `self.alpha`.
        """

        centers, lowers, uppers = [], [], []

        for alpha in self.alpha:
            center, lower, upper = self._predict_level(
                X, str(alpha), ensemble, **predict_params
            )
            centers.append(center)
            lowers.append(lower)
            uppers.append(upper)

        # The central estimator is shared across confidence levels, so every entry of
        # `centers` is identical: return it once, without an alpha axis.
        return centers[0], np.stack(lowers, axis=-1), np.stack(uppers, axis=-1)

    def _predict_level(
        self, X: ArrayLike, level: str, ensemble: bool, **predict_params
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Predicts the central, lower and upper quantiles for a single confidence level.

        Parameters
        ----------
        X : ArrayLike
            Input data for which to predict quantiles.
        level : str
            The confidence level to predict.
        ensemble : bool
            Whether to use the ensemble of estimators for prediction (for
            compatibility with scores).
        **predict_params : Any
            Additional parameters to pass to the predict method of the estimators.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            Predicted central, lower and upper quantiles, each of shape
            `(n_samples,)`.
        """
        if self.method == "base":
            y_pred = self._predict_base(X, level=level, **predict_params)
        else:
            y_pred = self._predict_aggregate(X, level=level, **predict_params)

        return y_pred[:, 2], y_pred[:, 0], y_pred[:, 1]


class CrossConformalizedQuantileRegressor(_QuantileConformalizer):
    """
    Computes prediction intervals using the cross-conformalized quantile regression technique.

    The estimator fits a cross-validated set of quantile regressors and then
    calibrates them on a conformalization set to produce prediction intervals.
    The `fit_conformalize` convenience method performs both steps in sequence.

    Parameters
    ----------
    estimator : RegressorMixin, default=QuantileRegressor()
        Base regressor used to estimate the lower, upper and central quantiles.

    confidence_level : float or Iterable[float], default=0.9
        Target confidence level of the prediction intervals. An iterable requests
        one interval per level, which `predict_interval` returns along the last axis.

    conformity_score : QuantileRegressionScore or type, default=AbsoluteQuantileRegressionScore
        Conformity score used to calibrate the quantile estimates. A class is
        instantiated with its defaults; pass an instance to configure it.

    method : str, default="plus"
        Cross-conformalization strategy. Allowed values are
        `"base"`, `"plus"` and `"minmax"`.

    cv : int or BaseCrossValidator, default=5
        Cross-validation splitter used to build the ensemble of estimators.

    n_jobs : Optional[int], default=None
        Number of jobs used for parallel fitting.

    verbose : int, default=0
        Verbosity level passed to the underlying parallel fitting routine.

    random_state : Optional[int or np.random.RandomState], default=None
        Controls randomness for the internal cross-validation procedures: it seeds
        the shuffling of the `KFold` built when `cv` is an integer. Pass an integer
        for reproducible folds. If `None`, the seed is drawn from the global numpy
        random state, so the folds differ from one instance to the next. Ignored
        when `cv` is a cross-validator, which carries its own randomness.

    central_estimator : Optional[RegressorMixin], default=None
        Optional estimator used to predict the central value directly.

    fit_central_estimator : bool, default=True
        Whether to fit an estimator dedicated to the central prediction.

    References
    ----------
    Yaniv Romano, Evan Patterson and Emmanuel J. Candès.
    "Conformalized Quantile Regression"
    Advances in neural information processing systems 32 (2019).
    """

    _VALID_METHODS = ["base", "plus", "minmax"]
    ALLOWED_AGG_FUNCTIONS = ["mean", "median", "pinball_weighted_mean"]

    def __init__(
        self,
        estimator: RegressorMixin = QuantileRegressor(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[
            QuantileRegressionScore, Type[QuantileRegressionScore]
        ] = AbsoluteQuantileRegressionScore,
        method: str = "plus",
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        central_estimator: Optional[RegressorMixin] = None,
        fit_central_estimator: Optional[bool] = True,
    ) -> None:
        _check_if_param_in_allowed_values(
            method, "method", CrossConformalizedQuantileRegressor._VALID_METHODS
        )
        _check_cv_not_string(cv)
        _check_cv_not_subsample(cv)
        self._check_quantile_estimator(estimator)
        self._check_score(conformity_score)
        check_random_state(random_state)

        # Instantiate conformity score if it's a class
        if isinstance(conformity_score, type):
            self.score = conformity_score()
        else:
            self.score = conformity_score
        self.estimator = estimator
        self.method = method
        self.random_state = random_state
        # `random_state` seeds the shuffling of the `KFold` that an integer `cv` is
        # turned into. Without it `_check_cv` draws a seed from the global numpy
        # random state, making the folds differ from one instance to the next.
        self.cv = _check_cv(cv, random_state=random_state)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.alpha = _transform_confidence_level_to_alpha_list(confidence_level)
        self.is_fitted = self.is_conformalized = False

        self._predict_params = {}
        self._central_estimator = central_estimator
        self.fit_central_estimator = fit_central_estimator
        self.quantiles = {str(alpha): self._check_alpha(alpha) for alpha in self.alpha}
        self.agg_function = "mean"

    # ---------------------Fit and Conformalize
    # TODO: Nearly duplicated from CrossConformalRegressor -> should be factorize in next refacto
    def fit_conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
    ) -> CrossConformalizedQuantileRegressor:
        """
        Estimates the uncertainty of the base regressor in a cross-validation style:
        fits the base regressor on different folds of the dataset
        and computes conformity scores on the corresponding out-of-fold data.

        If called on an instance that has already been fitted, a `UserWarning` is
        emitted and the previously computed conformity scores are discarded before
        the new fit. Call `reset()` explicitly to suppress the warning.

        Parameters
        ----------
        X : ArrayLike
            Features

        y : ArrayLike
            Targets

        groups: Optional[ArrayLike] of shape (n_samples,), default=None
            Groups to pass to the cross-validator.

        fit_params : Optional[dict], default=None
            Parameters to pass to the `fit` method of the base regressor.

        predict_params : Optional[dict], default=None
            Parameters to pass to the `predict` method of the base regressor.
            These parameters will also be used in the `predict_interval`
            and `predict` methods of this CrossConformalRegressor.

        Returns
        -------
        Self
            This CrossConformalRegressor instance, fitted and conformalized.
        """
        if self.is_fitted_and_conformalized:
            warnings.warn(
                "CrossConformalRegressor.fit_conformalize was already called; "
                "conformity scores from the previous fit will be discarded. "
                "Call .reset() explicitly before fit_conformalize to suppress "
                "this warning.",
                UserWarning,
                stacklevel=2,
            )
            self.reset()

        fit_params_ = _prepare_params(fit_params)
        predict_params_ = _prepare_params(predict_params)
        self._predict_params = predict_params_
        self.fit(
            X,
            y,
            groups=groups,
            **fit_params_,
        )
        self.conformalize(X, y, groups, **predict_params_)
        return self

    @property
    def is_fitted_and_conformalized(self) -> bool:
        """Returns True if the estimator is fitted and conformalized"""
        return self.is_fitted and self.is_conformalized

    # --------------------- Prediction
    # TODO: Nearly duplicated from CrossConformalRegressor
    def predict_interval(
        self,
        X: ArrayLike,
        aggregate_point_predictions: Optional[str] = AGGREGATE_POINT_PREDICTIONS_AUTO,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predicts points and intervals.

        If several confidence levels were provided during initialisation, several
        intervals will be predicted for each sample. See the return signature.

        By default, points are predicted using an aggregation.
        See the `aggregate_point_predictions` parameter.

        Parameters
        ----------
        X : ArrayLike
            Features

        aggregate_point_predictions : Optional[str], default="auto"
            The method to predict a point. The valid options depend on `method`,
            since only `method="base"` fits an estimator on the entire data while
            `"plus"` and `"minmax"` aggregate the cross-validation folds. Options:

            - "auto": resolves to `None` if `method="base"`, to
                `"pinball_weighted_mean"` otherwise
            - None: a point is predicted using the regressor trained on the entire
                data. Only valid with `method="base"`
            - "mean": Averages the predictions of the regressors trained on each
                cross-validation fold
            - "median": Aggregates (using median) the predictions of the regressors
                trained on each cross-validation fold
            - "pinball_weighted_mean": Aggregates the predictions of the regressors
                trained on each cross-validation fold using a weighted mean, where the
                weights are the pinball losses of each fold.

            The three aggregations above are invalid with `method="base"`.

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the interval width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.


        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape `(n_samples,)`
            - Prediction intervals, of shape `(n_samples, 2, n_confidence_levels)`
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval",
            "conformalize",
            self.is_conformalized,
        )

        ensemble = self._set_aggregate_point_predictions_and_return_ensemble(
            aggregate_point_predictions
        )

        # `get_bounds` vectorizes over the confidence levels of `alpha_np`, but it takes
        # a single conformity score array and a single pair of predicted bounds, shared
        # across those levels. Here every level has its own quantile estimators and its
        # own scores, so the bounds are computed level by level and stacked afterwards.
        bounds_low, bounds_up = [], []
        y_pred: Optional[NDArray] = None

        for alpha in self.alpha:
            level = str(alpha)
            alpha_np = np.array([alpha], dtype=float)
            scores = self.conformity_scores[level]

            if not allow_infinite_bounds:
                n = self.score.get_effective_calibration_samples(scores)
                _check_alpha_and_n_samples(alpha_np, n)

            # Predict the target with confidence intervals
            y_pred, y_pred_low, y_pred_up = self.score.predict_set(
                cast(NDArray, X),
                alpha_np,
                estimator=self,
                conformity_scores=scores,
                ensemble=ensemble,
                method=self.method,
                optimize_beta=minimize_interval_width,
                allow_infinite_bounds=allow_infinite_bounds,
                **self._predict_params,
            )
            # A single level is requested per call, so `get_bounds` returns bounds of
            # shape (n_samples, 1); drop that axis before stacking the levels.
            bounds_low.append(np.reshape(y_pred_low, -1))
            bounds_up.append(np.reshape(y_pred_up, -1))

        intervalles = np.stack(
            [np.stack(bounds_low, axis=-1), np.stack(bounds_up, axis=-1)], axis=1
        )

        return cast(NDArray, y_pred), intervalles

    # TODO: Duplicated from CrossConformalRegressor
    def predict(
        self,
        X: ArrayLike,
        aggregate_point_predictions: Optional[str] = AGGREGATE_POINT_PREDICTIONS_AUTO,
    ) -> NDArray:
        """
        Predicts points.

        By default, points are predicted using an aggregation.
        See the `aggregate_point_predictions` parameter.

        Parameters
        ----------
        X : ArrayLike
            Features

        aggregate_point_predictions : Optional[str], default="auto"
            The method to predict a point. The valid options depend on `method`,
            since only `method="base"` fits an estimator on the entire data while
            `"plus"` and `"minmax"` aggregate the cross-validation folds. Options:

            - "auto": resolves to `None` if `method="base"`, to
              `"pinball_weighted_mean"` otherwise
            - None: a point is predicted using the regressor trained on the entire
              data. Only valid with `method="base"`
            - "mean": Averages the predictions of the regressors trained on each
              cross-validation fold
            - "median": Aggregates (using median) the predictions of the regressors
              trained on each cross-validation fold
            - "pinball_weighted_mean": Aggregates the predictions of the regressors
              trained on each cross-validation fold using a weighted mean, where the
              weights are the pinball losses of each fold.

            The three aggregations above are invalid with `method="base"`.

        Returns
        -------
        NDArray
            Array of point predictions, with shape `(n_samples,)`.
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        ensemble = self._set_aggregate_point_predictions_and_return_ensemble(
            aggregate_point_predictions
        )

        # Reached only with `method="base"
        if not ensemble:
            return np.asarray(
                self._base_central_estimator_[0]
                .predict(X, **self._predict_params)
                .ravel()
            )

        y_pred_multi = np.vstack(
            [
                self._predict_center(X, index=i, **self._predict_params)
                for i in range(len(self.central_estimators_))
            ]
        )

        # `aggregate_point_predictions` may be the `"auto"` sentinel, so read the
        # aggregation resolved above rather than the argument.
        level = str(self.alpha[0])
        if self.agg_function == "pinball_weighted_mean" and (
            self.quantiles[level].size == 3
        ):
            central_weights = 1 / (
                np.asarray(self.pinball_losses[level], dtype=float)[:, 2] + 1e-8
            )
            weights = central_weights / central_weights.sum()
            return np.asarray(np.sum((np.atleast_2d(weights).T * y_pred_multi), axis=0))
        if self.agg_function == "median":
            return np.asarray(np.median(y_pred_multi, axis=0))
        return np.asarray(np.mean(y_pred_multi, axis=0))

    # TODO: Duplicated from CrossConformalRegressor
    def _set_aggregate_point_predictions_and_return_ensemble(
        self, aggregate_point_predictions: Optional[str]
    ) -> bool:
        """
        Resolves the point aggregation and checks it against the method.

        `None` means that no aggregation happens: the point is predicted by the
        estimator fitted on the entire data, which only the `"base"` method fits.
        Conversely `"plus"` and `"minmax"` aggregate the cross-validation folds, so
        they require an aggregation function. `"auto"` resolves to whichever of the
        two the method supports.

        Parameters
        ----------
        aggregate_point_predictions : Optional[str]
            The aggregation asked for, possibly `"auto"`.

        Returns
        -------
        bool
            Whether the cross-validation folds must be aggregated, that is the
            `ensemble` flag passed down to the conformity score.
        """
        if aggregate_point_predictions == AGGREGATE_POINT_PREDICTIONS_AUTO:
            aggregate_point_predictions = (
                None if self.method == "base" else "pinball_weighted_mean"
            )

        if aggregate_point_predictions is None:
            if self.method != "base":
                raise ValueError(
                    "aggregate_point_predictions=None predicts points with an "
                    "estimator fitted on the entire data, which only method='base' "
                    f"fits. Got method='{self.method}', which aggregates the "
                    "cross-validation folds: pass one of "
                    f"{CrossConformalizedQuantileRegressor.ALLOWED_AGG_FUNCTIONS}."
                )
            return False

        if self.method == "base":
            raise ValueError(
                "method='base' predicts points with the estimator fitted on the "
                "entire data, so it does not aggregate the cross-validation folds. "
                "Only aggregate_point_predictions=None is valid, got "
                f"'{aggregate_point_predictions}'."
            )

        if (
            aggregate_point_predictions
            not in CrossConformalizedQuantileRegressor.ALLOWED_AGG_FUNCTIONS
        ):
            raise ValueError("The value of the aggregation function is not correct")

        self.agg_function = aggregate_point_predictions
        return True


class ConformalizedQuantileRegressor:
    """
    Computes prediction intervals using the conformalized quantile regression technique:

    1. The `fit` method fits three models to the training data using the provided
       regressor: a model to predict the target, and models to predict upper
       and lower quantiles around the target.
    2. The `conformalize` method estimates the uncertainty of the quantile models
       using the conformalization set.
    3. The `predict_interval` computes prediction points and intervals.

    Parameters
    ----------
    estimator : Union[`RegressorMixin`, `Pipeline`, \
`List[Union[RegressorMixin, Pipeline]]`]
        The regressor used to predict points and quantiles.

        When `prefit=False` (default), a single regressor that supports the quantile
        loss must be passed. Valid options:

        - `sklearn.linear_model.QuantileRegressor`
        - `sklearn.ensemble.GradientBoostingRegressor`
        - `sklearn.ensemble.HistGradientBoostingRegressor`
        - `lightgbm.LGBMRegressor`

        When `prefit=True`, a list of three fitted quantile regressors predicting the
        lower, upper, and median quantiles must be passed (in that order).
        These quantiles must be:

        - `lower quantile = (1 - confidence_level) / 2`
        - `upper quantile = (1 + confidence_level) / 2`
        - `median quantile = 0.5`

    confidence_level : float default=0.9
        The confidence level for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals.

    prefit : bool, default=False
        If True, three fitted quantile regressors must be provided, and the `fit`
        method must be skipped.

        If False, the three regressors will be fitted during the `fit` method.

    Examples
    --------
    >>> from mapie.regression import ConformalizedQuantileRegressor
    >>> from mapie.utils import train_conformalize_test_split
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import QuantileRegressor

    >>> X, y = make_regression(n_samples=500, n_features=2, noise=1.0)
    >>> (
    ...     X_train, X_conformalize, X_test,
    ...     y_train, y_conformalize, y_test
    ... ) = train_conformalize_test_split(
    ...     X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
    ... )

    >>> mapie_regressor = ConformalizedQuantileRegressor(
    ...     estimator=QuantileRegressor(),
    ...     confidence_level=0.95,
    ... ).fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)

    >>> predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)

    References
    ----------
    Yaniv Romano, Evan Patterson and Emmanuel J. Candès.
    "Conformalized Quantile Regression"
    Advances in neural information processing systems 32 (2019).
    """

    def __init__(
        self,
        estimator: Optional[
            Union[RegressorMixin, Pipeline, List[Union[RegressorMixin, Pipeline]]]
        ] = None,
        confidence_level: float = 0.9,
        prefit: bool = False,
    ) -> None:
        self._alpha = _transform_confidence_level_to_alpha(confidence_level)
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False

        self._mapie_quantile_regressor = _MapieQuantileRegressor(
            estimator=estimator,
            method="quantile",
            cv="prefit" if prefit else "split",
            alpha=self._alpha,
        )

        self._predict_params: dict = {}

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> ConformalizedQuantileRegressor:
        """
        Fits three models using the regressor provided at initialisation:

        - a model to predict the target
        - a model to predict the upper quantile of the target
        - a model to predict the lower quantile of the target

        Parameters
        ----------
        X_train : ArrayLike
            Training data features.

        y_train : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Parameters to pass to the `fit` method of the regressors.

        Returns
        -------
        Self
            The fitted ConformalizedQuantileRegressor instance.
        """
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)

        fit_params_ = _prepare_params(fit_params)
        self._mapie_quantile_regressor._initialize_fit_conformalize()
        self._mapie_quantile_regressor._fit_estimators(
            X=X_train,
            y=y_train,
            **fit_params_,
        )

        self._is_fitted = True
        return self

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> ConformalizedQuantileRegressor:
        """
        Estimates the uncertainty of the quantile regressors by computing
        conformity scores on the conformalization set.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Features of the conformalization set.

        y_conformalize : ArrayLike
            Targets of the conformalization set.

        predict_params : Optional[dict], default=None
            Parameters to pass to the `predict` method of the regressors.
            These parameters will also be used in the `predict_interval`
            and `predict` methods of this SplitConformalRegressor.

        Returns
        -------
        Self
            The ConformalizedQuantileRegressor instance.
        """
        _raise_error_if_previous_method_not_called(
            "conformalize",
            "fit",
            self._is_fitted,
        )
        _raise_error_if_method_already_called(
            "conformalize",
            self._is_conformalized,
        )

        self._predict_params = _prepare_params(predict_params)
        self._mapie_quantile_regressor.conformalize(
            X_conformalize, y_conformalize, **self._predict_params
        )

        self._is_conformalized = True
        return self

    def predict_interval(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
        symmetric_correction: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predicts points (using the base regressor) and intervals.

        The returned NDArray containing the prediction intervals is of shape
        (n_samples, 2, 1). The third dimension is unnecessary, but kept for consistency
        with the other conformal regression methods available in MAPIE.

        Parameters
        ----------
        X : ArrayLike
            Features

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the intervals width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.

        symmetric_correction : bool, default=False
            To produce prediction intervals, the conformalized quantile regression
            technique corrects the predictions of the upper and lower quantile
            regressors by adding a constant.

            If `symmetric_correction` is set to `False` , this constant is different
            for the upper and the lower quantile predictions. If set to `True`,
            this constant is the same for both.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape `(n_samples,)`
            - Prediction intervals, of shape `(n_samples, 2, 1)`
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval",
            "conformalize",
            self._is_conformalized,
        )

        predictions = self._mapie_quantile_regressor.predict(
            X,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds,
            symmetry=symmetric_correction,
            **self._predict_params,
        )
        return _cast_predictions_to_ndarray_tuple(predictions)

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Predicts points.

        Parameters
        ----------
        X : ArrayLike
            Features

        Returns
        -------
        NDArray
            Array of point predictions with shape `(n_samples,)`.
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "conformalize",
            self._is_conformalized,
        )

        estimator = self._mapie_quantile_regressor
        predictions, _ = estimator.predict(X, **self._predict_params)
        return predictions

    @property
    def conformity_scores(self) -> NDArray:
        """
        Returns the conformity scores computed by the `conformalize` method
        on the conformalization set.

        For conformalized quantile regression, three scores are stored per
        sample: the signed residual against the lower-quantile estimator,
        the signed residual against the upper-quantile estimator, and their
        pointwise maximum.

        Returns
        -------
        NDArray
            Array of conformity scores, with shape `(3, n_samples)`.
        """
        _raise_error_if_previous_method_not_called(
            "conformity_scores",
            "conformalize",
            self._is_conformalized,
        )
        return cast(NDArray, self._mapie_quantile_regressor.conformity_scores)


class _MapieQuantileRegressor(_MapieRegressor):
    """
    Note to users: _MapieQuantileRegressor is now private, and may change at any time.
    Please use ConformalizedQuantileRegressor instead.
    See the v1 release notes for more information.

    This class implements the conformalized quantile regression strategy
    as proposed by Romano et al. (2019) to make conformal predictions.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with `fit` and `predict` methods).
        If `None`, estimator defaults to a `QuantileRegressor` instance.

        By default `"None"`.

    method: str
        Method to choose for prediction, in this case, the only valid method
        is the `"quantile"` method.

        By default `"quantile"`.

    cv: Optional[str]
        The cross-validation strategy for computing conformity scores.
        In theory a split method is implemented as it is needed to provide
        both a training and calibration set.

        By default `None`.

    alpha: float
        Between `0.0` and `1.0`, represents the risk level of the
        confidence interval.
        Lower `alpha` produce larger (more conservative) prediction
        intervals.
        `alpha` is the complement of the target coverage level.

        By default `0.1`.

    Attributes
    ----------
    valid_methods_: List[str]
        List of all valid methods.

    single_estimator_: RegressorMixin
        Estimator fitted on the whole training set.

    estimators_: List[RegressorMixin]
        - [0]: Estimator with quantile value of alpha/2
        - [1]: Estimator with quantile value of 1 - alpha/2
        - [2]: Estimator with quantile value of 0.5

    conformity_scores_: NDArray of shape (n_samples_train, 3)
        Conformity scores between `y_calib` and `y_pred`.

        - [:, 0]: for `y_calib` coming from prediction estimator
          with quantile of alpha/2
        - [:, 1]: for `y_calib` coming from prediction estimator
          with quantile of 1 - alpha/2
        - [:, 2]: maximum of those first two scores

    n_calib_samples: int
        Number of samples in the calibration dataset.

    References
    ----------
    Yaniv Romano, Evan Patterson and Emmanuel J. Candès.
    "Conformalized Quantile Regression"
    Advances in neural information processing systems 32 (2019).
    """

    valid_methods_ = ["quantile"]
    fit_attributes = [
        "estimators_",
        "conformity_scores",
        "n_calib_samples",
    ]

    quantile_estimator_params = {
        "GradientBoostingRegressor": {"loss_name": "loss", "alpha_name": "alpha"},
        "QuantileRegressor": {"loss_name": "quantile", "alpha_name": "quantile"},
        "HistGradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "quantile",
        },
        "LGBMRegressor": {"loss_name": "objective", "alpha_name": "alpha"},
    }

    def __init__(
        self,
        estimator: Optional[
            Union[RegressorMixin, Pipeline, List[Union[RegressorMixin, Pipeline]]]
        ] = None,
        method: str = "quantile",
        cv: Optional[str] = None,
        alpha: float = 0.1,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
        )
        self.cv = cv
        self.alpha = alpha
        self._is_fitted = False
        self._is_fitted = True if self.cv == "prefit" else False
        self._central_predictor = None

    @property
    def is_fitted(self):
        """Returns True if the estimator is fitted"""
        return self._is_fitted

    def _check_alpha(
        self,
        alpha: float = 0.1,
    ) -> NDArray:
        """
        Perform several checks on the alpha value and changes it from
        a float to an ArrayLike.

        Parameters
        ----------
        alpha : float
            Can only be a float value between `0.0` and `1.0`.
            Represent the risk level of the confidence interval.
            Lower alpha produce larger (more conservative) prediction
            intervals. Alpha is the complement of the target coverage level.

            By default `0.1`.

        Returns
        -------
        ArrayLike
            An ArrayLike of three values:

            - [0]: alpha value of alpha/2
            - [1]: alpha value of of 1 - alpha/2
            - [2]: alpha value of 0.5

        Raises
        ------
        ValueError
            If alpha is not a float.

        ValueError
            If the value of `alpha` is not between `0.0` and `1.0`.
        """
        if isinstance(alpha, float):
            if np.any(np.logical_or(alpha <= 0, alpha >= 1.0)):
                raise ValueError(
                    "Invalid confidence_level. Allowed values are between 0.0 and 1.0."
                )
            else:
                alpha_values = [alpha / 2, 1 - alpha / 2, 0.5]
                alpha_np = np.array(alpha_values)
        else:
            raise ValueError("Invalid confidence_level. Allowed values are float.")
        return alpha_np

    def _check_estimator(
        self,
        estimator: Optional[Union[RegressorMixin, Pipeline]] = None,
    ) -> Union[RegressorMixin, Pipeline]:
        """
        Perform several checks on the estimator to check if it has
        all the required specifications to be used with this methodology.
        The estimators that can be used in _MapieQuantileRegressor need to
        have a `fit` and `predict` attribute, but also need to allow
        a quantile loss and therefore also setting a quantile value.
        Note that there is a `TypedDict` to check which methods allow for
        quantile regression.

        Parameters
        ----------
        estimator : Optional[RegressorMixin], optional
            Estimator to check, by default `None`.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default `QuantileRegressor` instance
            with `solver` set to "highs".

        Raises
        ------
        ValueError
            If the estimator implements `fit` or `predict` methods.

        ValueError
            We check if it's a known estimator that does quantile regression
            according to the dictionnary set quantile_estimator_params.
            This dictionnary will need to be updated with the latest new
            available estimators.

        ValueError
            The estimator does not have the `"loss_name"` in its parameters
            and therefore can not be used as an estimator.

        ValueError
            There is no quantile `"loss_name"` and therefore this estimator
            can not be used as a `_MapieQuantileRegressor`.

        ValueError
            The parameter to set the alpha value does not exist in this
            estimator and therefore we cannot use it.
        """
        if estimator is None:
            return QuantileRegressor(
                solver="highs-ds",
                alpha=0.0,
            )
        _check_estimator_fit_predict(estimator)
        if isinstance(estimator, Pipeline):
            self._check_estimator(estimator[-1])
            return estimator
        else:
            name_estimator = estimator.__class__.__name__
            if name_estimator == "QuantileRegressor":
                return estimator
            else:
                if name_estimator in self.quantile_estimator_params:
                    param_estimator = estimator.get_params()
                    loss_name, alpha_name = self.quantile_estimator_params[
                        name_estimator
                    ].values()
                    if loss_name in param_estimator:
                        if param_estimator[loss_name] != "quantile":
                            raise ValueError(
                                "You need to set the loss/objective argument"
                                + " of your base model to `quantile`."
                            )
                        else:
                            if alpha_name in param_estimator:
                                return estimator
                            else:
                                raise ValueError(
                                    "The matching parameter `alpha_name` for"
                                    " estimator does not exist. "
                                    "Make sure you set it when initializing "
                                    "your estimator."
                                )
                    else:
                        raise ValueError(
                            "The matching parameter `loss_name` for"
                            + " estimator does not exist."
                        )
                else:
                    raise ValueError(
                        "The base model is not supported. \n"
                        "Give a base model among: \n"
                        f"{self.quantile_estimator_params.keys()} "
                        "Or, add your base model to" + " `quantile_estimator_params`."
                    )

    def _check_cv(self, cv: Optional[str] = None) -> str:
        """
        Check if cv argument is `None`, `"split"` or `"prefit"`.

        Parameters
        ----------
        cv : Optional[str], optional
           cv to check, by default `None`.

        Returns
        -------
        str
            cv itself or a default `"split"`.

        Raises
        ------
        ValueError
            Raises an error if the cv is anything else but the method
            `"split"` or `"prefit"`.
            Only the split method has been implemented.
        """
        if cv is None:
            return "split"
        if cv in ("split", "prefit"):
            return cv
        else:
            raise ValueError("Invalid cv method, only valid method is `split`.")

    def _train_calib_split(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
        if sample_weight is None:
            X_train, X_calib, y_train, y_calib = train_test_split(
                X,
                y,
                test_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            )
            sample_weight_train = sample_weight
        else:
            (
                X_train,
                X_calib,
                y_train,
                y_calib,
                sample_weight_train,
                _,
            ) = train_test_split(
                X,
                y,
                sample_weight,
                test_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            )
        return X_train, y_train, X_calib, y_calib, sample_weight_train

    def _check_prefit_params(
        self,
        estimator: List[Union[RegressorMixin, Pipeline]],
    ) -> None:
        """
        Check the parameters set for the specific case of prefit
        estimators.

        Parameters
        ----------
        estimator : List[Union[RegressorMixin, Pipeline]]
            List of three prefitted estimators that should have
            pre-defined quantile levels of alpha/2, 1 - alpha/2 and 0.5.

        Raises
        ------
        ValueError
            If a non-iterable variable is provided for estimator.

        ValueError
            If less or more than three models are defined.

        Warning
            If the alpha is defined, warns the user that it must be set
            accordingly with the prefit estimators.
        """
        if isinstance(estimator, Iterable) is False:
            raise ValueError("Estimator for prefit must be an iterable object.")
        if len(estimator) == 3:
            for est in estimator:
                _check_estimator_fit_predict(est)
                check_sklearn_user_model_is_fitted(est)
        else:
            raise ValueError(
                "You need to have provided 3 different estimators, they"
                " need to be preset with alpha values"
                "(alpha = 1 - confidence_level)"
                "in the following order [alpha/2, 1 - alpha/2, 0.5]."
            )

    def _initialize_fit_conformalize(self) -> None:
        self.cv = self._check_cv(cast(str, self.cv))
        self.alpha_np = self._check_alpha(self.alpha)
        self.estimators_: List[RegressorMixin] = []

    def _initialize_and_check_prefit_estimators(self) -> None:
        estimator = cast(List, self.estimator)
        self._check_prefit_params(estimator)
        self.estimators_ = list(estimator)
        self.single_estimator_ = self.estimators_[2]

    def _prepare_train_calib(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        X_calib: Optional[ArrayLike] = None,
        y_calib: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]]:
        """
        Handles the preparation of training and calibration datasets,
        including validation and splitting.
        Returns: X_train, y_train, X_calib, y_calib, sample_weight_train
        """
        self._check_parameters()
        random_state = check_random_state(random_state)
        X, y = indexable(X, y)

        if X_calib is None or y_calib is None:
            return self._train_calib_split(
                X, y, sample_weight, calib_size, random_state, shuffle, stratify
            )
        else:
            return X, y, X_calib, y_calib, sample_weight

    # Second function: Handles estimator fitting
    def _fit_estimators(
        self,
        X: ArrayLike,
        y: ArrayLike,
        **fit_params,
    ) -> None:
        """
        Fits the estimators with provided training data
        and stores them in self.estimators_.
        """
        sample_weight = fit_params.pop("sample_weight", None)
        checked_estimator = self._check_estimator(self.estimator)

        X, y = indexable(X, y)
        y = _check_y(y)

        sample_weight, X, y = _check_null_weight(sample_weight, X, y)

        if isinstance(checked_estimator, Pipeline):
            estimator = checked_estimator[-1]
        else:
            estimator = checked_estimator

        name_estimator = estimator.__class__.__name__
        alpha_name = self.quantile_estimator_params[name_estimator]["alpha_name"]
        for i, alpha_ in enumerate(self.alpha_np):
            cloned_estimator_ = clone(checked_estimator)
            params = {alpha_name: alpha_}
            if isinstance(checked_estimator, Pipeline):
                cloned_estimator_[-1].set_params(**params)
            else:
                cloned_estimator_.set_params(**params)

            self.estimators_.append(
                _fit_estimator(
                    cloned_estimator_,
                    X,
                    y,
                    sample_weight,
                    **fit_params,
                )
            )

        self._is_fitted = True

        self.single_estimator_ = self.estimators_[2]

    def conformalize(  # type: ignore[override]
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        # Parameter groups kept for compliance with superclass _MapieRegressor
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> _MapieRegressor:
        if self.cv == "prefit":
            self._initialize_and_check_prefit_estimators()

        X_calib, y_calib = cast(ArrayLike, X), cast(ArrayLike, y)
        X_calib, y_calib = indexable(X_calib, y_calib)
        y_calib = _check_y(y_calib)

        self.n_calib_samples = _num_samples(y_calib)
        _check_alpha_and_n_samples(self.alpha, self.n_calib_samples)

        y_calib_preds = np.full(shape=(3, self.n_calib_samples), fill_value=np.nan)

        for i, est in enumerate(self.estimators_):
            y_calib_preds[i] = est.predict(X_calib, **kwargs).ravel()

        self.conformity_scores = np.full(
            shape=(3, self.n_calib_samples), fill_value=np.nan
        )

        self.conformity_scores[0] = y_calib_preds[0] - y_calib
        self.conformity_scores[1] = y_calib - y_calib_preds[1]
        self.conformity_scores[2] = np.max(
            [self.conformity_scores[0], self.conformity_scores[1]], axis=0
        )
        return self

    @staticmethod
    def _check_defined_variables_predict(
        ensemble: bool,
        alpha: Union[float, Iterable[float], None],
    ) -> None:
        """
        Check that the parameters defined for the predict method
        of `_MapieQuantileRegressor` are correct.

        Parameters
        ----------
        ensemble: bool
            Ensemble has not been defined in predict and therefore should
            will not have any effects in this method.
        alpha: Optional[Union[float, Iterable[float]]]
            For `MapieQuantileRegresor` the alpha has to be defined
            directly in initial arguments of the class.

        Raises
        ------
        Warning
            If the ensemble value is defined in the predict function
            of `_MapieQuantileRegressor`.
        Warning
            If the alpha value is defined in the predict function
            of `_MapieQuantileRegressor`.

        Examples
        --------
        >>> import warnings
        >>> warnings.filterwarnings("error")
        >>> CQR = _MapieQuantileRegressor()
        >>> try:
        ...     CQR._check_defined_variables_predict(True, None)
        ... except Exception as exception:
        ...     print(exception)
        ...
        WARNING: ensemble is not utilized in `_MapieQuantileRegressor`.
        """

        if ensemble is True:
            warnings.warn(
                "WARNING: ensemble is not utilized in `_MapieQuantileRegressor`."
            )
        if alpha is not None:
            warnings.warn(
                "WARNING: Alpha should not be specified"
                + "in the prediction method\n"
                + "with conformalized quantile regression."
            )

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        symmetry: Optional[bool] = True,
        **predict_params,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation.
        Prediction Intervals for a given `alpha` are deduced from the
        quantile regression at the alpha values: alpha/2, 1 - (alpha/2)
        while adding a constant based uppon their residuals.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Ensemble has not been defined in predict and therefore should
            will not have any effects in this method.

        alpha: Optional[Union[float, Iterable[float]]]
            For `MapieQuantileRegresor` the alpha has to be defined
            directly in initial arguments of the class.

        symmetry: Optional[bool]
            Deciding factor to whether to find the quantile value for
            each residuals separatly or to use the maximum of the two
            combined.

        predict_params : dict
            Additional predict parameters.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if `alpha` is `None`.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if `alpha` is not `None`.
              - [:, 0, :]: Lower bound of the prediction interval.
              - [:, 1, :]: Upper bound of the prediction interval.
        """
        check_is_fitted(self)
        self._check_defined_variables_predict(ensemble, alpha)
        alpha = self.alpha if symmetry else self.alpha / 2
        _check_alpha_and_n_samples(alpha, self.n_calib_samples)

        n = self.n_calib_samples
        q = (1 - (alpha)) * (1 + (1 / n))

        y_preds = np.full(
            shape=(3, _num_samples(X)),
            fill_value=np.nan,
            dtype=float,
        )
        for i, est in enumerate(self.estimators_):
            y_preds[i] = est.predict(X, **predict_params)
        _check_lower_upper_bounds(y_preds[0], y_preds[1], y_preds[2])
        quantile: NDArray
        if symmetry:
            quantile = np.full(
                2, np.quantile(self.conformity_scores[2], q, method="higher")
            )
        else:
            quantile = np.array(
                [
                    np.quantile(self.conformity_scores[0], q, method="higher"),
                    np.quantile(self.conformity_scores[1], q, method="higher"),
                ]
            )
        y_pred_low = y_preds[0][:, np.newaxis] - quantile[0]
        y_pred_up = y_preds[1][:, np.newaxis] + quantile[1]
        _check_lower_upper_bounds(y_pred_low, y_pred_up, y_preds[2])
        return y_preds[2], np.stack([y_pred_low, y_pred_up], axis=1)
