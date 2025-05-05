from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple, Union, cast
from typing_extensions import Self

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_y, check_is_fitted, indexable

from numpy.typing import ArrayLike, NDArray
from mapie.conformity_scores import (BaseRegressionScore,
                                     ResidualNormalisedScore)
from mapie.conformity_scores.utils import (
    check_regression_conformity_score,
    check_and_select_conformity_score,
)
from mapie.estimator.regressor import EnsembleRegressor
from mapie.subsample import Subsample
from mapie.utils import (_check_alpha, _check_alpha_and_n_samples,
                         _check_cv, _check_estimator_fit_predict,
                         _check_n_features_in, _check_n_jobs, _check_null_weight,
                         _check_verbose, _get_effective_calibration_samples,
                         _check_predict_params)
from mapie.utils import (
    _transform_confidence_level_to_alpha_list,
    _check_if_param_in_allowed_values,
    _check_cv_not_string,
    _cast_point_predictions_to_ndarray,
    _cast_predictions_to_ndarray_tuple,
    _prepare_params,
    _prepare_fit_params_and_sample_weight,
    _raise_error_if_previous_method_not_called,
    _raise_error_if_method_already_called,
    _raise_error_if_fit_called_in_prefit_mode,
)


class SplitConformalRegressor:
    """
    Computes prediction intervals using the split conformal regression technique:

    1. The ``fit`` method (optional) fits the base regressor to the training data.
    2. The ``conformalize`` method estimates the uncertainty of the base regressor by
       computing conformity scores on the conformity set.
    3. The ``predict_interval`` method predicts points and intervals.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regressor used to predict points.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float is
        provided, it represents a single confidence level. If a list, multiple
        prediction intervals for each specified confidence level are returned.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The method used to compute conformity scores

        Valid options:

        - "absolute"
        - "gamma"
        - "residual_normalized"
        - Any subclass of BaseRegressionScore

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

        See :ref:`theoretical_description_conformity_scores`.
    prefit : bool, default=False
        If True, the base regressor must be fitted, and the ``fit``
        method must be skipped.

        If False, the base regressor will be fitted during the ``fit`` method.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level.
        Higher values increase the output details.

    Examples
    --------
    >>> from mapie.regression import SplitConformalRegressor
    >>> from mapie.utils import train_conformalize_test_split
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge

    >>> X, y = make_regression(n_samples=500, n_features=2, noise=1.0)
    >>> (
    ...     X_train, X_conformalize, X_test,
    ...     y_train, y_conformalize, y_test
    ... ) = train_conformalize_test_split(
    ...     X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
    ... )

    >>> mapie_regressor = SplitConformalRegressor(
    ...     estimator=Ridge(),
    ...     confidence_level=0.95,
    ...     prefit=False,
    ... ).fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)

    >>> predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)
    """

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        prefit: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        _check_estimator_fit_predict(estimator)
        self._estimator = estimator
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False
        self._conformity_score = check_and_select_conformity_score(
            conformity_score,
            BaseRegressionScore,
        )

        # Note to developers: to implement this v1 class without touching the
        # v0 backend, we're for now using a hack. We always set cv="prefit",
        # and we fit the estimator if needed. See the .fit method below.
        self._mapie_regressor = _MapieRegressor(
            estimator=self._estimator,
            method="base",
            cv="prefit",
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=self._conformity_score,
        )

        self._alphas = _transform_confidence_level_to_alpha_list(
            confidence_level
        )
        self._predict_params: dict = {}

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base regressor to the training data.

        Parameters
        ----------
        X_train : ArrayLike
            Training data features.

        y_train : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Parameters to pass to the ``fit`` method of the base regressor.

        Returns
        -------
        Self
            The fitted SplitConformalRegressor instance.
        """
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)

        cloned_estimator = clone(self._estimator)
        fit_params_ = _prepare_params(fit_params)
        cloned_estimator.fit(X_train, y_train, **fit_params_)
        self._mapie_regressor.estimator = cloned_estimator

        self._is_fitted = True
        return self

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Estimates the uncertainty of the base regressor by computing
        conformity scores on the conformity set.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Features of the conformity set.

        y_conformalize : ArrayLike
            Targets of the conformity set.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` method of the base regressor.
            These parameters will also be used in the ``predict_interval``
            and ``predict`` methods of this SplitConformalRegressor.

        Returns
        -------
        Self
            The conformalized SplitConformalRegressor instance.
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
        self._mapie_regressor.fit(
            X_conformalize,
            y_conformalize,
            predict_params=self._predict_params
        )

        self._is_conformalized = True
        return self

    def predict_interval(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predicts points (using the base regressor) and intervals.

        If several confidence levels were provided during initialisation, several
        intervals will be predicted for each sample. See the return signature.

        Parameters
        ----------
        X : ArrayLike
            Features

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the intervals width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape ``(n_samples,)``
            - Prediction intervals, of shape ``(n_samples, 2, n_confidence_levels)``
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval",
            "conformalize",
            self._is_conformalized,
        )
        predictions = self._mapie_regressor.predict(
            X,
            alpha=self._alphas,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds,
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
            Array of point predictions, with shape (n_samples,).
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "conformalize",
            self._is_conformalized,
        )
        predictions = self._mapie_regressor.predict(
            X,
            alpha=None,
            **self._predict_params
        )
        return _cast_point_predictions_to_ndarray(predictions)


class CrossConformalRegressor:
    """
    Computes prediction intervals using the cross conformal regression technique:

    1. The ``fit_conformalize`` method estimates the uncertainty of the base regressor
       in a cross-validation style. It fits the base regressor on folds of the dataset
       and computes conformity scores on the out-of-fold data.
    2. The ``predict_interval`` computes prediction points and intervals.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regressor used to predict points.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float is
        provided, it represents a single confidence level. If a list, multiple
        prediction intervals for each specified confidence level are returned.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The method used to compute conformity scores
        Valid options:

        - "absolute"
        - "gamma"
        - The corresponding subclasses of BaseRegressionScore

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

        See :ref:`theoretical_description_conformity_scores`.

    method : str, default="plus"
        The method used to compute prediction intervals. Options are:

        - "base": Based on the conformity scores from each fold.
        - "plus": Based on the conformity scores from each fold and
          the test set predictions.
        - "minmax": Based on the conformity scores from each fold and
          the test set predictions, using the minimum and maximum among
          each fold models.

    cv : Union[int, BaseCrossValidator], default=5
        The cross-validator used to compute conformity scores.
        Valid options:

        - integer, to specify the number of folds
        - any ``sklearn.model_selection.BaseCrossValidator`` suitable for
          regression, or a custom cross-validator inheriting from it.

        Main variants in the cross conformal setting are:

        - ``sklearn.model_selection.KFold`` (vanilla cross conformal)
        - ``sklearn.model_selection.LeaveOneOut`` (jackknife)

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the
        output details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Examples
    --------
    >>> from mapie.regression import CrossConformalRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import Ridge

    >>> X_full, y_full = make_regression(n_samples=500,n_features=2,noise=1.0)
    >>> X, X_test, y, y_test = train_test_split(X_full, y_full)

    >>> mapie_regressor = CrossConformalRegressor(
    ...     estimator=Ridge(),
    ...     confidence_level=0.95,
    ...     cv=10
    ... ).fit_conformalize(X, y)

    >>> predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)
    """

    _VALID_METHODS = ["base", "plus", "minmax"]

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        method: str = "plus",
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        _check_if_param_in_allowed_values(
            method,
            "method",
            CrossConformalRegressor._VALID_METHODS
        )
        _check_cv_not_string(cv)

        self._mapie_regressor = _MapieRegressor(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=check_and_select_conformity_score(
                conformity_score,
                BaseRegressionScore,
            ),
            random_state=random_state,
        )

        self._alphas = _transform_confidence_level_to_alpha_list(
            confidence_level
        )
        self.is_fitted_and_conformalized = False

        self._predict_params: dict = {}

    def fit_conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Estimates the uncertainty of the base regressor in a cross-validation style:
        fits the base regressor on different folds of the dataset
        and computes conformity scores on the corresponding out-of-fold data.

        Parameters
        ----------
        X : ArrayLike
            Features

        y : ArrayLike
            Targets

        groups: Optional[ArrayLike] of shape (n_samples,), default=None
            Groups to pass to the cross-validator.

        fit_params : Optional[dict], default=None
            Parameters to pass to the ``fit`` method of the base regressor.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` method of the base regressor.
            These parameters will also be used in the ``predict_interval``
            and ``predict`` methods of this CrossConformalRegressor.

        Returns
        -------
        Self
            This CrossConformalRegressor instance, fitted and conformalized.
        """
        _raise_error_if_method_already_called(
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        fit_params_, sample_weight = _prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._predict_params = _prepare_params(predict_params)
        self._mapie_regressor.fit(
            X,
            y,
            sample_weight,
            groups,
            fit_params=fit_params_,
            predict_params=self._predict_params
        )

        self.is_fitted_and_conformalized = True
        return self

    def predict_interval(
        self,
        X: ArrayLike,
        aggregate_predictions: Optional[str] = "mean",
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predicts points and intervals.

        If several confidence levels were provided during initialisation, several
        intervals will be predicted for each sample. See the return signature.

        By default, points are predicted using an aggregation.
        See the ``ensemble`` parameter.

        Parameters
        ----------
        X : ArrayLike
            Features

        aggregate_predictions : Optional[str], default="mean"
            The method to predict a point. Options:

            - None: a point is predicted using the regressor trained on the entire data
            - "mean": Averages the predictions of the regressors trained on each
              cross-validation fold
            - "median": Aggregates (using median) the predictions of the regressors
              trained on each cross-validation fold

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the interval width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape ``(n_samples,)``
            - Prediction intervals, of shape ``(n_samples, 2, n_confidence_levels)``
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        ensemble = self._set_aggregate_predictions_and_return_ensemble(
            aggregate_predictions
        )
        predictions = self._mapie_regressor.predict(
            X,
            alpha=self._alphas,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds,
            ensemble=ensemble,
            **self._predict_params,
        )
        return _cast_predictions_to_ndarray_tuple(predictions)

    def predict(
        self,
        X: ArrayLike,
        aggregate_predictions: Optional[str] = "mean",
    ) -> NDArray:
        """
        Predicts points.

        By default, points are predicted using an aggregation.
        See the ``ensemble`` parameter.

        Parameters
        ----------
        X : ArrayLike
            Features

        aggregate_predictions : Optional[str], default="mean"
            The method to predict a point. Options:

            - None: a point is predicted using the regressor trained on the entire data
            - "mean": Averages the predictions of the regressors trained on each
              cross-validation fold
            - "median": Aggregates (using median) the predictions of the regressors
              trained on each cross-validation fold

        Returns
        -------
        NDArray
            Array of point predictions, with shape ``(n_samples,)``.
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        ensemble = self._set_aggregate_predictions_and_return_ensemble(
            aggregate_predictions
        )
        predictions = self._mapie_regressor.predict(
            X, alpha=None, ensemble=ensemble, **self._predict_params,
        )
        return _cast_point_predictions_to_ndarray(predictions)

    def _set_aggregate_predictions_and_return_ensemble(
        self, aggregate_predictions: Optional[str]
    ) -> bool:
        if not aggregate_predictions:
            ensemble = False
        else:
            ensemble = True
            self._mapie_regressor._check_agg_function(aggregate_predictions)
            # A hack here, to allow choosing the aggregation function at prediction time
            self._mapie_regressor.agg_function = aggregate_predictions
        return ensemble


class JackknifeAfterBootstrapRegressor:
    """
    Computes prediction intervals using the jackknife-after-bootstrap technique:

    1. The ``fit_conformalize`` method estimates the uncertainty of the base regressor
       using bootstrap sampling. It fits the base regressor on samples of the dataset
       and computes conformity scores on the out-of-sample data.
    2. The ``predict_interval`` computes prediction points and intervals.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regressor used to predict points.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float is
        provided, it represents a single confidence level. If a list, multiple
        prediction intervals for each specified confidence level are returned.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The method used to compute conformity scores

        Valid options:

        - "absolute"
        - "gamma"
        - The corresponding subclasses of BaseRegressionScore

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

        See :ref:``theoretical_description_conformity_scores``.

    method : str, default="plus"

        The method used to compute prediction intervals. Options are:

        - "plus": Based on the conformity scores from each bootstrap sample and
          the testing prediction.
        - "minmax": Based on the minimum and maximum conformity scores from
          each bootstrap sample.

        Note: The "base" method is not mentioned in the conformal inference
        literature for Jackknife after bootstrap strategies, hence not provided
        here.

    resampling : Union[int, Subsample], default=30
        Number of bootstrap resamples or an instance of ``Subsample`` for
        custom sampling strategy.

    aggregation_method : str, default="mean"
        Aggregation method for predictions across bootstrap samples. Options:

        - "mean"
        - "median"

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the output
        details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Examples
    --------
    >>> from mapie.regression import JackknifeAfterBootstrapRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import Ridge

    >>> X_full, y_full = make_regression(n_samples=500,n_features=2,noise=1.0)
    >>> X, X_test, y, y_test = train_test_split(X_full, y_full)

    >>> mapie_regressor = JackknifeAfterBootstrapRegressor(
    ...     estimator=Ridge(),
    ...     confidence_level=0.95,
    ...     resampling=25,
    ... ).fit_conformalize(X, y)

    >>> predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)
    """

    _VALID_METHODS = ["plus", "minmax"]
    _VALID_AGGREGATION_METHODS = ["mean", "median"]

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        method: str = "plus",
        resampling: Union[int, Subsample] = 30,
        aggregation_method: str = "mean",
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        _check_if_param_in_allowed_values(
            method,
            "method",
            JackknifeAfterBootstrapRegressor._VALID_METHODS
        )
        _check_if_param_in_allowed_values(
            aggregation_method,
            "aggregation_method",
            JackknifeAfterBootstrapRegressor._VALID_AGGREGATION_METHODS
        )

        cv = self._check_and_convert_resampling_to_cv(resampling)

        self._mapie_regressor = _MapieRegressor(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            agg_function=aggregation_method,
            conformity_score=check_and_select_conformity_score(
                conformity_score,
                BaseRegressionScore,
            ),
            random_state=random_state,
        )

        self._alphas = _transform_confidence_level_to_alpha_list(
            confidence_level
        )

        self.is_fitted_and_conformalized = False
        self._predict_params: dict = {}

    def fit_conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Estimates the uncertainty of the base regressor using bootstrap sampling:
        fits the base regressor on (potentially overlapping) samples of the dataset,
        and computes conformity scores on the corresponding out of samples data.

        Parameters
        ----------
        X : ArrayLike
            Features. Must be the same X used in .fit

        y : ArrayLike
            Targets. Must be the same y used in .fit

        fit_params : Optional[dict], default=None
            Parameters to pass to the ``fit`` method of the base regressor.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` method of the base regressor.
            These parameters will also be used in the ``predict_interval``
            and ``predict`` methods of this JackknifeAfterBootstrapRegressor.

        Returns
        -------
        Self
            This JackknifeAfterBootstrapRegressor instance, fitted and conformalized.
        """
        _raise_error_if_method_already_called(
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        fit_params_, sample_weight = _prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._predict_params = _prepare_params(predict_params)
        self._mapie_regressor.fit(
            X,
            y,
            sample_weight,
            fit_params=fit_params_,
            predict_params=self._predict_params,
        )

        self.is_fitted_and_conformalized = True
        return self

    def predict_interval(
        self,
        X: ArrayLike,
        ensemble: bool = True,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predicts points and intervals.

        If several confidence levels were provided during initialisation, several
        intervals will be predicted for each sample. See the return signature.

        By default, points are predicted using an aggregation.
        See the ``ensemble`` parameter.

        Parameters
        ----------
        X : ArrayLike
            Test data for prediction intervals.

        ensemble : bool, default=True
            If True, a predicted point is an aggregation of the predictions of the
            regressors trained on each bootstrap samples. This aggregation depends on
            the ``aggregation_method`` provided during initialisation.

            If False, a point is predicted using the regressor trained on the entire
            data

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the interval width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape ``(n_samples,)``
            - Prediction intervals, of shape ``(n_samples, 2, n_confidence_levels)``
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        predictions = self._mapie_regressor.predict(
            X,
            alpha=self._alphas,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds,
            ensemble=ensemble,
            **self._predict_params,
        )
        return _cast_predictions_to_ndarray_tuple(predictions)

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = True,
    ) -> NDArray:
        """
        Predicts points.

        By default, points are predicted using an aggregation.
        See the ``ensemble`` parameter.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        ensemble : bool, default=True
            If True, a predicted point is an aggregation of the predictions of the
            regressors trained on each bootstrap samples. This aggregation depends on
            the ``aggregation_method`` provided during initialisation.
            If False, a point is predicted using the regressor trained on the entire
            data

        Returns
        -------
        NDArray
            Array of point predictions, with shape ``(n_samples,)``.
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        predictions = self._mapie_regressor.predict(
            X, alpha=None, ensemble=ensemble, **self._predict_params,
        )
        return _cast_point_predictions_to_ndarray(predictions)

    @staticmethod
    def _check_and_convert_resampling_to_cv(
        resampling: Union[int, Subsample]
    ) -> Subsample:
        if isinstance(resampling, int):
            cv = Subsample(n_resamplings=resampling)
        elif isinstance(resampling, Subsample):
            cv = resampling
        else:
            raise ValueError(
                "resampling must be an integer or a Subsample instance"
            )
        return cv


class _MapieRegressor(RegressorMixin, BaseEstimator):
    """
    Note to users: _MapieRegressor is now private, and may change at any time.
    Please use CrossConformalRegressor, CrossConformalRegressor or
    JackknifeAfterBootstrapRegressor instead.
    See the v1 migration guide for more information.

    Prediction interval with out-of-fold conformity scores.

    This class implements the jackknife+ strategy and its variations
    for estimating prediction intervals on single-output data. The
    idea is to evaluate out-of-fold conformity scores (signed residuals,
    absolute residuals, residuals normalized by the predicted mean...)
    on hold-out validation sets and to deduce valid confidence intervals
    with strong theoretical guarantees.

    Parameters
    ----------
    estimator: Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

        By default ``None``.

    method: str
        Method to choose for prediction interval estimates.
        Choose among:

        - ``"naive"``, based on training set conformity scores,
        - ``"base"``, based on validation sets conformity scores,
        - ``"plus"``, based on validation conformity scores and
          testing predictions,
        - ``"minmax"``, based on validation conformity scores and
          testing predictions (min/max among cross-validation clones).

        By default ``"plus"``.

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing conformity scores.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to ``-1``, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
          - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
          - ``sklearn.model_selection.KFold`` (cross-validation),
          - ``subsample.Subsample`` object (bootstrap).
        - ``"split"``, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
          ``method`` parameter is set to ``"base"``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already,
          and the ``method`` parameter is set to ``"base"``.
          All data provided in the ``fit`` method is then used
          for computing conformity scores only.
          At prediction time, quantiles of these conformity scores are used
          to provide a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          conformity scores estimate are disjoint.

        By default ``None``.

    test_size: Optional[Union[int, float]]
        If ``float``, should be between ``0.0`` and ``1.0`` and represent the
        proportion of the dataset to include in the test split. If ``int``,
        represents the absolute number of test samples. If ``None``,
        it will be set to ``0.1``.

        If cv is not ``"split"``, ``test_size`` is ignored.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For ``n_jobs`` below ``-1``, ``(n_cpus + 1 - n_jobs)`` are used.
        ``None`` is a marker for `unset` that will be interpreted as
        ``n_jobs=1`` (sequential execution).

        By default ``None``.

    agg_function: Optional[str]
        Determines how to aggregate predictions from perturbed models, both at
        training and prediction time.

        If ``None``, it is ignored except if ``cv`` class is ``Subsample``,
        in which case an error is raised.
        If ``"mean"`` or ``"median"``, returns the mean or median of the
        predictions computed from the out-of-folds models.
        Note: if you plan to set the ``ensemble`` argument to ``True`` in the
        ``predict`` method, you have to specify an aggregation function.
        Otherwise an error would be raised.

        The Jackknife+ interval can be interpreted as an interval around the
        median prediction, and is guaranteed to lie inside the interval,
        unlike the single estimator predictions.

        When the cross-validation strategy is ``Subsample`` (i.e. for the
        Jackknife+-after-Bootstrap method), this function is also used to
        aggregate the training set in-sample predictions.

        If ``cv`` is ``"prefit"`` or ``"split"``, ``agg_function`` is ignored.

        By default ``"mean"``.

    verbose: int
        The verbosity level, used with joblib for multiprocessing.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    conformity_score: Optional[BaseRegressionScore]
        BaseRegressionScore instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteConformityScore`` conformity
          score
        - BaseRegressionScore: any ``BaseRegressionScore`` class

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random sampling.
        Pass an int for reproducible output across multiple function calls.

        By default ``None``.

    Attributes
    ----------
    valid_methods_: List[str]
        List of all valid methods.

    estimator_: EnsembleRegressor
        Sklearn estimator that handle all that is related to the estimator.

    conformity_score_function_: BaseRegressionScore
        Score function that handle all that is related to conformity scores.

    conformity_scores_: ArrayLike of shape (n_samples_train,)
        Conformity scores between ``y_train`` and ``y_pred``.

    n_features_in_: int
        Number of features passed to the ``fit`` method.

    References
    ----------
    Rina Foygel Barber, Emmanuel J. Candès,
    Aaditya Ramdas, and Ryan J. Tibshirani.
    "Predictive inference with the jackknife+."
    Ann. Statist., 49(1):486-507, February 2021.

    Byol Kim, Chen Xu, and Rina Foygel Barber.
    "Predictive Inference Is Free with the Jackknife+-after-Bootstrap."
    34th Conference on Neural Information Processing Systems (NeurIPS 2020).

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression.regression import _MapieRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> X_toy = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> clf = LinearRegression().fit(X_toy, y_toy)
    >>> mapie_reg = _MapieRegressor(estimator=clf, cv="prefit")
    >>> mapie_reg = mapie_reg.fit(X_toy, y_toy)
    >>> y_pred, y_pis = mapie_reg.predict(X_toy, alpha=0.5)
    >>> print(y_pis[:, :, 0])
    [[ 4.95714286  5.61428571]
     [ 6.84285714  7.5       ]
     [ 8.72857143  9.38571429]
     [10.61428571 11.27142857]
     [12.5        13.15714286]
     [14.38571429 15.04285714]]
    >>> print(y_pred)
    [ 5.28571429  7.17142857  9.05714286 10.94285714 12.82857143 14.71428571]
    """

    cv_need_agg_function_ = ["Subsample"]
    no_agg_cv_ = ["prefit", "split"]
    valid_methods_ = ["naive", "base", "plus", "minmax"]
    no_agg_methods_ = ["naive", "base"]
    valid_agg_functions_ = [None, "median", "mean"]
    ensemble_agg_functions_ = ["median", "mean"]
    default_sym_ = True
    fit_attributes = [
        "estimator_",
        "conformity_scores_",
        "conformity_score_function_",
        "n_features_in_",
    ]

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "plus",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        test_size: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[BaseRegressionScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.agg_function = agg_function
        self.verbose = verbose
        self.conformity_score = conformity_score
        self.random_state = random_state

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        self._check_method(self.method)
        _check_n_jobs(self.n_jobs)
        _check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_method(
        self, method: str
    ) -> str:
        """
        Check if ``method`` is correct.

        Parameters
        ----------
        method: str
            Method's name to check.

        Returns
        -------
        str
            ``method`` itself.

        Raises
        ------
        ValueError
            If ``method`` is not in ``self.valid_methods_``.
        """
        if method not in self.valid_methods_:
            raise ValueError(
                f"Invalid method. Allowed values are {self.valid_methods_}."
            )
        else:
            return method

    def _check_agg_function(
        self, agg_function: Optional[str] = None
    ) -> Optional[str]:
        """
        Check if ``agg_function`` is correct, and consistent with other
        arguments.

        Parameters
        ----------
        agg_function: Optional[str]
            Aggregation function's name to check, by default ``None``.

        Returns
        -------
        str
            ``agg_function`` itself or ``"mean"``.

        Raises
        ------
        ValueError
            If ``agg_function`` is not in [``None``, ``"mean"``, ``"median"``],
            or is ``None`` while cv class is in ``cv_need_agg_function_``.
        """
        if agg_function not in self.valid_agg_functions_:
            raise ValueError(
                "Invalid aggregation function. "
                f"Allowed values are '{self.valid_agg_functions_}'."
            )
        elif (agg_function is None) and (
            type(self.cv).__name__ in self.cv_need_agg_function_
        ):
            raise ValueError(
                "You need to specify an aggregation function."
            )
        elif agg_function is not None:
            return agg_function
        else:
            return "mean"

    def _check_estimator(
        self, estimator: Optional[RegressorMixin] = None
    ) -> RegressorMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LinearRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator: Optional[RegressorMixin]
            Estimator to check, by default ``None``.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``LinearRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no ``fit`` nor ``predict`` methods.

        NotFittedError
            If the estimator is not fitted
            and ``cv`` attribute is ``"prefit"``.
        """
        if estimator is None:
            return LinearRegression()
        else:
            _check_estimator_fit_predict(estimator)
            if self.cv == "prefit":
                if isinstance(estimator, Pipeline):
                    check_is_fitted(estimator[-1])
                else:
                    check_is_fitted(estimator)
            return estimator

    def _check_ensemble(
        self, ensemble: bool,
    ) -> None:
        """
        Check if ``ensemble`` is ``False`` and if ``self.agg_function``
        is ``None``. Else raise error.

        Parameters
        ----------
        ensemble: bool
            ``ensemble`` argument to check the coherennce with
            ``self.agg_function``.

        Raises
        ------
        ValueError
            If ``ensemble`` is ``True`` and ``self.agg_function`` is ``None``.
        """
        if ensemble and (self.agg_function is None):
            raise ValueError(
                "The aggregation function has to be in "
                f"{self.ensemble_agg_functions_}."
            )

    def _check_fit_parameters(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None
    ):
        """
        Perform several checks on class parameters.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y: ArrayLike
            Target values.

        sample_weight: Optional[NDArray] of shape (n_samples,)
            Non-null sample weights.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            By default ``None``.

        Raises
        ------
        ValueError
            If conformity score is FittedResidualNormalizing score and method
            is neither ``"prefit"`` or ``"split"``.

        ValueError
            If ``cv`` is `"prefit"`` or ``"split"`` and ``method`` is not
            ``"base"``.
        """
        # Checking
        self._check_parameters()
        cv = _check_cv(
            self.cv, test_size=self.test_size, random_state=self.random_state
        )
        if self.cv in ["split", "prefit"] and \
                self.method in ["naive", "plus", "minmax"]:
            self.method = "base"
        estimator = self._check_estimator(self.estimator)
        agg_function = self._check_agg_function(self.agg_function)
        cs_estimator = check_regression_conformity_score(
            self.conformity_score, self.default_sym_
        )
        if isinstance(cs_estimator, ResidualNormalisedScore) and \
           self.cv not in ["split", "prefit"]:
            raise ValueError(
                "The ResidualNormalisedScore can be used only with "
                "``SplitConformalRegressor``"
            )

        X, y = indexable(X, y)
        y = _check_y(y)
        sample_weight, X, y = _check_null_weight(sample_weight, X, y)
        self.n_features_in_ = _check_n_features_in(X)

        # Casting
        cv = cast(BaseCrossValidator, cv)
        estimator = cast(RegressorMixin, estimator)
        cs_estimator = cast(BaseRegressionScore, cs_estimator)
        agg_function = cast(Optional[str], agg_function)
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        sample_weight = cast(Optional[NDArray], sample_weight)
        groups = cast(Optional[NDArray], groups)

        return (
            estimator, cs_estimator, agg_function, cv,
            X, y, sample_weight, groups
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any
    ) -> _MapieRegressor:
        """
        Fit estimator and compute conformity scores used for
        prediction intervals.

        All the types of estimator (single or cross validated ones) are
        encapsulated under EnsembleRegressor.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no conformity scores.
            If weights are non-uniform,
            conformity scores are still uniformly weighted.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            By default ``None``.

        kwargs : dict
            Additional fit and predict parameters.

        Returns
        -------
        _MapieRegressor
            The model itself.
        """

        X, y, sample_weight, groups = self.init_fit(
            X, y, sample_weight, groups, **kwargs
        )

        self.fit_estimator(X, y, sample_weight, groups)
        self.conformalize(X, y, sample_weight, groups, **kwargs)

        return self

    def init_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any
    ):

        self._fit_params = kwargs.pop('fit_params', {})

        # Checks
        (estimator,
         self.conformity_score_function_,
         agg_function,
         cv,
         X,
         y,
         sample_weight,
         groups) = self._check_fit_parameters(X, y, sample_weight, groups)

        self.estimator_ = EnsembleRegressor(
            estimator,
            self.method,
            cv,
            agg_function,
            self.n_jobs,
            self.test_size,
            self.verbose
        )

        return (
            X, y, sample_weight, groups
        )

    def fit_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> _MapieRegressor:

        self.estimator_.fit_single_estimator(
            X,
            y,
            sample_weight=sample_weight,
            groups=groups,
            **self._fit_params
        )

        return self

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any
    ) -> _MapieRegressor:

        predict_params = kwargs.pop('predict_params', {})
        self._predict_params = len(predict_params) > 0

        self.estimator_.fit_multi_estimators(
            X,
            y,
            sample_weight,
            groups,
            **self._fit_params
        )

        # Predict on calibration data
        y_pred = self.estimator_.predict_calib(
                X, y=y, groups=groups, **predict_params
        )

        # Compute the conformity scores (manage jk-ab case)
        self.conformity_scores_ = \
            self.conformity_score_function_.get_conformity_scores(
                y, y_pred, X=X
            )

        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        **predict_params
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        Conformity scores from the training set and predictions
        from the model clones are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of conformity scores (``naive`` and ``base`` methods),
        - quantiles of (predictions +/- conformity scores) (``plus`` method),
        - quantiles of (max/min(predictions) +/- conformity scores)
          (``minmax`` method).

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

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between ``0`` and ``1``, represents the uncertainty of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        optimize_beta: bool
            Whether to optimize the PIs' width or not.

            By default ``False``.

        allow_infinite_bounds: bool
            Allow infinite prediction intervals to be produced.

            By default ``False``.

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
        # Checks
        if hasattr(self, '_predict_params'):
            _check_predict_params(self._predict_params, predict_params, self.cv)
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha = cast(Optional[NDArray], _check_alpha(alpha))

        # If alpha is None, predict the target without confidence intervals
        if alpha is None:
            y_pred = self.estimator_.predict(
                X, ensemble, return_multi_pred=False, **predict_params
            )
            return np.array(y_pred)

        else:
            # Check alpha and the number of effective calibration samples
            alpha_np = cast(NDArray, alpha)
            if not allow_infinite_bounds:
                n = _get_effective_calibration_samples(
                    self.conformity_scores_,
                    self.conformity_score_function_.sym
                )
                _check_alpha_and_n_samples(alpha_np, n)

            # Predict the target with confidence intervals
            outputs = self.conformity_score_function_.predict_set(
                X, alpha_np,
                estimator=self.estimator_,
                conformity_scores=self.conformity_scores_,
                ensemble=ensemble,
                method=self.method,
                optimize_beta=optimize_beta,
                allow_infinite_bounds=allow_infinite_bounds
            )
            y_pred, y_pred_low, y_pred_up = outputs

            return np.array(y_pred), np.stack([y_pred_low, y_pred_up], axis=1)
