from typing import Optional, Union, List, Tuple, Iterable
from typing_extensions import Self

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from mapie.subsample import Subsample
from numpy.typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseRegressionScore
from mapie.regression import MapieRegressor, MapieQuantileRegressor
from mapie.utils import check_estimator_fit_predict
from mapie_v1.conformity_scores._utils import check_and_select_conformity_score
from mapie_v1.utils import (
    transform_confidence_level_to_alpha_list,
    check_if_param_in_allowed_values,
    check_cv_not_string,
    cast_point_predictions_to_ndarray,
    cast_predictions_to_ndarray_tuple,
    prepare_params,
    prepare_fit_params_and_sample_weight,
    raise_error_if_previous_method_not_called,
    raise_error_if_method_already_called,
    raise_error_if_fit_called_in_prefit_mode, transform_confidence_level_to_alpha,
)


class SplitConformalRegressor:
    """
    Computes prediction intervals using the split conformal regression technique:

    1. The `fit` method (optional) fits the base regressor to the training data.
    2. The `conformalize` method estimates the uncertainty of the base regressor by
       computing conformity scores on the conformity set.
    3. The `predict_interval` computes prediction points and intervals.

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
        If True, the base regressor must be fitted, and the `fit`
        method must be skipped.

        If False, the base regressor will be fitted during the `fit` method.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level.
        Higher values increase the output details.

    Examples
    --------
    >>> from mapie_v1.regression import SplitConformalRegressor
    >>> from mapie_v1.utils import train_conformalize_test_split
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
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
        check_estimator_fit_predict(estimator)
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
        self._mapie_regressor = MapieRegressor(
            estimator=self._estimator,
            method="base",
            cv="prefit",
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=self._conformity_score,
        )

        self._alphas = transform_confidence_level_to_alpha_list(
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
            Parameters to pass to the `fit` method of the base regressor.

        Returns
        -------
        Self
            The fitted SplitConformalRegressor instance.
        """
        raise_error_if_fit_called_in_prefit_mode(self._prefit)
        raise_error_if_method_already_called("fit", self._is_fitted)

        cloned_estimator = clone(self._estimator)
        fit_params_ = prepare_params(fit_params)
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
            Parameters to pass to the `predict` method of the base regressor.
            These parameters will also be used in the `predict_interval`
            and `predict` methods of this SplitConformalRegressor.

        Returns
        -------
        Self
            The conformalized SplitConformalRegressor instance.
        """
        raise_error_if_previous_method_not_called(
            "conformalize",
            "fit",
            self._is_fitted,
        )
        raise_error_if_method_already_called(
            "conformalize",
            self._is_conformalized,
        )

        self._predict_params = prepare_params(predict_params)
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
        Predicts points and intervals.

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

            - Prediction points, of shape `(n_samples,)`
            - Prediction intervals, of shape `(n_samples, 2, n_confidence_levels)`
        """
        raise_error_if_previous_method_not_called(
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
        return cast_predictions_to_ndarray_tuple(predictions)

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
        raise_error_if_previous_method_not_called(
            "predict",
            "conformalize",
            self._is_conformalized,
        )
        predictions = self._mapie_regressor.predict(
            X,
            alpha=None,
            **self._predict_params
        )
        return cast_point_predictions_to_ndarray(predictions)


class CrossConformalRegressor:
    """
    Computes prediction intervals using the cross conformal regression technique:

    1. The `fit_conformalize` method estimates the uncertainty of the base regressor in
       a cross-validation style. It fits the base regressor on folds of the dataset and
       computes conformity scores on the out-of-fold data.
    2. The `predict_interval` computes prediction points and intervals.

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
    >>> from mapie_v1.regression import CrossConformalRegressor
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
        check_if_param_in_allowed_values(
            method,
            "method",
            CrossConformalRegressor._VALID_METHODS
        )
        check_cv_not_string(cv)

        self._mapie_regressor = MapieRegressor(
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

        self._alphas = transform_confidence_level_to_alpha_list(
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
            Parameters to pass to the `fit` method of the base regressor.

        predict_params : Optional[dict], default=None
            Parameters to pass to the `predict` method of the base regressor.
            These parameters will also be used in the `predict_interval`
            and `predict` methods of this CrossConformalRegressor.

        Returns
        -------
        Self
            The fitted CrossConformalRegressor instance.
        """
        raise_error_if_method_already_called(
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        fit_params_, sample_weight = prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._predict_params = prepare_params(predict_params)
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
        See the `ensemble` parameter.

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

            - Prediction points, of shape `(n_samples,)`
            - Prediction intervals, of shape `(n_samples, 2, n_confidence_levels)`
        """
        raise_error_if_previous_method_not_called(
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
        return cast_predictions_to_ndarray_tuple(predictions)

    def predict(
        self,
        X: ArrayLike,
        aggregate_predictions: Optional[str] = "mean",
    ) -> NDArray:
        """
        Predicts points.

        By default, points are predicted using an aggregation.
        See the `ensemble` parameter.

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
            Array of point predictions, with shape `(n_samples,)`.
        """
        raise_error_if_previous_method_not_called(
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
        return cast_point_predictions_to_ndarray(predictions)

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

    1. The `fit_conformalize` method estimates the uncertainty of the base regressor
       using bootstrap sampling. It fits the base regressor on samples of the dataset
       and computes conformity scores on the out-of-sample data.
    2. The `predict_interval` computes prediction points and intervals.

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

        - "plus": Based on the conformity scores from each bootstrap sample and
          the testing prediction.
        - "minmax": Based on the minimum and maximum conformity scores from
          each bootstrap sample.

        Note: The "base" method is not mentioned in the conformal inference
        literature for Jackknife after bootstrap strategies, hence not provided
        here.

    resampling : Union[int, Subsample], default=30
        Number of bootstrap resamples or an instance of `Subsample` for
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
    >>> from mapie_v1.regression import JackknifeAfterBootstrapRegressor
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
        check_if_param_in_allowed_values(
            method,
            "method",
            JackknifeAfterBootstrapRegressor._VALID_METHODS
        )
        check_if_param_in_allowed_values(
            aggregation_method,
            "aggregation_method",
            JackknifeAfterBootstrapRegressor._VALID_AGGREGATION_METHODS
        )

        if isinstance(resampling, int):
            cv = Subsample(n_resamplings=resampling)
        elif isinstance(resampling, Subsample):
            cv = resampling
        else:
            raise ValueError(
                "resampling must be an integer or a Subsample instance"
            )

        self._mapie_regressor = MapieRegressor(
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

        self._alphas = transform_confidence_level_to_alpha_list(
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
            Parameters to pass to the `fit` method of the base regressor.

        predict_params : Optional[dict], default=None
            Parameters to pass to the `predict` method of the base regressor.
            These parameters will also be used in the `predict_interval`
            and `predict` methods of this JackknifeAfterBootstrapRegressor.

        Returns
        -------
        Self
            The JackknifeAfterBootstrapRegressor instance.
        """
        raise_error_if_method_already_called(
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        fit_params_, sample_weight = prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._predict_params = prepare_params(predict_params)
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
        See the `ensemble` parameter.

        Parameters
        ----------
        X : ArrayLike
            Test data for prediction intervals.

        ensemble : bool, default=True
            If True, a predicted point is an aggregation of the predictions of the
            regressors trained on each bootstrap samples. This aggregation depends on
            the `aggregation_method` provided during initialisation.

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

            - Prediction points, of shape `(n_samples,)`
            - Prediction intervals, of shape `(n_samples, 2, n_confidence_levels)`
        """
        raise_error_if_previous_method_not_called(
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
        return cast_predictions_to_ndarray_tuple(predictions)

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = True,
    ) -> NDArray:
        """
        Predicts points.

        By default, points are predicted using an aggregation.
        See the `ensemble` parameter.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        ensemble : bool, default=True
            If True, a predicted point is an aggregation of the predictions of the
            regressors trained on each bootstrap samples. This aggregation depends on
            the `aggregation_method` provided during initialisation.
            If False, a point is predicted using the regressor trained on the entire
            data

        Returns
        -------
        NDArray
            Array of point predictions, with shape `(n_samples,)`.
        """
        raise_error_if_previous_method_not_called(
            "predict",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        predictions = self._mapie_regressor.predict(
            X, alpha=None, ensemble=ensemble, **self._predict_params,
        )
        return cast_point_predictions_to_ndarray(predictions)


class ConformalizedQuantileRegressor:
    """
    Computes prediction intervals using the conformalized quantile regression technique:

    1. The `fit` method fits three models to the training data using the provided
       regressor: a model to predict the target, and models to predict upper
       and lower quantiles around the target.
    2. The `conformalize` method estimates the uncertainty of the quantile models
       using the conformity set.
    3. The `predict_interval` computes prediction points and intervals.

    Parameters
    ----------
    estimator : Union[`RegressorMixin`, `Pipeline`, \
`List[Union[RegressorMixin, Pipeline]]`]
        The regressor used to predict points and quantiles.

        When `prefit=False` (default), a single regressor that supports the quantile
        loss must be passed. Valid options:

        - ``sklearn.linear_model.QuantileRegressor``
        - ``sklearn.ensemble.GradientBoostingRegressor``
        - ``sklearn.ensemble.HistGradientBoostingRegressor``
        - ``lightgbm.LGBMRegressor``

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
    >>> from mapie_v1.regression import ConformalizedQuantileRegressor
    >>> from mapie_v1.utils import train_conformalize_test_split
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
    """

    def __init__(
        self,
        estimator: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        confidence_level: float = 0.9,
        prefit: bool = False,
    ) -> None:
        self._alpha = transform_confidence_level_to_alpha(confidence_level)
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False

        self._mapie_quantile_regressor = MapieQuantileRegressor(
            estimator=estimator,
            method="quantile",
            cv="prefit" if prefit else "split",
            alpha=self._alpha,
        )

        self._sample_weight: Optional[ArrayLike] = None
        self._predict_params: dict = {}

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
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
        raise_error_if_fit_called_in_prefit_mode(self._prefit)
        raise_error_if_method_already_called("fit", self._is_fitted)

        fit_params_, self._sample_weight = prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._mapie_quantile_regressor._initialize_fit_conformalize()
        self._mapie_quantile_regressor._fit_estimators(
            X=X_train,
            y=y_train,
            sample_weight=self._sample_weight,
            **fit_params_,
        )

        self._is_fitted = True
        return self

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Estimates the uncertainty of the quantile regressors by computing
        conformity scores on the conformity set.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Features of the conformity set.

        y_conformalize : ArrayLike
            Targets of the conformity set.

        predict_params : Optional[dict], default=None
            Parameters to pass to the `predict` method of the regressors.
            These parameters will also be used in the `predict_interval`
            and `predict` methods of this SplitConformalRegressor.

        Returns
        -------
        Self
            The ConformalizedQuantileRegressor instance.
        """
        raise_error_if_previous_method_not_called(
            "conformalize",
            "fit",
            self._is_fitted,
        )
        raise_error_if_method_already_called(
            "conformalize",
            self._is_conformalized,
        )

        self._predict_params = prepare_params(predict_params)
        self._mapie_quantile_regressor.conformalize(
            X_conformalize,
            y_conformalize,
            **self._predict_params
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
        Predicts points and intervals.

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

            If `symmetric_correction` is set to `False` , this constant is different for
            the upper and the lower quantile predictions. If set to True, this constant
            is the same for both.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape `(n_samples,)`
            - Prediction intervals, of shape `(n_samples, 2, 1)`
        """
        raise_error_if_previous_method_not_called(
            "predict_interval",
            "conformalize",
            self._is_conformalized,
        )

        predictions = self._mapie_quantile_regressor.predict(
            X,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds,
            symmetry=symmetric_correction,
            **self._predict_params
        )
        return cast_predictions_to_ndarray_tuple(predictions)

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
        raise_error_if_previous_method_not_called(
            "predict",
            "conformalize",
            self._is_conformalized,
        )

        estimator = self._mapie_quantile_regressor
        predictions, _ = estimator.predict(X, **self._predict_params)
        return predictions
