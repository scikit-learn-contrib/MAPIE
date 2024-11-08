from typing import Optional, Union, List
from typing_extensions import Self

import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseRegressionScore


class SplitConformalRegressor:
    """
    A conformal regression model using split conformal prediction to generate
    prediction intervals with statistical guarantees. This method involves
    using a hold-out conformity set to determine prediction intervals around
    point predictions from a base regressor.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regression estimator used to generate point predictions.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        String options:
        - "absolute": absolute error between predicted and true target values.
        - "gamma": absolute error between predicted and true target values
          normalized by the target prediction.
        - "residualsNorm": absolute error between predicted and true target
          values normalized by the residuals.

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals. Indicates the
        desired coverage probability of the prediction intervals. If a float
        is provided, it represents a single confidence level. If a list,
        multiple prediction intervals for each specified confidence level are
        returned.

    prefit : bool, default=False
        If True, assumes that the base estimator is already fitted, and skips
        refitting. If False, fits the estimator on the provided training data.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level.
        Higher values increase the output details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Notes
    -----
    This implementation currently uses a ShuffleSplit cross-validation scheme
    for splitting the conformity set. Future implementations may allow the use
    of groups.

    Methods
    -------
    fit(X_train, y_train, fit_params=None) -> Self
        Fits the base estimator to the training data and initializes any
        internal parameters required for conformal prediction.

    conformalize(X_conf, y_conf, predict_params=None) -> Self
        Fits the conformity score using a separate conformity set,
        allowing the model to adjust the prediction intervals based
        on conformity errors.

    predict(X) -> NDArray
        Generates point predictions for the input data `X`.

    predict_set(X, minimize_interval_width=False, allow_infinite_bounds=False)
        -> NDArray
        Generates prediction intervals for the input data `X` based on the
        conformity score and confidence level. The resulting intervals are
        adjusted to achieve the desired coverage probability.

        Returns
        -------
        NDArray
            An array containing the prediction intervals with shape
            `(n_samples, 2)` if `confidence_level` is a single float, or
            `(n_samples, 2, n_confidence_levels)` if `confidence_level` is a
            list of floats.
    Examples
    --------
    >>> regressor = SplitConformalRegressor(estimator=LinearRegression(),
                                            confidence_level=0.95)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),  # Improved 'None'
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        confidence_level: Union[float, List[float]] = 0.9,
        prefit: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        # groups -> not used in the current implementation
    ) -> None:
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
    ) -> Self:
        """
        Fits the base estimator to the training data.

        Parameters
        ----------
        X_train : ArrayLike
            Training data features.

        y_train : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method of the base
            estimator.

        Returns
        -------
        Self
            The fitted SplitConformalRegressor instance.
        """
        pass

    def conformalize(
        self,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
        predict_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
    ) -> Self:
        """
        Calibrates the fitted model to the conformity set. This step analyzes
        the conformity scores and adjusts the prediction intervals based on
        conformity errors and specified confidence levels.

        Parameters
        ----------
        X_conf : ArrayLike
            Features for the conformity set.

        y_conf : ArrayLike
            Target values for the conformity set.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The SplitConformalRegressor instance with updated prediction
            intervals.
        """
        pass

    def predict_set(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
        # **predict_params  -> QUESTION: Is this redundant with .fit() ?
    ) -> NDArray:
        """
        Generates prediction intervals based on the calibrated model and
        conformal predictions framework.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating prediction intervals.

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the interval width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.

        Returns
        -------
        NDArray
            Prediction intervals with shape:
            - (n_samples, 2) for single confidence level,
            - (n_samples, 2, n_confidence_levels) if multiple
               confidence levels are specified.
        """
        pass

    def predict(
        self,
        X: ArrayLike,
        # **predict_params  -> Is this redundant with predict_params in fit() ?
    ) -> NDArray:
        """
        Generates point predictions for given data using the fitted model.

        Parameters
        ----------
        X : ArrayLike
            Data features for prediction.

        Returns
        -------
        NDArray
            Array of point predictions, with shape (n_samples,).
        """
        pass

    def fit_conformalize(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Dummy method to fit and conformalize in one step for testing purposes.
        """
        pass


class CrossConformalRegressor:
    """
    A conformal regression model using cross-conformal prediction to generate
    prediction intervals with statistical guarantees. This method involves
    cross-validation with conformity scoring across multiple folds to determine
    prediction intervals around point predictions from a base regressor.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regression estimator used to generate point predictions.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        String options:
        - "absolute": absolute error between predicted and true target values.
        - "gamma": absolute error between predicted and true target values
          normalized by the target prediction.
        - "residualsNorm": absolute error between predicted and true target
          values normalized by the residuals.

        A custom score function inheriting from BaseRegressionScore may also be
        provided.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float is
        provided, it represents a single confidence level. If a list, multiple
        prediction intervals for each specified confidence level are returned.

    method : str, default="plus"
        The method used for cross-conformal prediction. Options are:
        - "base": Based on the conformity scores from each fold.
        - "plus": Based on the conformity scores from each fold plus
          the testing prediction.
        - "minmax": Based on the minimum and maximum conformity scores
          from each fold.

    cv : Union[int, BaseCrossValidator], default=5
        The cross-validation splitting strategy. If an integer is passed, it is
        the number of folds for `KFold` cross-validation. Alternatively, a
        specific cross-validation splitter from scikit-learn can be provided.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the
        output details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Methods
    -------
    fit(X, y, fit_params=None) -> Self
        Fits the base estimator to the training data using cross-validation.

    conformalize(X, y, predict_params=None) -> Self
        Calibrates the model using cross-validation and updates the prediction
        intervals based on conformity errors observed across folds.

    predict(X, aggregation_method=None) -> NDArray
        Generates point predictions for the input data `X` using the specified
        aggregation method across the cross-validation folds.

    predict_set(X, minimize_interval_width=False, allow_infinite_bounds=False)
    -> NDArray
        Generates prediction intervals for the input data `X` based on the
        conformity score and confidence level, adjusted to achieve the desired
        coverage probability.

    Returns
    -------
    NDArray
        An array containing the prediction intervals with shape:
        - `(n_samples, 2)` if `confidence_level` is a single float
        - `(n_samples, 2, n_confidence_levels)` if `confidence_level`
           is a list of floats.

    Notes
    -----
    Cross-conformal prediction provides enhanced robustness through the
    aggregation of multiple conformal scores across cross-validation folds,
    potentially yielding tighter intervals with reliable coverage guarantees.

    Examples
    --------
    >>> regressor = CrossConformalRegressor(
    ...     estimator=LinearRegression(), confidence_level=0.95, cv=10)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        confidence_level: Union[float, List[float]] = 0.9,
        method: str = "plus",  # 'base' | 'plus' | 'minmax'
        cv: Union[int, BaseCrossValidator] = 5,
        # Updated default; removed str option
        # 'prefit' option removed as unnecessary in cross-validation context
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
        # sample_weight in fit_params
        # groups specified directly in the cv parameter
    ) -> Self:
        """
        Fits the base estimator to the training data using cross-validation.

        Parameters
        ----------
        X : ArrayLike
            Training data features.

        y : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method
            of the base estimator.

        Returns
        -------
        Self
            The fitted CrossConformalRegressor instance.
        """
        pass

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Calibrates the fitted model using cross-validation conformal folds.
        This step analyzes conformity scores across multiple cross-validation
        folds and adjusts the prediction intervals based on conformity errors
        and specified confidence levels.

        Parameters
        ----------
        X : ArrayLike
            Features for generating conformity scores across folds.

        y : ArrayLike
            Target values for generating conformity scores across folds.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The CrossConformalRegressor instance with calibrated prediction
            intervals based on cross-validated conformity scores.
        """
        pass

    def predict_set(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
        # **predict_params -> To remove
        # : redundant with predict_params in .fit()
    ) -> NDArray:
        """
        Generates prediction intervals for the input data `X` based on the
        calibrated model and cross-conformal prediction framework.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating prediction intervals.

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the interval width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds if
            necessary.

        Returns
        -------
        NDArray
            Prediction intervals with shape
            - `(n_samples, 2)` if `confidence_level`is a single float,
            - `(n_samples, 2, n_confidence_levels)` if multiple confidence
               levels are specified.
        """
        pass

    def predict(
        self,
        X: ArrayLike,
        # ensemble: bool = False, -> removed, see aggregation_method
        aggregation_method: Optional[
            str
        ] = None,  # None: no aggregation, 'mean', 'median'
    ) -> NDArray:
        """
        Generates point predictions for the input data `X` using the
        fitted model. Optionally aggregates predictions across cross-
        validation folds models.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        aggregation_method : Optional[str], default=None
            The method to aggregate predictions across folds. Options:
            - None: No aggregation, returns predictions from each fold.
            - "mean": Returns the mean prediction across folds.
            - "median": Returns the median prediction across folds.

        Returns
        -------
        NDArray
            Array of point predictions, with shape `(n_samples,)`.
        """
        pass


class JackknifeAfterBootstrapRegressor:
    """
    A conformal regression model using the jackknife-after-bootstrap approach
    to generate prediction intervals with statistical guarantees. This method
    combines bootstrap sampling with the jackknife technique to produce robust
    prediction intervals around point predictions from a base regressor.

    Parameters
    ----------
    estimator : RegressorMixin, default=LinearRegression()
        The base regression estimator used to generate point predictions.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        String options:
        - "absolute": absolute error between predicted and true target values.
        - "gamma": absolute error between predicted and true target values
          normalized by the target prediction.
        - "residualsNorm": absolute error between predicted and true target
          values normalized by the residuals.

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float
        is provided, it represents a single confidence level. If a list,
        multiple prediction intervals for each specified confidence level are
        returned.

    method : str, default="plus"
        The method used for jackknife-after-bootstrap prediction. Options are:
        - "base": Based on the conformity scores from each bootstrap sample.
        - "plus": Based on the conformity scores from each bootstrap sample and
           the testing prediction.
        - "minmax": Based on the minimum and maximum conformity scores from
          each bootstrap sample.

    n_bootstraps : int, default=100
        The number of bootstrap resamples to generate for the
        jackknife-after-bootstrap procedure.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the output
        details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Methods
    -------
    fit(X, y, fit_params=None) -> Self
        Fits the base estimator to the training data and initializes internal
        parameters required for the jackknife-after-bootstrap process.

    conformalize(X_conf, y_conf, predict_params=None) -> Self
        Calibrates the model on provided data using the
        jackknife-after-bootstrap approach, adjusting the prediction intervals
        based on the observed conformity scores.

    predict(X, aggregation_method="mean") -> NDArray
        Generates point predictions for the input data `X` using the specified
        aggregation method over bootstrap samples.

    predict_set(X, allow_infinite_bounds=False) -> NDArray
        Generates prediction intervals for the input data `X` based on the
        calibrated jackknife-after-bootstrap predictions.

    Returns
    -------
    NDArray
        An array containing the prediction intervals with shape
        `(n_samples, 2)`, where each row represents the lower and
        upper bounds for each sample.

    Examples
    --------
    >>> regressor = JackknifeAfterBootstrapRegressor(
    ...    estimator=LinearRegression(), confidence_level=0.9, n_bootstraps=8)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        confidence_level: Union[float, List[float]] = 0.9,
        method: str = "plus",  # 'base' | 'plus' | 'minmax',
        n_bootstraps: int = 100,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base estimator to the training data.

        Parameters
        ----------
        X : ArrayLike
            Training data features.

        y : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method
            of the base estimator.

        Returns
        -------
        Self
            The fitted JackknifeAfterBootstrapRegressor instance.
        """

    def conformalize(
        self,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Calibrates the model on the provided data using the
        jackknife-after-bootstrap approach, which leverages pre-generated
        bootstrap samples along with jackknife techniques to estimate
        prediction intervals. This step analyzes conformity scores and
        adjusts the intervals based on specified confidence levels.

        Parameters
        ----------
        X_conf : ArrayLike
            Features for the calibration (conformity) data.

        y_conf : ArrayLike
            Target values for the calibration (conformity) data.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The JackknifeAfterBootstrapRegressor instance with
            calibrated prediction intervals.
        """

    def predict_set(
        self,
        X: ArrayLike,
        allow_infinite_bounds: bool = False,
    ) -> NDArray:
        """
        Computes prediction intervals for each sample in `X` based on
        the jackknife-after-bootstrap framework.

        Parameters
        ----------
        X : ArrayLike
            Test data for prediction intervals.

        allow_infinite_bounds : bool, default=False
            If True, allows intervals to include infinite bounds
            if required for coverage.

        Returns
        -------
        NDArray
            Prediction intervals of shape `(n_samples, 2)`,
            with lower and upper bounds for each sample.
        """
        pass

    def predict(
        self,
        X: ArrayLike,
        # ensemble: bool = False, -> removed, see aggregation_method
        aggregation_method: str = "mean",  # 'mean', 'median'
    ) -> NDArray:
        """
        Generates point predictions for the input data using the fitted model,
        with optional aggregation over bootstrap samples.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        aggregation_method : str, default="mean"
            The method to aggregate predictions across bootstrap samples.
            Options:
            - "mean": Returns the mean prediction across samples.
            - "median": Returns the median prediction across samples.

        Returns
        -------
        NDArray
            Array of point predictions, with shape `(n_samples,)`.
        """
        pass


class ConformalizedQuantileRegressor:
    """
    A conformal quantile regression model that generates prediction intervals
    with statistical guarantees using quantile regression as the base
    estimator. This approach provides prediction intervals by leveraging
    quantile predictions and applying conformal adjustments to ensure coverage.

    Parameters
    ----------
    estimator : RegressorMixin, default=QuantileRegressor()
        The base quantile regression estimator used to generate point and
        interval predictions.

    conformity_score : Union[str, BaseRegressionScore], default="absolute"
        The conformity score method used to calculate the conformity error.
        String options:
        - "absolute": absolute error between predicted and true target values.
        - "gamma": absolute error between predicted and true target values
           normalized by the target prediction.
        - "residualsNorm": absolute error between predicted and true target
           values normalized by the residuals.

        A custom score function inheriting from BaseRegressionScore may also
        be provided.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals. If a float
        is provided, it represents a single confidence level. If a list,
        multiple prediction intervals for each specified confidence level
        are returned.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the regressor.

    Methods
    -------
    fit(X, y, fit_params=None) -> Self
        Fits the base estimator to the training data and initializes internal
        parameters required for conformal prediction.

    conformalize(X_conf, y_conf, predict_params=None) -> Self
        Calibrates the model on provided data, adjusting the prediction
        intervals to achieve the specified confidence levels.

    predict(X) -> NDArray
        Generates point predictions for the input data `X`.

    predict_set(X,
                allow_infinite_bounds=False,
                minimize_interval_width=False,
                symmetric_intervals=True) -> NDArray
        Generates prediction intervals for the input data `X`,
        adjusted for desired scoverage based on the calibrated
        quantile predictions.

    Returns
    -------
    NDArray
        An array containing the prediction intervals with shape
        `(n_samples, 2)`,  where each row represents the lower and
        upper bounds for each sample.

    Examples
    --------
    >>> regressor = ConformalizedQuantileRegressor(
    ...     estimator=QuantileRegressor(), confidence_level=0.95)
    >>> regressor.fit(X_train, y_train)
    >>> regressor.conformalize(X_conf, y_conf)
    >>> intervals = regressor.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: RegressorMixin = QuantileRegressor(),
        confidence_level: Union[float, List[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        # n_jobs: Optional[int] = None
        # Not yet available in MapieQuantileRegressor
        # verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base estimator to the training data.

        Parameters
        ----------
        X : ArrayLike
            Training data features.

        y : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Additional parameters to pass to the `fit` method
            of the base estimator.

        Returns
        -------
        Self
            The fitted ConformalizedQuantileRegressor instance.
        """

    def conformalize(
        self,
        X_conf: ArrayLike,
        y_conf: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Calibrates the model on the provided data, adjusting the prediction
        intervals based on quantile predictions and specified confidence
        levels. This step analyzes the conformity scores and refines the
        intervals to ensure desired coverage.

        Parameters
        ----------
        X_conf : ArrayLike
            Features for the calibration (conformity) data.

        y_conf : ArrayLike
            Target values for the calibration (conformity) data.

        predict_params : Optional[dict], default=None
            Additional parameters for generating predictions
            from the estimator.

        Returns
        -------
        Self
            The ConformalizedQuantileRegressor instance with calibrated
            prediction intervals.
        """

    def predict_set(
        self,
        X: ArrayLike,
        allow_infinite_bounds: bool = False,
        minimize_interval_width: bool = False,  # replace optimize_beta
        symmetric_intervals: bool = True,  # replace symmetric
    ) -> NDArray:
        """
        Computes prediction intervals for quantile regression based
        on calibrated predictions.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data for prediction intervals.

        allow_infinite_bounds : bool, default=False
            If True, allows intervals to include infinite bounds
            if required for coverage.

        minimize_interval_width : bool, default=False
            If True, narrows the prediction intervals as much as possible
            while maintaining the target coverage level.

        symmetric_intervals : bool, default=True
            If True, computes symmetric intervals around the predicted
            median or mean.
            If False, calculates separate upper and lower bounds for
            asymmetric intervals.

        Returns
        -------
        NDArray
            Prediction intervals with shape `(n_samples, 2)`, with lower
            and upper bounds for each sample.
        """
        pass

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Generates point predictions for the input data using the fitted model.

        Parameters
        ----------
        X : ArrayLike
            Data features for generating point predictions.

        Returns
        -------
        NDArray
            Array of point predictions with shape `(n_samples,)`.
        """
        pass


class GibbsConformalRegressor:
    pass  # TODO
