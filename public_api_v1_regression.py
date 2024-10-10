from typing import Optional, Union, Self, List

import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor

from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from mapie.conformity_scores import BaseRegressionScore


class NaiveConformalRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),  # Improved 'None' default
        conformity_score: Union[str, BaseRegressionScore] = "absolute",  # Add string option
        confidence_level: Union[float, List[float]] = 0.9,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        fit_params: Optional[dict] = None,  # Ex for LGBMClassifier: {'categorical_feature': 'auto'}
        predict_params: Optional[dict] = None,
    ) -> Self:
        pass

    def predict_set(
        self,
        X: ArrayLike,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        # **predict_params  -> QUESTION: Is this redundant with predict_params in .fit() ?
    ) -> NDArray:
        """
        Returns
        -------
        An array containing the prediction intervals,
        of shape (n_samples, 2) if confidence_level is a float,
        or (n_samples, 2, n_confidence_level) if confidence_level is an array of floats
        """
        pass

    def predict(
        self,
        X: ArrayLike,
        # **predict_params  -> Is this redundant with predict_params in .fit() ?
    ) -> NDArray:
        """
        Returns
        -------
        An array containing the point predictions, with shape (n_samples,)
        """
        pass


class SplitConformalRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),  # Improved 'None' default
        conformity_score: Union[str, BaseRegressionScore] = "absolute",  # Add string option
        confidence_level: Union[float, List[float]] = 0.9,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
        # groups -> not used in the current implementation (that is using ShuffleSplit)
    ) -> None:
        pass

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
    ) -> Self:
        pass

    def calibrate(
        self,
        X_calib: ArrayLike,
        y_calib: ArrayLike,
        predict_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
    ) -> Self:
        pass

    # predict and predict_set signatures are the same as NaiveConformalRegressor


class CrossConformalRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),  # Improved 'None' default
        conformity_score: Union[str, BaseRegressionScore] = "absolute",  # Add string option
        confidence_level: Union[float, List[float]] = 0.9,
        method: str = "plus",  # 'base' | 'plus' | 'minmax'
        cv: Union[int, BaseCrossValidator] = 5,  # Improved 'None' default, removed str option, update name. Note that we lose the prefit option, that was useless in a cross-validation context
        # agg_function -> moved to predict method
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        pass

    def fit_calibrate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        # groups: Optional[ArrayLike] = None,  ->  To specify directly in the cv parameter
    ) -> Self:
        pass

    def predict_set(
        self,
        X: ArrayLike,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        # **predict_params  -> To remove: redundant with predict_params in .fit()
    ) -> NDArray:  # See docstring in NaiveConformalRegressor for the return type details
        pass

    def predict(
        self,
        # ensemble: bool = False, -> removed, see aggregation_method
        aggregation_method: Optional[str] = None,  # None: no aggregation, 'mean', 'median'
    ) -> NDArray:
        pass


class JackknifeAfterBootstrapRegressor:
    pass  # TODO


class ConformalizedQuantileRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = QuantileRegressor(),  # Improved 'None' default
        confidence_level: Union[float, List[float]] = 0.9,
        random_state: Optional[Union[int, np.random.RandomState]] = None,  # Moved from .fit
        # Future API : n_jobs: Optional[int] = None,
        # Future API : verbose: int = 0,
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        # groups: Optional[ArrayLike] = None,  ->  To specify directly in the cv parameter
    ) -> Self:
        pass

    def calibrate(
        self,
        X_calib: ArrayLike,
        y_calib: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        pass

    def predict_set(
        self,
        X: ArrayLike,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        symmetry: bool = True,  # Corrected typing
    ) -> NDArray:
        pass

    # predict signature is the same as NaiveConformalRegressor


class DRAFTTimeSeriesConformalRegressor: # DRAFT
    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "enbpi",
        # Future API : n_jobs: Optional[int] = None,
        # Future API : verbose: int = 0,
        conformity_score: Optional[BaseRegressionScore] = None,
        # Future API : random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        pass

