from typing import Optional, Union, Self, List

import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor

from mapie._typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from mapie.conformity_scores import BaseRegressionScore


class SplitConformalRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),  # Improved 'None' default
        conformity_score: Union[str, BaseRegressionScore] = "absolute",  # Add string option
        confidence_level: Union[float, List[float]] = 0.9,
        prefit: bool = False,
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

    def conformalize(
        self,
        X_calib: ArrayLike,
        y_calib: ArrayLike,
        predict_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
    ) -> Self:
        pass

    # predict and predict_set signatures are the same as NaiveConformalRegressor
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

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        fit_params: Optional[dict] = None,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        # groups: Optional[ArrayLike] = None,  ->  To specify directly in the cv parameter
    ) -> Self:
        pass

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        predict_params: Optional[dict] = None,
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
        X: ArrayLike,
        # ensemble: bool = False, -> removed, see aggregation_method
        aggregation_method: Optional[str] = None,  # None: no aggregation, 'mean', 'median'
    ) -> NDArray:
        pass

class JackknifeAfterBootstrapRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        confidence_level: Union[float, List[float]] = 0.9,
        method: str = "plus",  # 'base' | 'plus' | 'minmax',
        n_bootstraps: int = 100,
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
    ) -> Self:
        pass

    def conformalize(
        self,
        X_calib: ArrayLike,
        y_calib: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        pass

    def predict_set(
        self,
        X: ArrayLike,
        allow_infinite_bounds: bool = False,
        # **predict_params  -> To remove: redundant with predict_params in .fit()
    ) -> NDArray:
        """
        Returns prediction intervals for each sample in `X`.
        """
        pass

    def predict(
        self,
        X: ArrayLike,
        # ensemble: bool = False, -> removed, see aggregation_method
        aggregation_method: Optional[str] = None,  # None: no aggregation, 'mean', 'median'
    ) -> NDArray:
        """
        Returns point predictions with shape (n_samples,).
        """
        pass


class ConformalizedQuantileRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = QuantileRegressor(),
        confidence_level: Union[float, List[float]] = 0.9,
        quantile: Union[float, List[float]] = 0.5,  # Quantile(s) to target
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
    ) -> Self:
        pass

    def conformalize(
        self,
        X_calib: ArrayLike,
        y_calib: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        pass

    def predict_set(
        self,
        X: ArrayLike,
        allow_infinite_bounds: bool = False,
    ) -> NDArray:
        """
        Returns prediction intervals for quantile regression.
        """
        pass

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Returns point predictions with shape (n_samples,).
        """
        pass

class GibbsConformalRegressor:
    pass  # TODO    