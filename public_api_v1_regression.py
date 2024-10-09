from typing import Optional, Union, Self, Tuple, List

import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor

from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from mapie.conformity_scores import BaseRegressionScore, AbsoluteConformityScore


class NaiveConformalRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),  # Improved 'None' default
        conformity_score: Union[str, BaseRegressionScore] = "absolute",  # Add string option
        alpha: Union[float, List[float]] = 0.1,  # QUESTION: Should we set this default, or should we keep None? I think an array is OK (already implemented, and avoid developing a less user-friendly reset_alpha method)
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
        of shape (n_samples, 2) if alpha is a float,
        or (n_samples, 2, n_alpha) if alpha is an array of floats
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
        alpha: Union[float, List[float]] = 0.1,  # See comment in NaiveConformalRegressor
        split_method: str = "simple",  # 'simple' (provide test_size in .fit) or 'prefit'. Future API: 'manual' (provide X_calib, Y_calib in fit) and BaseCrossValidator (restricted to splitters only)
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None
        # groups -> not used in the current implementation (that is using ShuffleSplit)
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        test_size: Union[int, float] = 0.1,  # Moved from __init__, improved 'None' default. Invalid if split_method != 'simple'
        # Future API: X_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        # Future API: y_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
    ) -> Self:
        pass

    # predict and predict_set signatures are the same as NaiveConformalRegressor


class CrossConformalRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression(),  # Improved 'None' default
        conformity_score: Union[str, BaseRegressionScore] = "absolute",  # Add string option
        alpha: Union[float, List[float]] = 0.1,  # See comment in NaiveConformalRegressor
        method: str = "plus",  # 'base' | 'plus' | 'minmax'
        cross_val: Union[int, BaseCrossValidator] = 5,  # Improved 'None' default, removed str option, update name. Note that we lose the prefit option, that was I think useless in a cross-validation context QUESTION
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
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        # groups: Optional[ArrayLike] = None,  ->  To specify directly in the cross_val parameter
        fit_params: Optional[dict] = None,
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
        alpha: Union[float, List[float]] = 0.1,  # See comment in NaiveConformalRegressor
        split_method: str = "simple",  # 'simple' (provide test_size in .fit), 'prefit' or 'manual'. Future API: BaseCrossValidator (restricted to splitters only)
        random_state: Optional[Union[int, np.random.RandomState]] = None,  # Moved from .fit
        # Future API : n_jobs: Optional[int] = None,
        # Future API : verbose: int = 0,
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        # groups: Optional[ArrayLike] = None,  ->  To specify directly in the cross_val parameter
        # shuffle: Optional[bool] = True, -> To implement in a future version (using the BaseCrossValidator split_method). In that case we would lose that feature in the v1.0.0 QUESTION
        # stratify: Optional[ArrayLike] = None, -> same comment as shuffle
        test_size: Union[int, float] = 0.1,  # Renamed from 'calib_size'
        X_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        y_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        fit_params: Optional[dict] = None,
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