from typing import Optional, Union, Self, Iterable, Tuple, Any, List

import numpy as np
from sklearn.linear_model import LinearRegression

from mapie.regression import MapieRegressor
from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from mapie.conformity_scores import BaseRegressionScore, AbsoluteConformityScore


class NaiveConformalRegressor:
    def __init__(
        self,
        estimator: RegressorMixin = LinearRegression,  # None doesn't exist anymore
        conformity_score: BaseRegressionScore = AbsoluteConformityScore,  # Should we set this default?
        alpha: Union[float, List[float]] = 0.9,  # Should we set this default? Actually an array is OK (already implemented, and avoid developing a less user-friendly reset_alpha method)
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
        fit_params: dict,  # -> In __init__ ?
        predict_params: dict,  # -> In __init__ ?
    ) -> Self:
        pass

    def predict(
        self,
        X: ArrayLike,
        optimize_beta: bool = False,  # Don't understand that one
        allow_infinite_bounds: bool = False,
        # **predict_params  -> To remove: redundant with predict_params in .fit()
    ) -> Tuple[NDArray, NDArray]:
        """
        Returns
        -------
        Tuple[NDArray, NDArray]:
          - the first element contains the point predictions, with shape (n_samples,)
          - the second element contains the prediction intervals,
            with shape (n_samples, 2) if alpha is a float, or (n_samples, 2, n_alpha) if alpha is an array of floats
        """
        pass
