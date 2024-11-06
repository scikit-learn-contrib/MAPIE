from __future__ import annotations

from typing import Optional, Union, List

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import LogisticRegression

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseClassificationScore


class SplitConformalClassifier:

    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        # Can be a string or a BaseClassificationScore object
        confidence_level: Union[float, List[float]] = 0.9,
        split_method: str = "simple",
        # 'simple' (provide test_size in .fit) or 'prefit'. Future API: 'manual' (provide X_calib, Y_calib in .fit) and BaseCrossValidator (restricted to splitters only)
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0,
    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        # groups: Optional[ArrayLike] = None,  # Removed, because it is not used in split conformal classifier
        test_size: Union[int, float] = 0.1,  # -> In __init__ ?
        # Future API: X_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        # Future API:   y_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        fit_params: Optional[dict] = None,  # For example, LBGMClassifier :  {'categorical_feature': 'auto'}
        predict_params: Optional[dict] = None,  # For example, LBGMClassifier :  {'pred_leaf': False}
    ) -> SplitConformalClassifier:
        return self

    def predict(self,
                X: ArrayLike) -> NDArray:
        """
        Return
        ----- 
        Return ponctual prediction similar to predict method of scikit-learn classifiers
        Shape (n_samples,)
        """

    def predict_sets(self,
                     X: ArrayLike,
                     conformity_score_params: Optional[dict] = None,
                     # Parameters specific to conformal method, For example: include_last_label
                     ) -> NDArray:
        """
        Return
        -----
        An array containing the prediction sets 
        Shape (n_samples, n_classes) if confidence_level is float,
        Shape (n_samples, n_classes, confidence_level) if confidence_level is a list of floats
        """

        pass


class CrossConformalClassifier:

    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        conformity_score: Union[str, BaseClassificationScore] = 'lac',
        cross_val: Union[BaseCrossValidator, str] = 5,
        confidence_level: Union[float, List[float]] = 0.9,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0,

    ) -> None:
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        # sample_weight: Optional[ArrayLike] = None, -> in fit_params
        # groups: Optional[ArrayLike] = None,  
        fit_params: Optional[dict] = None,  # For example, LBGMClassifier :  {'categorical_feature': 'auto'}
        predict_params: Optional[dict] = None,
    ) -> CrossConformalClassifier:
        pass

    def predict(self,
                X: ArrayLike):  # Parameters specific to conformal method, For example: include_last_label) -> ArrayLike:

        """
        Return 
        ----- 
        Return ponctual prediction similar to predict method of scikit-learn classifiers
        Shape (n_samples,)
        """
        pass

    def predict_sets(self,
                     X: ArrayLike,
                     agg_scores: Optional[str] = "mean",  # how to aggregate the scores by the estimators on test data
                     conformity_score_params: Optional[
                         dict] = None, ):  # Parameters specific to conformal method, For example: include_last_label) -> NDArray

        """
        Return
        -----
        An array containing the prediction sets 
        Shape (n_samples, n_classes) if confidence_level is float,
        Shape (n_samples, n_classes, confidence_level) if confidence_level is a list of floats
        """
