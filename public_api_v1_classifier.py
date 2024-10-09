from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional, Tuple, Union, cast, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, check_is_fitted, indexable)
from sklearn.linear_model import LogisticRegression

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseClassificationScore
from mapie.conformity_scores.sets.raps import RAPSConformityScore
from mapie.conformity_scores.sets.lac import LACConformityScore

from mapie.conformity_scores.utils import (
    check_depreciated_size_raps, check_classification_conformity_score,
    check_target
)
from mapie.estimator.classifier import EnsembleClassifier
from mapie.utils import (check_alpha, check_alpha_and_n_samples, check_cv,
                         check_estimator_classification, check_n_features_in,
                         check_n_jobs, check_null_weight, check_predict_params,
                         check_verbose)


class SplitConformalClassifier:
    
    def __init__(   
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        conformity_score: Union[str, BaseClassificationScore] = "lac", # Can be a string or a BaseClassificationScore object
        alpha: Union[float, List[float]] = 0.1,
        split_method: str = "simple",  # 'simple' (provide test_size in .fit) or 'prefit'. Future API: 'manual' (provide X_calib, Y_calib in .fit) and BaseCrossValidator (restricted to splitters only)
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
        test_size: Union[int, float] = 0.1,  #  -> In __init__ ?
        # Future API: X_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        # Future API:   y_calib: Optional[ArrayLike] = None,  # Must be None if split_method != 'manual'
        fit_params: Optional[dict] = None,  # For example, LBGMClassifier :  {'categorical_feature': 'auto'}
        predict_params: Optional[dict] = None, # For example, LBGMClassifier :  {'pred_leaf': False}
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
                     conformoty_score_params: Optional[dict] = None, # Parameters specific to conformal method, For example: include_last_label
                     ) -> NDArray:
        
        """
        Return
        -----
        An array containing the prediction sets 
        Shape (n_samples, n_classes) if alpha is float,
        Shape (n_samples, n_classes, alpha) if alpha is a list of floats
        """

        pass

class CrossConformalClassifier:
    
    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        conformity_score: Union[str, BaseClassificationScore] = 'lac',
        cross_val : Union[BaseCrossValidator, str] = 5,
        alpha: Union[float, List[float]] = 0.1,
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
                X: ArrayLike): # Parameters specific to conformal method, For example: include_last_label) -> ArrayLike:

        """
        Return 
        ----- 

        """
        pass

    def predict_sets(self,
                     X: ArrayLike,
                     agg_scores: Optional[str] = "mean", # how to aggregate the scores by the estimators on test data
                     conformoty_score_params: Optional[dict] = None,): # Parameters specific to conformal method, For example: include_last_label) -> NDArray
        
        """
        Return
        -----
        An array containing the prediction sets 
        Shape (n_samples, n_classes) if alpha is float,
        Shape (n_samples, n_classes, alpha) if alpha is a list of floats
        """
        