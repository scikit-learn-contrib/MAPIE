from copy import deepcopy
from typing import Union, cast

import numpy as np
from sklearn.utils.validation import _check_y, check_is_fitted, indexable

from mapie.classification import MapieClassifier
from mapie.conformity_scores import (
    AbsoluteConformityScore,
    APSConformityScore,
    GammaConformityScore,
    LACConformityScore,
    NaiveConformityScore,
    TopKConformityScore
)
from mapie.multi_label_classification import MapieMultiLabelClassifier
from mapie.regression import MapieRegressor
from mapie.utils import check_alpha
from mapie._typing import ArrayLike, NDArray



class Mondrian:

    allowed_estimators = (
        MapieClassifier, MapieRegressor, MapieMultiLabelClassifier
    )
    allowed_classification_ncs_str = [
        "lac", "score", "cumulated_score", "aps", "topk"
    ]
    allowed_classification_ncs_class = (
        LACConformityScore, NaiveConformityScore, APSConformityScore,
        TopKConformityScore
    )
    allowed_regression_ncs = (
        GammaConformityScore, AbsoluteConformityScore, APSConformityScore
    )
    fit_attributes = [
        "unique_groups",
        "mapie_estimators"
    ]

    def __init__(
        self, mapie_estimator: Union[MapieClassifier, MapieRegressor, MapieMultiLabelClassifier]
    ):
        self.mapie_estimator = mapie_estimator

    def _check_mapie_classifier(self):
        if not self.mapie_estimator.cv == "prefit":
            raise ValueError(
                "Mondrian can only be used if the underlying Mapie estimator "+
                "uses cv='prefit'"
            )

    def _check_groups_fit(X, groups: NDArray):
        """Check that each group is defined by an integer and check that there
        are at least 2 individuals per group"""
        if not np.issubdtype(groups.dtype, np.integer):
            raise ValueError("The groups must be defined by integers")
        _, counts = np.unique(groups, return_counts=True)
        if np.min(counts) < 2:
            raise ValueError("There must be at least 2 individuals per group")
        if len(groups) != X.shape[0]:
            raise ValueError("The number of individuals in the groups must be equal to the number of rows in X")

    def _check_groups_predict(self, X, groups):
        """Check that there is no new group in the prediction"""
        if not np.all(np.isin(groups, self.unique_groups)):
            raise ValueError("There is a new group in the prediction")
        if len(groups) != X.shape[0]:
            raise ValueError("The number of individuals in the groups must be equal to the number of rows in X")
        
    def _check_estimator(self):
        if not isinstance(self.mapie_estimator, self.allowed_estimators):
            raise ValueError(
                "The estimator must be a MapieClassifier, MapieRegressor or MapieMultiLabelClassifier"
            )
    
    def _check_confomity_score(self):
        if isinstance(self.mapie_estimator, MapieClassifier):
            if self.mapie_estimator.conformity_score is not None:
                if self.mapie_estimator.conformity_score not in self.allowed_classification_ncs_class:
                    raise ValueError(
                        "The conformity score for the MapieClassifier must be one of "+
                        f"{self.allowed_classification_ncs_class}"
                    )
            else:
                if self.mapie_estimator.ncs_str not in self.allowed_classification_ncs_str:
                    raise ValueError(
                        "The conformity score for the MapieClassifier must be one of "+
                        f"{self.allowed_classification_ncs_str}"
                    )
        elif isinstance(self.mapie_estimator, MapieRegressor):
            if self.mapie_estimator.conformity_score is not None:
                if self.mapie_estimator.conformity_score not in self.allowed_regression_ncs:
                    raise ValueError(
                        "The conformity score for the MapieRegressor must be one of "+
                        f"{self.allowed_regression_ncs}"
                    )

    def _check_fit_parameters(self, X, y, groups):
        self._check_estimator()
        self._check_mapie_classifier()
        self._check_confomity_score()
        self._check_groups_fit(X, groups)
        X, y = indexable(X, y)
        y = _check_y(y)
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        groups = cast(NDArray, groups)

        return X, y, groups

    def fit(self, X: ArrayLike, y: ArrayLike, groups: ArrayLike, **kwargs):
        
        self._check_fit_parameters(X, y, groups)
        self.unique_groups = np.unique(groups)
        self.mapie_estimators = {}

        for group in self.unique_groups:
            mapie_group_estimator = deepcopy(self.mapie_estimator)
            indices_groups = np.argwhere(groups == group)[:, 0]
            X_g, y_g = X[indices_groups], y[indices_groups]
            mapie_group_estimator.fit(X_g, y_g, **kwargs)
            self.mapie_estimators[group] = mapie_group_estimator
        return self

    def predict(self, X: ArrayLike, alpha, groups, **kwargs):

        check_is_fitted(self, self.fit_attributes)
        self._check_groups_predict(X, groups)
        if alpha is None:
            return self.mapie_estimator.predict(X, **kwargs)
        else:
            alpha_np = cast(NDArray, check_alpha(alpha))
            unique_groups = np.unique(groups)
            for i, group in enumerate(unique_groups):
                indices_groups = np.argwhere(groups == group)[:, 0]
                X_g = X[indices_groups]
                y_pred_g, y_pss_g = self.mapie_estimators[group].predict(X_g, alpha=alpha_np, **kwargs)
                if i == 0:
                    if len(y_pred_g.shape) == 1:
                        y_pred = np.empty((X.shape[0],))
                    else:
                        y_pred = np.empty((X.shape[0], y_pred_g.shape[1]))
                    y_pss = np.empty((X.shape[0], y_pss_g.shape[1], len(alpha_np)))
                y_pred[indices_groups] = y_pred_g
                y_pss[indices_groups] = y_pss_g

            return y_pred, y_pss
