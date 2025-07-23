from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, List, Any

import numpy as np
from numpy._typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class BinaryClassificationRisk:
    # Any risk that can be defined in the following way will work using the binary Hoeffding-Bentkus p-values used in MAPIE
    # Take the example of precision in the docstring to explain how the class works.
    def __init__(
        self,
        occurrence: Callable[[int, int], Optional[float]],
        # (y_true, y_pred) (y_pred possibly None), output: float between 0 and 1 or None if undefined
        higher_is_better: bool,
        binary: bool,  # True if occurrence is binary (too advanced?)
    ):
        self.occurrence = occurrence
        self.higher_is_better = higher_is_better
        self.binary = binary

    def get_value_and_effective_sample_size(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike
    ) -> Tuple[float, int]:  # float between 0 and 1, int between 0 and len(y_true)
        pass  # this function will most probably be the same for all risks

    def transform_to_opposite(self) -> BinaryClassificationRisk:
        def opposite_occurrence(y_true, y_pred):
            None if not self.occurrence(y_true, y_pred) else 1 - self.occurrence(
                y_true,
                y_pred
            )

        return BinaryClassificationRisk(
            occurrence=opposite_occurrence,
            higher_is_better=not self.higher_is_better,
            binary=self.binary
        )


class BinaryClassificationController:
    def __init__(
        self,
        predict_function: Union[Callable[..., int], Callable[[Any], float]],
        # either a predict_proba (X -> proba), either anything predicting using parameters ((X, *params) -> 0/1)
        risk: Union[BinaryClassificationRisk | List[BinaryClassificationRisk]],
        target_level: Union[float, List[float]],
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[str, BinaryClassificationRisk] = "auto",
        # Can't be "auto" in multi-risk or for custom risk
        predict_params=None,  # Can't be None if predict_function is not a predict_proba
        # list of dict (each element of the list corresponds to a set of parameters for predict_func)
        # dict of list possible? in this case we compute all combinations, a bit violent combinatorially
        # can also be an array of parameters (numpy array), it works in both cases, a parameter would be needed to choose (combination or array)
        predict_params_graph=None,
        # list of lists (for each param value, the list of its neighbors), we start with the first element of the list of params_values
    ):
        self.predict_function = predict_function
        self.risk = risk
        self.target_level = target_level
        self.confidence_level = confidence_level
        self.best_predict_param_choice = best_predict_param_choice
        self.predict_params = predict_params
        self.predict_params_graph = predict_params_graph

    def calibrate(
        self,
        X_calibrate: ArrayLike,
        y_calibrate: ArrayLike,
    ) -> BinaryClassificationController:
        pass

    def predict(self, X_test: ArrayLike) -> NDArray:
        pass


# Examples
clf = LogisticRegression()  #.fit(X_train, y_train)

precision = BinaryClassificationRisk(
    occurrence=lambda y_true, y_pred: None if y_pred == 0 else int(y_pred == y_true),
    higher_is_better=True,
    binary=True,
)

false_discovery_rate = precision.transform_to_opposite()

accuracy = BinaryClassificationRisk(
    occurrence=lambda y_true, y_pred: int(y_pred == y_true),
    higher_is_better=True,
    binary=True,
)

# Simplest API
BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk=precision,
    target_level=0.8,
)

# Simple API with custom choice
BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk=false_discovery_rate,
    target_level=0.7,
    confidence_level=0.95,
    best_predict_param_choice=accuracy,
)
# Multi-risk
BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk=[precision, accuracy],
    target_level=[0.8, 0.9],
    best_predict_param_choice=precision,
)


# Dummy definitions to demonstrate use case 3
def abstention_classifier(X, param_1, param_2):
    return 1


negative_predictive_value = precision
abstention_rate = precision
predict_params = []

# Use case 3: abstention classifier with multiple risks and multi-dimensional lambda
controller_use_case_3 = BinaryClassificationController(
    predict_function=abstention_classifier,
    risk=[precision, negative_predictive_value],
    target_level=0.8,
    confidence_level=0.95,
    best_predict_param_choice=abstention_rate,
    predict_params=predict_params,
)

# Using SVC decision_function
svc = SVC()  #.fit()


def svc_predict(X, threshold):
    return svc.decision_function(X) >= threshold


controller_svc = BinaryClassificationController(
    predict_function=svc_predict,
    risk=precision,
    target_level=0.8,
    predict_params=np.linspace(-1, 1, 100),
)
