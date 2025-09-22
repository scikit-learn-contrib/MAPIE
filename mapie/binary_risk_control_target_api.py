# pylint: disable=line-too-long

from __future__ import annotations
from typing import Callable, Optional, Tuple, Union, List, Any, Literal

import numpy as np
from numpy._typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from mapie.risk_control import precision, accuracy, BinaryClassificationRisk


class BinaryClassificationRisk_:
    def __init__(
        self,
        risk_occurrence: Callable[[int, int], bool],
        risk_condition: Callable[[int, int], bool],
        # or Callable[[int, int], Union[bool, int]] if binary allowed
        higher_is_better: bool,
        binary: bool,  # True if occurrence is binary (too advanced?)
    ):
        self._risk_occurrence = risk_occurrence
        self._risk_condition = risk_condition
        self.higher_is_better = higher_is_better
        self.binary = binary

    def get_value_and_effective_sample_size(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> Tuple[float, int]:
        return 0, 0


class BinaryClassificationController:
    def __init__(
        self,
        predict_function: Callable[..., NDArray],  # either a predict_proba (X -> proba), either anything predicting using parameters ((X, *params) -> 0/1)
        risk: Union[BinaryClassificationRisk, List[BinaryClassificationRisk]],
        target_level: Union[float, List[float]],
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[
            Literal["auto"], BinaryClassificationRisk] = "auto",  # Shall we allow "auto" for multi-risk ?
        predict_params: Optional[NDArray] = None,  # Can't be None if predict_function is not a predict_proba. An array of parameters of shape (nb_combinations, nb_parameters), or shape (nb_combinations, ) in case of a unidimensional parameter, explicitly listing all possible joint values of the parameters
        predict_params_graph=None,  # list of lists (for each param value, the list of its neighbors), we start with the first element of the list of params_values
    ):
        self.predict_function = predict_function
        self.risk = risk
        self.target_level = target_level
        self.confidence_level = confidence_level
        self.best_predict_param_choice = best_predict_param_choice
        self.predict_params = predict_params
        self.predict_params_graph = predict_params_graph
        return

    def calibrate(
        self,
        X_calibrate: ArrayLike,
        y_calibrate: ArrayLike,
    ) -> BinaryClassificationController:
        return self

    def predict(self, X_test: ArrayLike) -> NDArray:
        return np.array([])


# Examples
clf = LogisticRegression()  #.fit(X_train, y_train)

# Multi-risk
BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk=[precision, accuracy],
    target_level=[0.8, 0.9],
    best_predict_param_choice=precision,
)

# Dummy definitions to demonstrate use case 3
def abstention_classifier(X, param_1, param_2):
    return np.array([1] * len(X))  # dummy

negative_predictive_value = precision  # dummy
abstention_rate = precision  # dummy
predict_params = np.array([])  # dummy

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
    return (np.array(svc.decision_function(X)) >= threshold).astype(int)

controller_svc = BinaryClassificationController(
    predict_function=svc_predict,
    risk=precision,
    target_level=0.8,
    predict_params=np.linspace(-1, 1, 100),
)
