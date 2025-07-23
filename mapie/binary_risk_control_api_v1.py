from __future__ import annotations
from typing import Callable, Optional, Tuple, Union

from numpy._typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression


class BinaryClassificationRisk:
    # Any risk that can be defined in the following way will work using the binary Hoeffding-Bentkus p-values used in MAPIE
    # Take the example of precision in the docstring to explain how the class works.
    def __init__(
        self,
        occurrence: Callable[[int, int], Optional[int]],
        # (y_true, y_pred), output: int (0 or 1) or None if undefined
        higher_is_better: bool,
    ):
        self.occurrence = occurrence
        self.higher_is_better = higher_is_better

    def get_value_and_effective_sample_size(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike
    ) -> Tuple[float, int]:  # float between 0 and 1, int between 0 and len(y_true)
        pass  # this function will most probably be the same for all risks
        # if the risk is not defined, return None


class BinaryClassificationController:
    def __init__(
        self,
        predict_function: Callable[..., int],
        risk: BinaryClassificationRisk,
        target_level: float,
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[str, BinaryClassificationRisk] = "auto",
        # Can't be "auto" for custom risk
        predict_params: Optional[NDArray] = None,
        # list of lists (for each param value, the list of its neighbors), we start with the first element of the list of params_values
    ):
        self.predict_function = predict_function
        self.risk = risk
        self.target_level = target_level
        self.confidence_level = confidence_level
        self.best_predict_param_choice = best_predict_param_choice
        self.predict_params = predict_params

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
)

accuracy = BinaryClassificationRisk(
    occurrence=lambda y_true, y_pred: int(y_pred == y_true),
    higher_is_better=True,
)

recall = BinaryClassificationRisk(
    occurrence=lambda y_true, y_pred: None if y_true == 0 else int(y_pred == y_true),
    higher_is_better=True,
)