import warnings
from typing import Optional, Union, Callable

import numpy as np
from numpy._typing import ArrayLike, NDArray

from mapie.control_risk.ltt import ltt_procedure
from mapie.risk_control import BinaryClassificationRisk


# General TODOs:
# TODO : in calibration and prediction,
#  use _transform_pred_proba or a function adapted to binary
# to get the probabilities depending on the classifier


# TODO: remove the no cover below
class BinaryClassificationController:  # pragma: no cover
    # TODO : test that this is working with a sklearn pipeline
    # TODO : test that this is working with a pandas dataframes
    def __init__(
        self,
        # X -> y_proba of shape (n_samples, 2)
        predict_function: Callable[[ArrayLike], NDArray],
        risk: BinaryClassificationRisk,
        target_level: float,
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[str, BinaryClassificationRisk] = "auto",
    ):
        self._predict_function = predict_function
        self._risk = risk
        self._best_predict_param_choice = best_predict_param_choice
        self._target_level = target_level
        self._delta = 1 - confidence_level

        self._thresholds: NDArray = np.linspace(0, 0.99, 100)
        # TODO: add a _is_calibrated attribute to check at prediction time

        self.valid_thresholds: Optional[NDArray] = None
        self.best_threshold: Optional[float] = None

    def calibrate(self, X_calibrate: ArrayLike, y_calibrate: ArrayLike) -> None:
        # TODO: Make sure the following works with sklearn train_test_split/Series
        y_calibrate_ = np.asarray(y_calibrate)

        predictions_proba = self._predict_function(X_calibrate)[:, 1]

        predictions_per_threshold = (
            predictions_proba[:, np.newaxis] >= self._thresholds
        ).T.astype(int)

        risks_and_eff_sizes = np.array(
            [self._risk.get_value_and_effective_sample_size(
                y_calibrate_,
                predictions
            ) for predictions in predictions_per_threshold]
        )

        risks_per_threshold = risks_and_eff_sizes[:, 0]
        eff_sample_sizes_per_threshold = risks_and_eff_sizes[:, 1]

        if self._risk.higher_is_better:
            risks_per_threshold = 1 - risks_per_threshold
            alpha = 1 - self._target_level
        else:
            alpha = self._target_level

        valid_thresholds_index = ltt_procedure(
            risks_per_threshold,
            np.array([alpha]),
            self._delta,
            eff_sample_sizes_per_threshold,
            True,
        )
        self.valid_thresholds = self._thresholds[valid_thresholds_index[0]]
        if len(self.valid_thresholds) == 0:
            warnings.warn(
                "No predict parameters were found to control the risk at the given "
                "target and confidence levels. "
                "Try using a larger calibration set or a better model.",
            )
        else:
            # Minimum in case of precision control only
            self.best_threshold = min(self.valid_thresholds)

    def predict(self, X_test: ArrayLike) -> NDArray:
        if self.best_threshold is None:
            raise ValueError(
                "No predict parameters were found to control the risk. Cannot predict."
            )
        predictions_proba = self._predict_function(X_test)[:, 1]
        return (predictions_proba >= self.best_threshold).astype(int)
