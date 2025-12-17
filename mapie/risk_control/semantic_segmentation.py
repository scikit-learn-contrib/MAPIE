from typing import Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.utils import check_is_fitted

from .multi_label_classification import MultiLabelClassificationController
from .risks import precision_image, recall_image


class SemanticSegmentationController(MultiLabelClassificationController):
    """
    Risk controller for semantic segmentation tasks,
    inheriting from MultiLabelClassificationController.
    """

    risk_choice_map = {
        "precision": precision_image,
        "recall": recall_image,
    }

    def _transform_pred_proba(
        self, y_pred_proba: Union[Sequence[NDArray], NDArray], ravel: bool = True
    ) -> NDArray:
        """
        Transform predicted probabilities for semantic segmentation tasks.
        Parameters
        ----------
        y_pred_proba: Union[Sequence[NDArray], NDArray]
            Predicted probabilities or logits for each class.
        ravel: bool, default=True
            Whether to ravel the output array. Ravel is used when computing risks
            on the calibration dataset.
        Returns
        -------
        NDArray
            Transformed predicted probabilities of shape (1, n_samples, 1).
        """
        if not isinstance(y_pred_proba, np.ndarray):
            y_pred_proba = np.array(y_pred_proba)
        if np.min(y_pred_proba) < 0 or np.max(y_pred_proba) > 1:
            # Apply sigmoid to convert logits to probabilities
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
        if ravel:
            return y_pred_proba.ravel()[np.newaxis, :, np.newaxis]
        return y_pred_proba

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Prediction sets on new samples based on the target risk level.
        Prediction sets for a given ``alpha`` are deduced from the computed
        risks.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
        """

        check_is_fitted(self)

        # Estimate prediction sets
        y_pred_proba = self._predict_function(X)
        y_pred_proba_array = self._transform_pred_proba(y_pred_proba, ravel=False)

        y_pred_proba_array = np.repeat(y_pred_proba_array, len(self._alpha), axis=1)
        y_pred_proba_array = (
            y_pred_proba_array
            > self.best_predict_param[np.newaxis, :, np.newaxis, np.newaxis]
        )
        return y_pred_proba_array
