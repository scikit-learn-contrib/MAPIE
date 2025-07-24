import warnings
from typing import Any, Optional, Union

import numpy as np
from numpy._typing import ArrayLike, NDArray
from sklearn.utils import check_random_state

from mapie.control_risk.ltt import ltt_procedure
from mapie.utils import _check_n_jobs, _check_verbose

# General TODOs:
# TODO: maybe use type float instead of float32?
# TODO : in calibration and prediction,
#  use _transform_pred_proba or a function adapted to binary
# to get the probabilities depending on the classifier


class BinaryClassificationController:  # pragma: no cover
    # TODO : test that this is working with a sklearn pipeline
    # TODO : test that this is working with a pandas dataframes
    """
    Controller for the calibration of our binary classifier.

    Parameters
    ----------
    fitted_binary_classifier: Any
        Any object that provides a `predict_proba` method.

    metric: str
        The performance metric we want to control (ex: "precision")

    target_level: float
        The target performance level we want to achieve (ex: 0.8)

    confidence_level: float
        The maximum acceptable probability of the precision falling below the
        target precision level (ex: 0.8)

    Attributes
    ----------
    precision_per_threshold: NDArray
        Precision of the binary classifier on the calibration set for each
        threshold from self._thresholds.

    valid_threshold: NDArray
        Thresholds that meet the target precision with the desired confidence.

    best_threshold: float
        Valid threshold that maximizes the recall, i.e. the smallest valid
        threshold.
    """

    def __init__(
        self,
        fitted_binary_classifier: Any,
        metric: str,
        target_level: float,
        confidence_level: float = 0.9,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ):
        _check_n_jobs(n_jobs)
        _check_verbose(verbose)
        check_random_state(random_state)

        self._classifier = fitted_binary_classifier
        self._alpha = 1 - target_level
        self._delta = 1 - confidence_level
        self._n_jobs = n_jobs  # TODO : use this in the class or delete
        self._random_state = random_state  # TODO : use this in the class or delete
        self._verbose = verbose  # TODO : use this in the class or delete

        self._thresholds: NDArray[np.float32] = np.arange(0, 1, 0.01)
        # TODO: add a _is_calibrated attribute to check at prediction time

        self.valid_thresholds: Optional[NDArray[np.float32]] = None
        self.best_threshold: Optional[float] = None

    def calibrate(self, X_calibrate: ArrayLike, y_calibrate: ArrayLike) -> None:
        """
        Find the threshold that statistically guarantees the desired precision
        level while maximizing the recall.

        Parameters
        ----------
        X_calibrate: ArrayLike
            Features of the calibration set.

        y_calibrate: ArrayLike
            True labels of the calibration set.

        Raises
        ------
        ValueError
            If no thresholds that meet the target precision with the desired
            confidence level are found.
        """
        # TODO: Make sure this works with sklearn train_test_split/Series
        y_calibrate_ = np.asarray(y_calibrate)

        predictions_proba = self._classifier.predict_proba(X_calibrate)[:, 1]

        risk_per_threshold = 1 - self._compute_precision(
            predictions_proba, y_calibrate_
        )

        valid_thresholds_index, _ = ltt_procedure(
            risk_per_threshold,
            np.array([self._alpha]),
            self._delta,
            len(y_calibrate_),
            True,
        )
        self.valid_thresholds = self._thresholds[valid_thresholds_index[0]]
        if len(self.valid_thresholds) == 0:
            # TODO: just warn, and raise error at prediction if no valid thresholds
            warnings.warn("No valid thresholds found", UserWarning)

        # Minimum in case of precision control only
        self.best_threshold = min(self.valid_thresholds)

    def predict(self, X_test: ArrayLike) -> NDArray:
        """
        Predict binary labels on the test set, using the best threshold found
        during calibration.

        Parameters
        ----------
        X_test: ArrayLike
            Features of the test set.

        Returns
        -------
        ArrayLike
            Predicted labels (0 or 1) for each sample in the test set.
        """
        predictions_proba = self._classifier.predict_proba(X_test)[:, 1]
        return (predictions_proba >= self.best_threshold).astype(int)

    def _compute_precision(  # TODO: use sklearn or MAPIE ?
        self, predictions_proba: NDArray[np.float32], y_cal: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Compute the precision for each threshold.
        """
        predictions_per_threshold = (
            predictions_proba[:, np.newaxis] >= self._thresholds
        ).astype(int)

        true_positives = np.sum(
            (predictions_per_threshold == 1) & (y_cal[:, np.newaxis] == 1),
            axis=0,
        )
        false_positives = np.sum(
            (predictions_per_threshold == 1) & (y_cal[:, np.newaxis] == 0),
            axis=0,
        )

        positive_predictions = true_positives + false_positives

        # Avoid division by zero
        precision_per_threshold = np.ones_like(self._thresholds, dtype=float)
        nonzero_mask = positive_predictions > 0
        precision_per_threshold[nonzero_mask] = (
            true_positives[nonzero_mask] / positive_predictions[nonzero_mask]
        )

        return precision_per_threshold
