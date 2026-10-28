from __future__ import annotations

import warnings
from typing import Any, Callable, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import column_or_1d

from mapie.utils import check_is_fitted


def _false_negative_loss(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """Per-example loss: 1 if a positive example is predicted negative."""
    return np.asarray((y_true == 1) & (y_pred == 0), dtype=float)


class BinaryClassificationCRController(BaseEstimator):
    """
    Controls the expected risk of a binary classifier with Conformal Risk Control.

    BinaryClassificationCRController selects a decision threshold on the
    positive-class probability such that, in expectation, a user-defined loss
    is below ``target_level`` on a new exchangeable example [1].

    Usage:

    1. Instantiate a BinaryClassificationCRController, providing the
       predict_proba method of your fitted binary classifier
    2. Call the calibrate method to find the threshold
    3. Use the predict method to predict using the selected threshold

    Parameters
    ----------
    predict_function : Callable[[ArrayLike], NDArray]
        predict_proba method of a fitted binary classifier.
        Its output must be of shape (len(X), 2), and column 1 must contain
        the probability of the positive class.

    target_level : float
        Maximum expected risk (the "alpha" of [1]). Must be in
        (0, ``upper_bound``).

    risk_function : Optional[Callable[[NDArray, NDArray], NDArray]], default=None
        Per-example loss ``risk_function(y_true, y_pred)``, where ``y_true`` and
        ``y_pred`` are arrays of 0/1 (1 is the positive class) and the output is
        an array of the same length with values in [0, ``upper_bound``].
        It must act example by example (no aggregation) and be monotone in the
        threshold, as stated by ``monotonicity``.
        If None, the false negative loss is used (1 if a positive example is
        predicted negative, else 0).

    monotonicity : {"increasing", "decreasing"}, default="increasing"
        Behavior of the loss when the threshold increases (i.e. when fewer
        examples are predicted positive).
        "increasing" means the loss is non-decreasing (e.g. false negative
        loss), "decreasing" means it is non-increasing (e.g. false positive
        loss). The default loss is only compatible with "increasing".

    upper_bound : float, default=1.0
        Upper bound ``B`` of the loss (see [1]).

    positive_class : Optional[Any], default=None
        Label of the positive class in the calibration labels.
        If None, labels must already be in {0, 1}.
        It should be the class associated with column 1 of ``predict_function``
        (``classes_[1]`` for scikit-learn classifiers).

    Attributes
    ----------
    threshold_ : float
        Selected threshold. ``-inf`` (resp. ``inf``) means that all examples are
        predicted positive (resp. negative).

    thresholds_ : NDArray
        Candidate thresholds, sorted: the distinct calibration probabilities,
        plus ``-inf`` and ``inf``.

    empirical_risk_ : NDArray
        Average calibration loss for each candidate threshold.

    risk_bound_ : NDArray
        Corrected risk ``(n * empirical_risk + B) / (n + 1)`` for each
        candidate threshold. The selected threshold is the most extreme one
        such that this quantity (for it and all less extreme ones) is at most
        ``target_level``.

    Notes
    -----
    Under the assumptions below, the CRC theorem of [1] states that, for the
    selected threshold, :math:`E[L_{n+1}(\\hat{\\lambda})] \\leq` ``target_level``,
    where the expectation is over the calibration set *and* the new example.

    - Calibration and test examples are exchangeable.
    - The loss is bounded in [0, ``upper_bound``].
    - The loss is monotone in the threshold, according to ``monotonicity``.
    - The loss at the most extreme threshold that predicts the "safest" label
      (``-inf`` for "increasing", ``inf`` for "decreasing") is at most
      ``target_level``. It holds for the default loss, where it is 0.
      Otherwise, it is only checked on calibration data (a warning is raised).
      If no threshold is feasible, this safest threshold is returned.

    This is a guarantee on the *expected* risk, averaged over calibration sets.
    It does not mean that the risk is below ``target_level`` with high
    probability for a given calibration set; this is what Learn Then Test
    (``BinaryClassificationController``) or RCPS provide, with a confidence
    level. That is why there is no ``confidence_level`` parameter here.

    The monotonicity of the loss, and its bound, can only be checked on the
    calibration examples: passing these checks does not prove them on unseen
    data. Computing the loss for all thresholds costs O(n^2) for n calibration
    examples.

    The threshold candidates depend on the calibration probabilities, which
    is without consequence: the selected threshold is the same as the one
    defined by [1] over all real thresholds, as the calibration loss only changes
    at these values. Predictions use ``proba >= threshold_`` for "increasing" and
    ``proba > threshold_`` for "decreasing", which ensures the (left or right)
    continuity of the loss required by the theorem.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from mapie.risk_control import BinaryClassificationCRController

    >>> X, y = make_classification(
    ...     n_features=2,
    ...     n_redundant=0,
    ...     n_informative=2,
    ...     n_clusters_per_class=1,
    ...     class_sep=2.0,
    ...     random_state=42,
    ... )
    >>> X_train, X_temp, y_train, y_temp = train_test_split(
    ...     X, y, test_size=0.6, random_state=42
    ... )
    >>> X_calib, X_test, y_calib, y_test = train_test_split(
    ...     X_temp, y_temp, test_size=0.5, random_state=42
    ... )
    >>> clf = LogisticRegression().fit(X_train, y_train)

    >>> controller = BinaryClassificationCRController(
    ...     predict_function=clf.predict_proba,
    ...     target_level=0.2,
    ... )
    >>> y_pred = controller.calibrate(X_calib, y_calib).predict(X_test)
    >>> y_pred.shape
    (30,)
    >>> selected = controller.thresholds_ == controller.threshold_
    >>> bool(controller.risk_bound_[selected][0] <= 0.2)
    True

    References
    ----------
    [1] Angelopoulos, Anastasios N., Stephen Bates, Adam Fisch, Lihua Lei,
    and Tal Schuster. "Conformal Risk Control." (2022).
    """

    def __init__(
        self,
        predict_function: Callable[[ArrayLike], NDArray],
        target_level: float,
        risk_function: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
        monotonicity: Literal["increasing", "decreasing"] = "increasing",
        upper_bound: float = 1.0,
        positive_class: Optional[Any] = None,
    ):
        self.predict_function = predict_function
        self.target_level = target_level
        self.risk_function = risk_function
        self.monotonicity = monotonicity
        self.upper_bound = upper_bound
        self.positive_class = positive_class

    @property
    def is_fitted(self) -> bool:
        """Returns True if the controller is calibrated."""
        return hasattr(self, "threshold_")

    def calibrate(
        self, X_calibrate: ArrayLike, y_calibrate: ArrayLike
    ) -> BinaryClassificationCRController:
        """
        Calibrate the BinaryClassificationCRController.
        Sets attributes threshold_, thresholds_, empirical_risk_ and risk_bound_.

        Parameters
        ----------
        X_calibrate : ArrayLike
            Features of the calibration set.

        y_calibrate : ArrayLike
            Binary labels of the calibration set.

        Returns
        -------
        BinaryClassificationCRController
            The calibrated controller instance.
        """
        self._check_parameters()
        y = self._encode_labels(y_calibrate)
        proba = self._get_positive_proba(X_calibrate)
        if len(proba) != len(y):
            raise ValueError(
                "X_calibrate and y_calibrate must have the same number of samples."
            )
        n = len(y)

        thresholds = np.unique(np.concatenate([[-np.inf], proba, [np.inf]]))
        losses = self._compute_losses(y, proba, thresholds)
        empirical_risk = losses.mean(axis=0)
        risk_bound = (n * empirical_risk + self.upper_bound) / (n + 1)
        feasible = risk_bound <= self.target_level
        increasing = self.monotonicity == "increasing"

        # The safest threshold predicts only the label that cannot be wrong.
        safest = 0 if increasing else len(thresholds) - 1
        if losses[:, safest].max() > self.target_level:
            warnings.warn(
                "The loss at the terminal threshold is above target_level on "
                "calibration data. The assumptions of the Conformal Risk Control "
                "theorem are not met: the risk is not guaranteed to be controlled."
            )
        if not feasible[safest]:
            warnings.warn(
                "The risk cannot be controlled: no threshold satisfies the "
                "corrected risk bound for this target_level and calibration set "
                "size. Returning the safest threshold."
            )
            index = safest
        elif increasing:
            index = int(np.cumprod(feasible).sum()) - 1
        else:
            index = len(thresholds) - int(np.cumprod(feasible[::-1]).sum())

        self.thresholds_ = thresholds
        self.empirical_risk_ = empirical_risk
        self.risk_bound_ = risk_bound
        self.threshold_ = float(thresholds[index])
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict the labels (1 for the positive class) using the selected threshold.

        Parameters
        ----------
        X : ArrayLike
            Features to predict.

        Returns
        -------
        NDArray of shape (n_samples,)
            Predictions in {0, 1}.
        """
        check_is_fitted(self)
        proba = self._get_positive_proba(X)
        return self._decide(proba, self.threshold_).astype(int)

    def _decide(self, proba: NDArray, threshold: Any) -> NDArray:
        if self.monotonicity == "increasing":
            return np.asarray(np.greater_equal(proba, threshold))
        return np.asarray(np.greater(proba, threshold))

    def _compute_losses(
        self, y: NDArray, proba: NDArray, thresholds: NDArray
    ) -> NDArray:
        """Per-example losses of shape (n_samples, n_thresholds)."""
        risk_function = self.risk_function or _false_negative_loss
        y_pred = self._decide(proba[:, np.newaxis], thresholds[np.newaxis, :])
        losses = np.empty(y_pred.shape)
        for j in range(len(thresholds)):
            loss = np.asarray(risk_function(y, y_pred[:, j].astype(int)), dtype=float)
            if loss.shape != y.shape:
                raise ValueError(
                    "risk_function must return one loss per sample, "
                    f"expected shape {y.shape} but got {loss.shape}."
                )
            losses[:, j] = loss
        tol = 1e-12
        if not np.isfinite(losses).all() or (
            np.min(losses) < -tol or np.max(losses) > self.upper_bound + tol
        ):
            raise ValueError(
                f"risk_function must return finite values in [0, {self.upper_bound}]."
            )
        sign = 1 if self.monotonicity == "increasing" else -1
        if (sign * np.diff(losses, axis=1) < -tol).any():
            raise ValueError(
                f"risk_function is not {self.monotonicity} with the threshold "
                "on calibration data, which is required by the CRC guarantee."
            )
        return losses

    def _check_parameters(self) -> None:
        if not callable(self.predict_function):
            raise TypeError(
                "predict_function must be callable. You should provide your "
                "classifier's predict_proba method."
            )
        if self.monotonicity not in ("increasing", "decreasing"):
            raise ValueError("monotonicity must be 'increasing' or 'decreasing'.")
        if self.risk_function is None and self.monotonicity != "increasing":
            raise ValueError(
                "The default false negative loss is only compatible with "
                "monotonicity='increasing'."
            )
        if not np.isfinite(self.upper_bound) or self.upper_bound <= 0:
            raise ValueError("upper_bound must be a positive finite number.")
        if not 0 < self.target_level < self.upper_bound:
            raise ValueError(
                f"target_level must be in (0, upper_bound={self.upper_bound})."
            )

    def _encode_labels(self, y: ArrayLike) -> NDArray:
        """Convert labels to 0/1, where 1 is the positive class."""
        y = column_or_1d(y)
        if self.positive_class is None:
            if not np.isin(y, [0, 1]).all():
                raise ValueError(
                    "y_calibrate must contain only 0 and 1 when positive_class is None."
                )
            return np.asarray(y, dtype=int)
        labels = np.unique(y)
        if len(labels) > 2:
            raise ValueError("y_calibrate must contain at most two distinct labels.")
        if len(labels) == 2 and self.positive_class not in labels:
            raise ValueError("positive_class is not among the labels of y_calibrate.")
        return np.asarray(y == self.positive_class, dtype=int)

    def _get_positive_proba(self, X: ArrayLike) -> NDArray:
        proba = np.asarray(self.predict_function(X), dtype=float)
        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError(
                "predict_function must return probabilities of shape (n_samples, 2)."
            )
        if not np.isfinite(proba).all() or proba.min() < 0 or proba.max() > 1:
            raise ValueError("predict_function must return probabilities in [0, 1].")
        return proba[:, 1]
