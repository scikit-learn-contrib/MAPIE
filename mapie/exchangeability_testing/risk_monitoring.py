from numpy.typing import NDArray

from mapie.exchangeability_testing.confidence_bounds import (
    conjugate_mixture_empirical_bernstein_bound,
)
from mapie.risk_control.risks import Risk


class RiskMonitoring:
    """
    Risk monitoring.

    Parameters:
    ----------


    Attributes:
    ----------


    Examples:
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from mapie.risk_control import BinaryClassificationController, precision

    >>> X, y = make_classification(
    ...     n_features=2,
    ...     n_redundant=0,
    ...     n_informative=2,
    ...     n_clusters_per_class=1,
    ...     n_classes=2,
    ...     random_state=42,
    ...     class_sep=2.0
    ... )
    >>> X_train, X_temp, y_train, y_temp = train_test_split(
    ...     X, y, test_size=0.4, random_state=42
    ... )
    >>> X_test, X_online, y_test, y_online = train_test_split(
    ...     X_temp, y_temp, test_size=0.1, random_state=42
    ... )

    >>> clf = LogisticRegression().fit(X_train, y_train)

    >>> risk_monitoring = RiskMonitoring(
    ...     risk="precision",
    ...     confidence_level=0.95,
    ...     reference_risk=0.1,
    ...     tolerance=0.05,
    ...     warn=True
    ... )

    >>> y_pred = clf.predict_proba(X_test.reshape(1, -1))[0, 1]
    >>> risk_monitoring.compute_threshold(y_test, y_pred)
    >>> y_pred_online = clf.predict_proba(X_online.reshape(1, -1))[0, 1]
    >>> risk_monitoring.update_online_risk(y_online, y_pred_online)
    >>> risk_monitoring.summary()
    """

    def __init__(
        self,
        risk: Risk,
        confidence_level: float = 0.95,
        reference_risk=None,
        tolerance=0.05,
        tolerance_type="absolute",
        threshold=None,
        warn=True,
    ):
        self.risk = risk
        self.tolerance = tolerance
        self.tolerance_type = tolerance_type
        self.warn = warn
        self.reference_risk = reference_risk
        self.threshold = threshold

        delta = 1 - confidence_level
        self.delta_reference = delta / 2
        self.delta_online = delta / 2

        self.online_risk_sequence_history = np.array([], dtype=float)

        # Initialize other necessary attributes for the test

    @property
    def harmful_shift_detected(self):
        if len(self.online_risk_sequence_history) == 0:
            raise ValueError(
                "Online risk lower limit must be computed with update_online_risk before checking for harmful shift."
            )
        if self.threshold is None:
            raise ValueError(
                "Threshold must be computed with compute_threshold or set at initialization before checking for harmful shift."
            )
        return self.online_risk_lower_bound_sequence[-1] > self.threshold

    def _compute_risk_sequence(self, y_true: NDArray, y_pred: NDArray) -> NDArray:
        # TODO: à faire dans risks.py
        risk_occurrences = self.risk._risk_occurrence(y_true, y_pred)
        risk_conditions = self.risk._risk_condition(y_true, y_pred)
        risks = risk_occurrences[risk_conditions]
        return risks

    def compute_threshold(self, y_true: NDArray, y_pred: NDArray) -> "RiskMonitoring":
        if self.threshold is not None:
            raise ValueError("Threshold is already computed.")

        reference_risk_sequence = self._compute_risk_sequence(y_true, y_pred)

        self.reference_risk_upper_bound = hoeffding_upper_limit(
            reference_risk_sequence, self.delta_reference
        )

        if self.tolerance_type == "absolute":
            self.threshold = self.reference_risk_upper_bound + self.tolerance
        elif self.tolerance_type == "relative":
            self.threshold = self.reference_risk_upper_bound * (1 + self.tolerance)
        else:
            raise ValueError(
                "Invalid tolerance type. Must be 'absolute' or 'relative'."
            )

        return self

    def update_online_risk(self, y_true: NDArray, y_pred: NDArray) -> "RiskMonitoring":
        if self.threshold is None:
            raise ValueError(
                "Threshold must be computed with compute_threshold or set at initialization before updating the online risk"
            )

        new_risk_sequence = self._compute_risk_sequence(y_true, y_pred)
        self.online_risk_sequence_history = np.concatenate(
            [self.online_risk_sequence_history, new_risk_sequence]
        )

        # in the current implementation, the bound is recomputed from scratch with the full history
        self.online_risk_lower_bound_sequence = (
            conjugate_mixture_empirical_bernstein_bound(
                self.online_risk_sequence_history,
                v_opt=1,
                alpha=self.delta_online,
                bound_side="lower",
            )
        )

        if self.harmful_shift_detected and self.warn:
            warnings.warn(
                f"Harmful shift detected. The last value of the online risk lower bound ({self.online_risk_lower_bound_sequence[-1]:.3f}) is greater than the threshold ({self.threshold:.3f})."
            )

        return self

    def summary(self):
        pass
