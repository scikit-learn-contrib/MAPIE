import warnings
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from mapie.exchangeability_testing.confidence_bounds import (
    conjugate_mixture_empirical_bernstein_bound,
    hoeffding_bound,
)
from mapie.risk_control.risks import BinaryClassificationRisk, RiskLike, risk_choice_map


class RiskMonitoring:
    """
    Monitor a risk on an online stream relative to a reference set.

    The class first estimates an acceptable risk threshold on reference data and
    then updates a time-uniform lower confidence bound on the online risk as new
    observations arrive. A harmful shift is detected when the latest online
    lower bound exceeds the threshold.

    Parameters
    ----------
    risk : RiskLike
        Risk to monitor. If a string is provided, it must be one of the keys in
        :data:`mapie.risk_control.risks.risk_choice_map`.
    test_level : float, default=0.05
        Level used to test the hypothesis that the online risk is greater than the reference risk.
        The probability that the test gives a false positive is at most test_level (type I error).
    tolerance : float, default=0.05
        Margin applied to the reference upper confidence bound to define the
        monitoring threshold.
    tolerance_type : {"absolute", "relative"}, default="absolute"
        Whether `tolerance` is added to the reference upper bound or applied as
        a multiplicative factor.
    threshold : Optional[float], default=None
        Precomputed monitoring threshold. If provided, `compute_threshold` must
        not be called.
    warn : bool, default=True
        Whether to emit a warning when a harmful shift is detected.

    Attributes
    ----------
    risk : BinaryClassificationRisk
        Resolved risk object used internally.
    threshold : Optional[float]
        Monitoring threshold used to flag harmful shifts.
    reference_risk_upper_bound : Optional[float]
        Upper confidence bound estimated on the reference risk, available after
        `compute_threshold`.
    online_risk_sequence_history : NDArray[np.float64]
        Concatenated sequence of observed online risk values.
    online_risk_lower_bound_sequence_history : NDArray[np.float64]
        History of online lower confidence bounds.
    online_risk_lower_bound_latest : Optional[float]
        Latest value of the online lower confidence bound.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from mapie.exchangeability_testing import RiskMonitoring

    >>> X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, n_informative=2, random_state=42, class_sep=2.0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=42
    ... )
    >>> clf = LogisticRegression().fit(X_train, y_train)
    >>> monitor = RiskMonitoring(risk="accuracy", warn=False)
    >>> y_pred = clf.predict(X_test)
    >>> _ = monitor.compute_threshold(y_test, y_pred)
    >>> X_online, y_online = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, random_state=42, class_sep=0.3)
    >>> y_pred_online = clf.predict(X_online)
    >>> _ = monitor.update_online_risk(y_online, y_pred_online)
    >>> print(monitor.harmful_shift_detected)
    True

    References
    ----------
    [1] Aleksandr Podkopaev and Aaditya Ramdas. Tracking the risk of a deployed
    model and detecting harmful distribution shifts.
    International Conference on Learning Representations, 2022.
    """

    def __init__(
        self,
        risk: RiskLike,
        test_level: float = 0.05,
        tolerance: float = 0.05,
        tolerance_type: Literal["absolute", "relative"] = "absolute",
        threshold: Optional[float] = None,
        warn: bool = True,
    ) -> None:
        try:
            resolved_risk = risk_choice_map[risk] if isinstance(risk, str) else risk
        except KeyError as e:
            raise ValueError(
                "When risk is provided as a string, it must be one of: "
                f"{list(risk_choice_map.keys())}"
            ) from e
        if not isinstance(resolved_risk, BinaryClassificationRisk):
            raise TypeError(
                "risk must be a single BinaryClassificationRisk instance or a "
                "supported risk name."
            )
        self.risk: BinaryClassificationRisk = resolved_risk
        self.tolerance = tolerance
        self.tolerance_type = tolerance_type
        self.warn = warn
        self.threshold = threshold

        if not (0.0 < test_level < 1.0):
            raise ValueError("test_level must be in (0, 1).")
        self.test_level_reference = test_level / 2
        self.test_level_online = test_level / 2

        self.reference_risk_upper_bound: Optional[float] = None
        self.online_risk_sequence_history: NDArray[np.float64] = np.array(
            [], dtype=float
        )
        self.online_risk_lower_bound_sequence_history: NDArray[np.float64] = np.array(
            [], dtype=float
        )
        self.online_risk_lower_bound_latest: Optional[float] = None

    @property
    def harmful_shift_detected(self) -> bool:
        """Whether the latest online lower bound exceeds the threshold."""
        if self.online_risk_lower_bound_latest is None:
            raise ValueError(
                "Online risk lower bound must be computed with update_online_risk before checking for harmful shift."
            )
        if self.threshold is None:
            raise ValueError(
                "Threshold must be computed with compute_threshold or set at initialization before checking for harmful shift."
            )
        return bool(self.online_risk_lower_bound_latest > self.threshold)

    def compute_threshold(self, y_true: NDArray, y_pred: NDArray) -> "RiskMonitoring":
        """
        Estimate the monitoring threshold from reference predictions.
        Data can be the test set on which the model is evaluated before deployment.

        Parameters
        ----------
        y_true : NDArray
            Ground-truth binary labels for the reference data.
        y_pred : NDArray
            Predicted binary labels for the reference data.

        Returns
        -------
        RiskMonitoring
            The fitted instance.
        """
        if self.threshold is not None:
            warnings.warn(
                "Threshold is already computed and will be replaced.",
                UserWarning,
            )

        reference_risk_sequence = self.risk.get_risk_sequence(y_true, y_pred)
        if reference_risk_sequence.size == 0:
            raise ValueError(
                "Reference risk is undefined because no samples satisfy the risk condition."
            )

        self.reference_risk_upper_bound = hoeffding_bound(
            reference_risk_sequence,
            self.test_level_reference,
            bound_side="upper",
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
        """
        Update the online risk history and its lower confidence bound.

        Parameters
        ----------
        y_true : NDArray
            Ground-truth binary labels for the newly observed online data.
        y_pred : NDArray
            Predicted binary labels for the newly observed online data.

        Returns
        -------
        RiskMonitoring
            The updated instance.
        """
        if self.threshold is None:
            raise ValueError(
                "Threshold must be computed with compute_threshold or set at initialization before updating the online risk"
            )

        new_risk_sequence = self.risk.get_risk_sequence(y_true, y_pred)
        if new_risk_sequence.size == 0:
            return self
        self.online_risk_sequence_history = np.concatenate(
            [self.online_risk_sequence_history, new_risk_sequence]
        )

        # in the current implementation, the bound is recomputed from scratch with the full history
        new_risk_lower_bound_sequence = conjugate_mixture_empirical_bernstein_bound(
            self.online_risk_sequence_history,
            v_opt=1,
            alpha=self.test_level_online,
            bound_side="lower",
        )

        self.online_risk_lower_bound_sequence_history = np.asarray(
            new_risk_lower_bound_sequence, dtype=float
        )
        self.online_risk_lower_bound_latest = (
            self.online_risk_lower_bound_sequence_history[-1]
        )

        if self.harmful_shift_detected and self.warn:
            warnings.warn(
                f"Harmful shift detected. The last value of the online risk lower bound ({self.online_risk_lower_bound_latest:.3f}) is greater than the threshold ({self.threshold:.3f})."
            )

        return self

    def summary(self) -> None:
        """Placeholder for a future summary API."""
        pass
