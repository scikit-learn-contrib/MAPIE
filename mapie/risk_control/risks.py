from __future__ import annotations

import warnings
from typing import Callable, List, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class _BaseRisk:
    """
    Base class factoring out the logic shared between :class:`BinaryRisk` and
    :class:`ContinuousRisk`.

    Subclasses must implement :meth:`_compute_values_and_effective_mask`, which
    returns, for each sample, (i) the per-sample risk value and (ii) whether the
    sample is effective (i.e. counts toward the risk average).

    Parameters
    ----------
    higher_is_better : bool
        Whether this risk instance is a risk (``False``) or a performance metric
        (``True``). When ``True``, aggregated values and risk sequences are
        returned as ``1 - value`` so that the output always behaves like a risk.

    Attributes
    ----------
    higher_is_better : bool
        See params.
    """

    def __init__(self, higher_is_better: bool):
        self.higher_is_better = higher_is_better

    @staticmethod
    def _warn_if_nan_values(values: NDArray) -> None:
        if np.isnan(values).any():
            warnings.warn(
                "NaN values detected in per-sample risk values returned by "
                "_compute_values_and_effective_mask. The aggregated risk or risk "
                "sequence may contain NaN values.",
                UserWarning,
                stacklevel=3,
            )

    def _compute_values_and_effective_mask(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Return the per-sample risk values and a boolean mask indicating which
        samples are "effective" (contribute to the risk average).

        The aggregated risk is
        ``sum(values[effective_mask]) / sum(effective_mask)``.
        """
        raise NotImplementedError  # pragma: no cover

    def get_value_and_effective_sample_size(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> Tuple[float, int]:
        """
        Computes the value of a risk given an array of ground truth labels and
        the corresponding predictions. Also returns the number of samples used
        to compute that value.

        That number can be different from the total number of samples. For
        example, in the case of precision, only the samples with positive
        predictions are used. For continuous risks like MAE/MSE, every sample
        is effective.

        In the case of a performance metric, this function returns
        ``1 - perf_value``.

        Parameters
        ----------
        y_true : NDArray
            NDArray of ground truth labels, of shape (n_samples,).

        y_pred : NDArray
            NDArray of predictions, of shape (n_samples,).

        Returns
        -------
        Tuple[float, int]
            A tuple containing the value of the risk and the number of
            effective samples used to compute that value (between 1 and
            n_samples).

            In the case of a performance metric, this function returns
            ``1 - perf_value``.

            If the risk is not defined (no effective sample), the value is set
            to 1, and the number of effective samples is set to -1.
        """
        values, effective_mask = self._compute_values_and_effective_mask(y_true, y_pred)
        self._warn_if_nan_values(values)

        effective_sample_size = int(np.sum(effective_mask))
        if effective_sample_size > 0:
            risk_sum = float(np.sum(values[effective_mask]))
            risk_value = risk_sum / effective_sample_size
            if self.higher_is_better:
                risk_value = 1 - risk_value
            return risk_value, effective_sample_size
        else:
            # In this case, the corresponding lambda shouldn't be considered valid.
            # In the current LTT implementation, providing n_obs=-1 will result
            # in an infinite p_value, effectively invaliding the lambda
            return 1, -1

    def get_risk_sequence(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> NDArray:
        """
        Returns the sequence of per-sample risks restricted to the effective
        samples.

        Parameters
        ----------
        y_true : NDArray
            NDArray of ground truth labels, of shape (n_samples,).

        y_pred : NDArray
            NDArray of predictions, of shape (n_samples,).

        Returns
        -------
        NDArray
            Per-sample risk values restricted to the effective samples.
        """
        values, effective_mask = self._compute_values_and_effective_mask(y_true, y_pred)
        self._warn_if_nan_values(values)
        risk_sequence = values[effective_mask]
        if self.higher_is_better:
            risk_sequence = 1 - risk_sequence
        return risk_sequence


class BinaryRisk(_BaseRisk):
    """
    Define a risk (or a performance metric) to be used with the
    BinaryClassificationController. Predefined instances are implemented,
    see :data:`mapie.risk_control.precision`, :data:`mapie.risk_control.recall`,
    :data:`mapie.risk_control.accuracy`,
    :data:`mapie.risk_control.false_positive_rate`, and
    :data:`mapie.risk_control.predicted_positive_fraction`.

    Here, a binary classification risk (or performance) is defined by an occurrence and
    a condition. Let's take the example of precision. Precision is the sum of true
    positives over the total number of predicted positives. In other words, precision is
    the average of correct predictions (occurrence) given that those predictions
    are positive (condition). Programmatically,
    `precision = (sum(y_pred == y_true) if y_pred == 1)/sum(y_pred == 1)`.
    Because precision is a performance metric rather than a risk, `higher_is_better`
    must be set to `True`. See the implementation of `precision` in mapie.risk_control.

    Note: any risk or performance metric that can be defined as
    `sum(occurrence if condition) / sum(condition)` can be theoretically controlled
    with the BinaryClassificationController, thanks to the LearnThenTest framework [1]
    and the binary Hoeffding-Bentkus p-values implemented in MAPIE.

    Note: by definition, the value of the risk (or performance metric) here is always
    between 0 and 1.

    Parameters
    ----------
    risk_occurrence : Callable[[int, int], bool]
        A function defining the occurrence of the risk for a given sample.
        Must take y_true and y_pred as input and return a boolean.

    risk_condition : Callable[[int, int], bool]
        A function defining the condition of the risk for a given sample,
        Must take y_true and y_pred as input and return a boolean.

    higher_is_better : bool
        Whether this BinaryRisk instance is a risk
        (higher_is_better=False) or a performance metric (higher_is_better=True).

    Attributes
    ----------
    higher_is_better : bool
        See params.

    References
    ----------
    [1] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al.
    "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022)
    """

    def __init__(
        self,
        risk_occurrence: Callable[
            [NDArray[np.integer], NDArray[np.integer]], NDArray[np.bool_]
        ],
        risk_condition: Callable[
            [NDArray[np.integer], NDArray[np.integer]], NDArray[np.bool_]
        ],
        higher_is_better: bool,
    ):
        super().__init__(higher_is_better=higher_is_better)
        self._risk_occurrence = risk_occurrence
        self._risk_condition = risk_condition

    def _compute_values_and_effective_mask(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        risk_occurrences = self._risk_occurrence(y_true, y_pred).astype(int)
        risk_conditions = self._risk_condition(y_true, y_pred)
        return risk_occurrences, risk_conditions


class BinaryClassificationRisk(BinaryRisk):
    """
    Deprecated alias for :class:`BinaryRisk`.

    Use ``BinaryRisk`` instead.
    """

    def __init__(
        self,
        risk_occurrence: Callable[
            [NDArray[np.integer], NDArray[np.integer]], NDArray[np.bool_]
        ],
        risk_condition: Callable[
            [NDArray[np.integer], NDArray[np.integer]], NDArray[np.bool_]
        ],
        higher_is_better: bool,
    ):
        warnings.warn(
            "`BinaryClassificationRisk` is deprecated and will be removed in a "
            "future release. Use `BinaryRisk` instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(risk_occurrence, risk_condition, higher_is_better)


BinaryRiskNames = Literal[
    "precision",
    "recall",
    "accuracy",
    "fpr",
    "predicted_positive_fraction",
    "positive_predictive_value",
    "negative_predictive_value",
    "abstention_rate",
]
BinaryRiskLike = Union[
    BinaryRisk,
    BinaryRiskNames,
    List[BinaryRisk],
    List[BinaryRiskNames],
    List[Union[BinaryRisk, BinaryRiskNames]],
]

ContinuousRiskNames = Literal["mae", "mse"]
ContinuousRiskLike = Union[
    "ContinuousRisk",
    ContinuousRiskNames,
    List["ContinuousRisk"],
    List[ContinuousRiskNames],
    List[Union["ContinuousRisk", ContinuousRiskNames]],
]

RiskLike = Union[BinaryRiskLike, ContinuousRiskLike]


precision = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred.ravel() == y_true.ravel(),
    risk_condition=lambda y_true, y_pred: y_pred.ravel() == 1,
    higher_is_better=True,
)

accuracy = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: np.repeat(True, len(y_true)),
    higher_is_better=True,
)

recall = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred.ravel() == y_true.ravel(),
    risk_condition=lambda y_true, y_pred: y_true.ravel() == 1,
    higher_is_better=True,
)

false_positive_rate = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == 1,
    risk_condition=lambda y_true, y_pred: y_true == 0,
    higher_is_better=False,
)

predicted_positive_fraction = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == 1,
    risk_condition=lambda y_true, y_pred: np.repeat(True, len(y_true)),
    higher_is_better=False,
)

positive_predictive_value = precision
ppv = positive_predictive_value

negative_predictive_value = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: y_pred == 0,
    higher_is_better=True,
)

npv = negative_predictive_value

abstention_rate = BinaryRisk(
    risk_occurrence=lambda y_true, y_pred: np.isnan(y_pred),
    risk_condition=lambda y_true, y_pred: np.repeat(True, len(y_true)),
    higher_is_better=False,
)


_best_predict_param_choice_map = {
    precision: recall,
    recall: precision,
    accuracy: accuracy,
    false_positive_rate: recall,
}


binary_risk_choice_map = {
    "precision": precision,
    "recall": recall,
    "accuracy": accuracy,
    "fpr": false_positive_rate,
    "predicted_positive_fraction": predicted_positive_fraction,
    "positive_predictive_value": positive_predictive_value,
    "negative_predictive_value": negative_predictive_value,
    "abstention_rate": abstention_rate,
}


class ContinuousRisk(_BaseRisk):
    """
    Define a continuous risk (or performance metric) to be used with a risk
    controller, for problems where predictions and targets are real-valued
    (e.g. regression).

    A continuous risk is defined by a single per-sample function mapping the
    ground-truth and predicted values to a non-negative per-sample risk value.
    The aggregated risk is the mean of these per-sample values across all
    samples. Unlike :class:`BinaryRisk`, there is no separate "condition": every
    sample counts (``risk_occurrence`` is not needed), so the effective sample
    size is always equal to ``n_samples``.

    Typical instances are :data:`mapie.risk_control.mae` (mean absolute error)
    and :data:`mapie.risk_control.mse` (mean squared error).

    Note: for theoretical risk control guarantees (Hoeffding-Bentkus), the
    per-sample risk values must be bounded. The caller is responsible for
    ensuring this, e.g. by clipping residuals to a known range.

    Parameters
    ----------
    risk_condition : Callable[[NDArray[float], NDArray[float]], NDArray[float]]
        A function defining the per-sample risk value. Must take ``y_true`` and
        ``y_pred`` as input (arrays of floats of shape ``(n_samples,)``) and
        return an array of floats of the same shape. For example, MAE uses
        ``lambda y_true, y_pred: np.abs(y_true - y_pred)``.

    higher_is_better : bool, default=False
        Whether this ContinuousRisk instance is a risk (``False``) or a
        performance metric (``True``). When ``True``, aggregated values are
        returned as ``1 - value``; the per-sample values are then expected to
        live in ``[0, 1]``.

    Attributes
    ----------
    higher_is_better : bool
        See params.
    """

    def __init__(
        self,
        risk_condition: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
        higher_is_better: bool = False,
    ):
        super().__init__(higher_is_better=higher_is_better)
        self._risk_condition = risk_condition

    def _compute_values_and_effective_mask(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        values = np.asarray(self._risk_condition(y_true, y_pred), dtype=float).ravel()
        effective_mask = np.ones(values.shape, dtype=bool)
        return values, effective_mask


mean_absolute_error = ContinuousRisk(
    risk_condition=lambda y_true, y_pred: np.abs(y_true.ravel() - y_pred.ravel()),
    higher_is_better=False,
)
mae = mean_absolute_error

mean_squared_error = ContinuousRisk(
    risk_condition=lambda y_true, y_pred: (y_true.ravel() - y_pred.ravel()) ** 2,
    higher_is_better=False,
)
mse = mean_squared_error


continuous_risk_choice_map = {
    "mae": mae,
    "mse": mse,
}
