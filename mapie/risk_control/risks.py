from __future__ import annotations

from typing import Callable, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import column_or_1d


def compute_risk_recall(lambdas: NDArray, y_pred_proba: NDArray, y: NDArray) -> NDArray:
    """
    In `PrecisionRecallController` when `metric_control=recall`,
    compute the recall per observation for each different
    thresholds lambdas.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    lambdas: NDArray of shape (n_lambdas, )
        Threshold that permit to compute recall.

    Returns
    -------
    NDArray of shape (n_samples, n_labels, n_lambdas)
        Risks for each observation and each value of lambda.
    """
    if y_pred_proba.ndim != 3:
        raise ValueError(
            "y_pred_proba should be a 3d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if y.ndim != 2:
        raise ValueError(
            "y should be a 2d array, got an array of shape {} instead.".format(
                y_pred_proba.shape
            )
        )
    if not np.array_equal(y_pred_proba.shape[:-1], y.shape):
        raise ValueError("y and y_pred_proba could not be broadcast.")
    lambdas = cast(NDArray, column_or_1d(lambdas))

    n_lambdas = len(lambdas)
    y_pred_proba_repeat = np.repeat(y_pred_proba, n_lambdas, axis=2)
    y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

    y_repeat = np.repeat(y[..., np.newaxis], n_lambdas, axis=2)
    risks = 1 - (_true_positive(y_pred_th, y_repeat) / y.sum(axis=1)[:, np.newaxis])
    return risks


def compute_risk_precision(
    lambdas: NDArray, y_pred_proba: NDArray, y: NDArray
) -> NDArray:
    """
    In `PrecisionRecallController` when `metric_control=precision`,
    compute the precision per observation for each different
    thresholds lambdas.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    lambdas: NDArray of shape (n_lambdas, )
        Threshold that permit to compute precision score.

    Returns
    -------
    NDArray of shape (n_samples, n_labels, n_lambdas)
        Risks for each observation and each value of lambda.
    """
    if y_pred_proba.ndim != 3:
        raise ValueError(
            "y_pred_proba should be a 3d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if y.ndim != 2:
        raise ValueError(
            "y should be a 2d array, got an array of shape {} instead.".format(
                y_pred_proba.shape
            )
        )
    if not np.array_equal(y_pred_proba.shape[:-1], y.shape):
        raise ValueError("y and y_pred_proba could not be broadcast.")
    lambdas = cast(NDArray, column_or_1d(lambdas))

    n_lambdas = len(lambdas)
    y_pred_proba_repeat = np.repeat(y_pred_proba, n_lambdas, axis=2)
    y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

    y_repeat = np.repeat(y[..., np.newaxis], n_lambdas, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        risks = 1 - _true_positive(y_pred_th, y_repeat) / y_pred_th.sum(axis=1)
    risks[np.isnan(risks)] = 1  # nan value indicate high risks.

    return risks


def _true_positive(y_pred_th: NDArray, y_repeat: NDArray) -> NDArray:
    """
    Compute the number of true positive.

    Parameters
    ----------
    y_pred_proba : NDArray of shape (n_samples, n_labels, 1)
        Predicted probabilities for each label and each observation.

    y: NDArray of shape (n_samples, n_labels)
        True labels.

    Returns
    -------
    tp: float
        The number of true positive.
    """
    tp = (y_pred_th * y_repeat).sum(axis=1)
    return tp


class BinaryClassificationRisk:
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
    ``precision = (sum(y_pred == y_true) if y_pred == 1)/sum(y_pred == 1)``.
    Because precision is a performance metric rather than a risk, `higher_is_better`
    must be set to `True`. See the implementation of `precision` in mapie.risk_control.

    Note: any risk or performance metric that can be defined as
    ``sum(occurrence if condition) / sum(condition)`` can be theoretically controlled
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
        Whether this BinaryClassificationRisk instance is a risk
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
        self._risk_occurrence = risk_occurrence
        self._risk_condition = risk_condition
        self.higher_is_better = higher_is_better

    def get_value_and_effective_sample_size(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> Tuple[float, int]:
        """
        Computes the value of a risk given an array of ground
        truth labels and the corresponding predictions. Also returns the number of
        samples used to compute that value.

        That number can be different from the total number of samples. For example, in
        the case of precision, only the samples with positive predictions are used.

        In the case of a performance metric, this function returns 1 - perf_value.

        Parameters
        ----------
        y_true : NDArray
            NDArray of ground truth labels, of shape (n_samples,), with values in {0, 1}

        y_pred : NDArray
            NDArray of predictions, of shape (n_samples,), with values in {0, 1}

        Returns
        -------
        Tuple[float, int]
            A tuple containing the value of the risk between 0 and 1,
            and the number of effective samples used to compute that value
            (between 1 and n_samples).

            In the case of a performance metric, this function returns 1 - perf_value.

            If the risk is not defined (condition never met), the value is set to 1,
            and the number of effective samples is set to -1.
        """
        risk_occurrences = self._risk_occurrence(y_true, y_pred)
        risk_conditions = self._risk_condition(y_true, y_pred)

        effective_sample_size = len(y_true) - np.sum(~risk_conditions)
        # Casting needed for MyPy with Python 3.9
        effective_sample_size_int = cast(int, effective_sample_size)
        if effective_sample_size_int != 0.0:
            risk_sum: int = np.sum(risk_occurrences[risk_conditions])
            risk_value = risk_sum / effective_sample_size_int
            if self.higher_is_better:
                risk_value = 1 - risk_value
            return risk_value, effective_sample_size_int
        else:
            # In this case, the corresponding lambda shouldn't be considered valid.
            # In the current LTT implementation, providing n_obs=-1 will result
            # in an infinite p_value, effectively invaliding the lambda
            return 1, -1


precision = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: y_pred == 1,
    higher_is_better=True,
)

accuracy = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: np.repeat(True, len(y_true)),
    higher_is_better=True,
)

recall = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: y_true == 1,
    higher_is_better=True,
)

false_positive_rate = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == 1,
    risk_condition=lambda y_true, y_pred: y_true == 0,
    higher_is_better=False,
)

predicted_positive_fraction = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == 1,
    risk_condition=lambda y_true, y_pred: np.repeat(True, len(y_true)),
    higher_is_better=False,
)
