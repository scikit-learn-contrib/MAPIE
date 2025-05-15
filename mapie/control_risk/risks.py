from typing import cast

import numpy as np
from sklearn.utils.validation import column_or_1d

from numpy.typing import NDArray


def compute_risk_recall(
    lambdas: NDArray,
    y_pred_proba: NDArray,
    y: NDArray
) -> NDArray:
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
            "y should be a 2d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if not np.array_equal(y_pred_proba.shape[:-1], y.shape):
        raise ValueError(
            "y and y_pred_proba could not be broadcast."
        )
    lambdas = cast(NDArray, column_or_1d(lambdas))

    n_lambdas = len(lambdas)
    y_pred_proba_repeat = np.repeat(
        y_pred_proba,
        n_lambdas,
        axis=2
    )
    y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

    y_repeat = np.repeat(y[..., np.newaxis], n_lambdas, axis=2)
    risks = 1 - (
        _true_positive(y_pred_th, y_repeat) /
        y.sum(axis=1)[:, np.newaxis]
    )
    return risks


def compute_risk_precision(
    lambdas: NDArray,
    y_pred_proba: NDArray,
    y: NDArray
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
            "y should be a 2d array, got an array of shape "
            "{} instead.".format(y_pred_proba.shape)
        )
    if not np.array_equal(y_pred_proba.shape[:-1], y.shape):
        raise ValueError(
            "y and y_pred_proba could not be broadcast."
        )
    lambdas = cast(NDArray, column_or_1d(lambdas))

    n_lambdas = len(lambdas)
    y_pred_proba_repeat = np.repeat(
        y_pred_proba,
        n_lambdas,
        axis=2
    )
    y_pred_th = (y_pred_proba_repeat > lambdas).astype(int)

    y_repeat = np.repeat(y[..., np.newaxis], n_lambdas, axis=2)
    with np.errstate(divide='ignore', invalid="ignore"):
        risks = 1 - _true_positive(y_pred_th, y_repeat)/y_pred_th.sum(axis=1)
    risks[np.isnan(risks)] = 1  # nan value indicate high risks.

    return risks


def _true_positive(
    y_pred_th: NDArray,
    y_repeat: NDArray
) -> NDArray:
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
