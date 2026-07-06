"""
Uncertainty evaluation metrics for MAPIE.

Implements AUROC and AUARC as model-agnostic metrics for
evaluating how well uncertainty estimates rank or reject
incorrect predictions.

These metrics apply to both regression and classification.
The caller is responsible for computing ``correctness``
(binary: 1 if the prediction is correct) and ``confidence``
(a scalar where higher values indicate greater confidence /
lower uncertainty — e.g. 1 / prediction interval width for
regression, or max softmax probability for classification).
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_consistent_length


def auroc(
    correctness: ArrayLike,
    confidence: ArrayLike,
) -> float:
    """
    Area Under the ROC Curve measuring how well confidence
    ranks correct predictions.

    A high score means that samples with higher confidence tend
    to be the ones where the model is correct — i.e. the confidence
    is *predictive* of correctness.

    Defined as ``AUROC(correctness, confidence)``.

    Parameters
    ----------
    correctness : ArrayLike of shape (n_samples,)
        Binary array. 1 if the prediction for that sample is
        considered correct, 0 if incorrect.

    confidence : ArrayLike of shape (n_samples,)
        Scalar confidence estimate per sample.
        Higher values must indicate higher confidence
        (lower uncertainty).

    Returns
    -------
    float
        AUROC score in [0, 1]. A score of 0.5 corresponds to a
        random confidence estimator. A score close to 1 means
        the confidence reliably identifies correct predictions.

    Raises
    ------
    ValueError
        If ``correctness`` and ``confidence`` have different
        lengths, contain NaN or Inf values, or if ``correctness``
        is not binary.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.uncertainty import auroc
    >>> correctness = np.array([1, 1, 0, 0])
    >>> confidence = np.array([0.9, 0.8, 0.2, 0.1])
    >>> auroc(correctness, confidence)
    1.0
    """
    correctness_arr = column_or_1d(np.asarray(correctness, dtype=float))
    confidence_arr = column_or_1d(np.asarray(confidence, dtype=float))

    check_consistent_length(correctness_arr, confidence_arr)

    if np.any(np.isnan(correctness_arr)) or np.any(np.isnan(confidence_arr)):
        raise ValueError("correctness and confidence must not contain NaN values.")
    if np.any(np.isinf(correctness_arr)) or np.any(np.isinf(confidence_arr)):
        raise ValueError("correctness and confidence must not contain Inf values.")

    unique_labels = np.unique(correctness_arr)
    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        raise ValueError(
            "correctness must be a binary array containing only 0 and 1. "
            f"Got unique values: {unique_labels}."
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "correctness must contain both 0 and 1 values to compute AUROC."
        )

    return float(roc_auc_score(correctness_arr, confidence_arr))


def auarc(
    correctness: ArrayLike,
    confidence: ArrayLike,
) -> float:
    """
    Area Under the Accuracy-Rejection Curve (AUARC).

    Measures how much accuracy improves when the model is allowed
    to abstain on low-confidence predictions. Samples are retained
    in descending order of confidence; at each retention threshold
    the accuracy on retained samples is recorded. The AUARC is the
    area under this curve, normalised to [0, 1].

    A higher AUARC means that rejecting low-confidence predictions
    leads to a larger accuracy gain — i.e. the confidence is
    actionable for selective prediction / human-in-the-loop
    workflows.

    Parameters
    ----------
    correctness : ArrayLike of shape (n_samples,)
        Binary array. 1 if the prediction for that sample is
        considered correct, 0 if incorrect.

    confidence : ArrayLike of shape (n_samples,)
        Scalar confidence estimate per sample.
        Higher values must indicate higher confidence
        (lower uncertainty).

    Returns
    -------
    float
        AUARC score in [0, 1]. A higher score indicates a better
        confidence estimator — the maximum achievable score depends
        on the fraction of correct predictions in the data. A score
        equal to the overall accuracy corresponds to a random estimator.

    Raises
    ------
    ValueError
        If ``correctness`` and ``confidence`` have different
        lengths, contain NaN or Inf values, or if ``correctness``
        is not binary.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.uncertainty import auarc
    >>> correctness = np.array([0, 0, 1, 1])
    >>> confidence = np.array([0.1, 0.2, 0.8, 0.9])
    >>> auarc(correctness, confidence)
    0.7916666666666666

    References
    ----------
    Nadeem, M. S. A., Zucker, J.-D., and Hanczar, B. (2009).
    "Accuracy-rejection curves (ARCs) for comparing classification
    methods with a reject option." Proceedings of Machine Learning
    Research, 8, 65-81.
    """
    correctness_arr = column_or_1d(np.asarray(correctness, dtype=float))
    confidence_arr = column_or_1d(np.asarray(confidence, dtype=float))

    check_consistent_length(correctness_arr, confidence_arr)

    if np.any(np.isnan(correctness_arr)) or np.any(np.isnan(confidence_arr)):
        raise ValueError("correctness and confidence must not contain NaN values.")
    if np.any(np.isinf(correctness_arr)) or np.any(np.isinf(confidence_arr)):
        raise ValueError("correctness and confidence must not contain Inf values.")

    unique_labels = np.unique(correctness_arr)
    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        raise ValueError(
            "correctness must be a binary array containing only 0 and 1. "
            f"Got unique values: {unique_labels}."
        )

    n = len(correctness_arr)

    # Sort by descending confidence: retain highest confidence first.
    order = np.argsort(-confidence_arr, kind="stable")
    correctness_sorted = correctness_arr[order]

    # accuracy_at_k[k] = accuracy when we retain the top k most
    # confident samples (k = n, n-1, ..., 1).
    cumulative_correct = np.cumsum(correctness_sorted)
    retained_counts = np.arange(1, n + 1, dtype=float)
    accuracy_curve = cumulative_correct / retained_counts

    return float(np.mean(accuracy_curve))
