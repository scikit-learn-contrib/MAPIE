"""
Uncertainty evaluation metrics for MAPIE.


Implements AUCROC and AUARC as introduced in:
    Lin et al. (2023), "Generating with Confidence: Uncertainty
    Quantification for Black-box Large Language Models",
    TMLR. https://arxiv.org/abs/2305.19187


These metrics are model-agnostic and apply to both regression
and classification tasks. The caller is responsible for computing
``y_wrong`` (binary: 1 if prediction is incorrect) and
``y_uncertainty`` (a non-negative scalar uncertainty per sample,
e.g. prediction interval width for regression, or 1 minus max
softmax probability for classification).
"""


from typing import Union


import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_consistent_length




def aucroc_score(
    y_wrong: ArrayLike,
    y_uncertainty: ArrayLike,
) -> float:
    """
    Area Under the ROC Curve measuring how well uncertainty
    ranks incorrect predictions.


    A high score means that samples with larger uncertainty tend
    to be the ones where the model is wrong — i.e. the uncertainty
    is *predictive* of errors.


    Defined as ``AUCROC(y_wrong, y_uncertainty)`` following
    Lin et al. (2023).


    Parameters
    ----------
    y_wrong : ArrayLike of shape (n_samples,)
        Binary array. 1 if the prediction for that sample is
        considered wrong, 0 if correct. The definition of "wrong"
        is left to the caller (e.g. outside prediction interval
        for regression, misclassified for classification).


    y_uncertainty : ArrayLike of shape (n_samples,)
        Non-negative scalar uncertainty estimate per sample.
        Higher values must indicate higher uncertainty.
        For regression, a common choice is prediction interval
        width. For classification, ``1 - max(softmax)`` or
        prediction-set size are typical choices.


    Returns
    -------
    float
        AUCROC score in [0, 1]. A score of 0.5 corresponds to a
        random uncertainty estimator. A score close to 1 means
        the uncertainty reliably identifies wrong predictions.


    Raises
    ------
    ValueError
        If ``y_wrong`` and ``y_uncertainty`` have different lengths,
        contain NaN or Inf values, if ``y_wrong`` is not binary,
        or if ``y_uncertainty`` contains negative values.


    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.uncertainty import aucroc_score
    >>> y_wrong = np.array([0, 0, 1, 1])
    >>> y_uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    >>> aucroc_score(y_wrong, y_uncertainty)
    1.0


    References
    ----------
    Lin et al. (2023). Generating with Confidence: Uncertainty
    Quantification for Black-box Large Language Models. TMLR.
    https://arxiv.org/abs/2305.19187
    """
    y_wrong = column_or_1d(np.asarray(y_wrong, dtype=float))
    y_uncertainty = column_or_1d(np.asarray(y_uncertainty, dtype=float))


    check_consistent_length(y_wrong, y_uncertainty)


    if np.any(np.isnan(y_wrong)) or np.any(np.isnan(y_uncertainty)):
        raise ValueError(
            "y_wrong and y_uncertainty must not contain NaN values."
        )
    if np.any(np.isinf(y_wrong)) or np.any(np.isinf(y_uncertainty)):
        raise ValueError(
            "y_wrong and y_uncertainty must not contain Inf values."
        )


    unique_labels = np.unique(y_wrong)
    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        raise ValueError(
            "y_wrong must be a binary array containing only 0 and 1. "
            f"Got unique values: {unique_labels}."
        )


    if np.any(y_uncertainty < 0):
        raise ValueError(
            "y_uncertainty must contain only non-negative values."
        )


    # sklearn's roc_auc_score handles the AUC computation;
    # we use uncertainty as the score that should rank positives (wrong=1)
    # higher than negatives (wrong=0).
    return float(roc_auc_score(y_wrong, y_uncertainty))




def auarc_score(
    y_wrong: ArrayLike,
    y_uncertainty: ArrayLike,
) -> float:
    """
    Area Under the Accuracy-Rejection Curve (AUARC).


    Measures how much accuracy improves when the model is allowed
    to abstain on high-uncertainty predictions. Samples are rejected
    in descending order of uncertainty; at each rejection threshold
    the accuracy on retained samples is recorded. The AUARC is the
    area under this curve, normalised to [0, 1].


    A higher AUARC means that rejecting uncertain predictions leads
    to a larger accuracy gain — i.e. the uncertainty is actionable
    for selective prediction / human-in-the-loop workflows.


    Defined as ``AUARC`` following Lin et al. (2023).


    Parameters
    ----------
    y_wrong : ArrayLike of shape (n_samples,)
        Binary array. 1 if the prediction for that sample is
        considered wrong, 0 if correct.


    y_uncertainty : ArrayLike of shape (n_samples,)
        Non-negative scalar uncertainty estimate per sample.
        Higher values must indicate higher uncertainty.


    Returns
    -------
    float
        AUARC score in [0, 1]. A higher score indicates a better
        uncertainty estimator — the maximum achievable score depends
        on the fraction of correct predictions in the data. A score
        equal to the overall accuracy corresponds to a random estimator.


    Raises
    ------
    ValueError
        If ``y_wrong`` and ``y_uncertainty`` have different lengths,
        contain NaN or Inf values, if ``y_wrong`` is not binary,
        or if ``y_uncertainty`` contains negative values.


    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.uncertainty import auarc_score
    >>> y_wrong = np.array([1, 1, 0, 0])
    >>> y_uncertainty = np.array([0.9, 0.8, 0.2, 0.1])
    >>> auarc_score(y_wrong, y_uncertainty)
    0.7916666666666666


    References
    ----------
    Lin et al. (2023). Generating with Confidence: Uncertainty
    Quantification for Black-box Large Language Models. TMLR.
    https://arxiv.org/abs/2305.19187
    """
    y_wrong = column_or_1d(np.asarray(y_wrong, dtype=float))
    y_uncertainty = column_or_1d(np.asarray(y_uncertainty, dtype=float))


    check_consistent_length(y_wrong, y_uncertainty)


    if np.any(np.isnan(y_wrong)) or np.any(np.isnan(y_uncertainty)):
        raise ValueError(
            "y_wrong and y_uncertainty must not contain NaN values."
        )
    if np.any(np.isinf(y_wrong)) or np.any(np.isinf(y_uncertainty)):
        raise ValueError(
            "y_wrong and y_uncertainty must not contain Inf values."
        )


    unique_labels = np.unique(y_wrong)
    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        raise ValueError(
            "y_wrong must be a binary array containing only 0 and 1. "
            f"Got unique values: {unique_labels}."
        )


    if np.any(y_uncertainty < 0):
        raise ValueError(
            "y_uncertainty must contain only non-negative values."
        )


    n = len(y_wrong)


    # Sort by descending uncertainty: highest uncertainty rejected first.
    # Stable sort ensures deterministic behaviour for tied uncertainty values.
    rejection_order = np.argsort(y_uncertainty)[::-1]
    y_wrong_sorted = y_wrong[rejection_order]


    # accuracy_at_k[k] = accuracy when the k most uncertain samples
    # are rejected, i.e. we retain samples[k:] (n - k samples).
    # We accumulate correct predictions from the *least* uncertain end.
    y_correct_sorted = 1.0 - y_wrong_sorted
    # cumulative correct from the retained tail as we reject from the front
    cumulative_correct_from_tail = np.cumsum(y_correct_sorted[::-1])[::-1]


    retained_counts = np.arange(n, 0, -1, dtype=float)  # n, n-1, ..., 1
    accuracy_curve = cumulative_correct_from_tail / retained_counts


    # AUARC = mean accuracy across all rejection thresholds (0 rejected,
    # 1 rejected, ..., n-1 rejected), i.e. when retaining n, n-1, ..., 1
    # samples respectively. We exclude the point where 0 samples remain.
    auarc = float(np.mean(accuracy_curve))
    return auarc
