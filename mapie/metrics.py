from sklearn.utils.validation import column_or_1d

from ._typing import ArrayLike


def coverage_score(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike,
) -> float:
    """
    Effective coverage score obtained by the prediction intervals.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction intervals.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True labels.
    y_pred_low : ArrayLike of shape (n_samples,)
        Lower bound of prediction intervals.
    y_pred_up : ArrayLike of shape (n_samples,)
        Upper bound of prediction intervals.

    Returns
    -------
    float
        Effective coverage obtained by the prediction intervals.

    Examples
    --------
    >>> from mapie.metrics import coverage_score
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_pred_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_pred_up = np.array([6, 9, 10, 12.5, 12])
    >>> print(coverage_score(y_true, y_pred_low, y_pred_up))
    0.8
    """
    y_true = column_or_1d(y_true)
    y_pred_low = column_or_1d(y_pred_low)
    y_pred_up = column_or_1d(y_pred_up)
    coverage = ((y_pred_low <= y_true) & (y_pred_up >= y_true)).mean()
    return float(coverage)
