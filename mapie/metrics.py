import numpy as np


def coverage(y_true: np.ndarray, y_preds: np.ndarray) -> float:
    """
    Effective coverage obtained by the prediction intervals.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction intervals.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True labels.
    y_preds : np.ndarray of shape (n_samples, 3)
        Predictions as returned by `MapieRegressor.predict()`.

    Returns
    -------
    float
        Effective coverage obtained by the prediction intervals.

    Examples
    --------
    >>> from mapie.metrics import coverage
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_preds = np.array([
    ...    [5, 4, 6],
    ...    [7.5, 6., 9.],
    ...    [9.5, 9, 10.],
    ...    [10.5, 8.5, 12.5],
    ...    [11.5, 10.5, 12.]
    ... ])
    >>> print(coverage(y_true, y_preds))
    0.8
    """
    if not isinstance(y_true, np.ndarray):
        raise ValueError("y_true is not an np.ndarray.")
    if not isinstance(y_preds, np.ndarray):
        raise ValueError("y_preds is not an np.ndarray.")
    if y_true.shape[0] != y_preds.shape[0]:
        raise ValueError("y_true and y_preds have different lengths.")
    if y_preds.shape[1] != 3:
        raise ValueError("y_preds.shape[1] is not equal to 3.")
    return (
        (y_preds[:, 1] <= y_true) & (y_preds[:, 2] >= y_true)
    ).mean()
