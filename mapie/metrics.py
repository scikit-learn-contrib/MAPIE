import numpy as np
from typing import List, Union, Any
import pandas as pd


def coverage(
    y_true: Union[List[Any], np.ndarray, pd.DataFrame],
    y_preds: Union[List[Any], np.ndarray, pd.DataFrame]
) -> np.float64:
    """
    Effective coverage obtained by the prediction intervals.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction intervals.

    Parameters
    ----------
    y_true : Union[List, np.ndarray, pd.DataFrame] of shape (n_samples,)
        True labels.
    y_preds : Union[List, np.ndarray, pd.DataFrame] of shape (n_samples, 3)
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
    if isinstance(y_true, List):
        y_true = np.stack(y_true)
    if isinstance(y_preds, List):
        y_preds = np.stack(y_preds, axis=0)
    if y_true.shape[0] != y_preds.shape[0]:
        raise ValueError("y_true and y_preds have different lengths.")
    if y_preds.shape[1] != 3:
        raise ValueError("y_preds.shape[1] is not equal to 3.")
    return np.float64((
        (y_preds[:, 1] <= y_true) & (y_preds[:, 2] >= y_true)
    ).mean())
