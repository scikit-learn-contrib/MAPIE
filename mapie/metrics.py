from typing import cast

import numpy as np
from sklearn.utils.validation import column_or_1d, check_array

from ._typing import ArrayLike, NDArray


def regression_coverage_score(
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
    >>> from mapie.metrics import regression_coverage_score
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_pred_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_pred_up = np.array([6, 9, 10, 12.5, 12])
    >>> print(regression_coverage_score(y_true, y_pred_low, y_pred_up))
    0.8
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_low = cast(NDArray, column_or_1d(y_pred_low))
    y_pred_up = cast(NDArray, column_or_1d(y_pred_up))
    coverage = np.mean(
        ((y_pred_low <= y_true) & (y_pred_up >= y_true))
    )
    return float(coverage)


def classification_coverage_score(
    y_true: ArrayLike,
    y_pred_set: ArrayLike
) -> float:
    """
    Effective coverage score obtained by the prediction sets.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction sets.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True labels.
    y_pred_set : ArrayLike of shape (n_samples, n_class)
        Prediction sets given by booleans of labels.

    Returns
    -------
    float
        Effective coverage obtained by the prediction sets.

    Examples
    --------
    >>> from mapie.metrics import classification_coverage_score
    >>> import numpy as np
    >>> y_true = np.array([3, 3, 1, 2, 2])
    >>> y_pred_set = np.array([
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True],
    ...     [False,  True,  True, False],
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True]
    ... ])
    >>> print(classification_coverage_score(y_true, y_pred_set))
    0.8
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_set = cast(
        NDArray,
        check_array(
            y_pred_set, force_all_finite=True, dtype=["bool"]
        )
    )
    coverage = np.take_along_axis(
        y_pred_set, y_true.reshape(-1, 1), axis=1
    ).mean()
    return float(coverage)


def regression_mean_width_score(
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike
) -> float:
    """
    Effective mean width score obtained by the prediction intervals.

    Parameters
    ----------
    y_pred_low : ArrayLike of shape (n_samples,)
        Lower bound of prediction intervals.
    y_pred_up : ArrayLike of shape (n_samples,)
        Upper bound of prediction intervals.

    Returns
    -------
    float
        Effective mean width of the prediction intervals.

    Examples
    --------
    >>> from mapie.metrics import regression_mean_width_score
    >>> import numpy as np
    >>> y_pred_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_pred_up = np.array([6, 9, 10, 12.5, 12])
    >>> print(regression_mean_width_score(y_pred_low, y_pred_up))
    2.3
    """
    y_pred_low = cast(NDArray, column_or_1d(y_pred_low))
    y_pred_up = cast(NDArray, column_or_1d(y_pred_up))
    mean_width = np.abs(y_pred_up - y_pred_low).mean()
    return float(mean_width)


def classification_mean_width_score(y_pred_set: ArrayLike) -> float:
    """
    Mean width of prediction set output by
    :class:`mapie.classification.MapieClassifier`.

    Parameters
    ----------
    y_pred_set : ArrayLike of shape (n_samples, n_class)
        Prediction sets given by booleans of labels.

    Returns
    -------
    float
        Mean width of the prediction set.

    Examples
    --------
    >>> from mapie.metrics import classification_mean_width_score
    >>> import numpy as np
    >>> y_pred_set = np.array([
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True],
    ...     [False,  True,  True, False],
    ...     [False, False,  True,  True],
    ...     [False,  True, False,  True]
    ... ])
    >>> print(classification_mean_width_score(y_pred_set))
    2.0
    """
    y_pred_set = cast(
        NDArray,
        check_array(
            y_pred_set, force_all_finite=True, dtype=["bool"]
        )
    )
    mean_width = y_pred_set.sum(axis=1).mean()
    return float(mean_width)
