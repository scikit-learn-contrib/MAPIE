from typing import Optional, cast, overload

import numpy as np
from covmetrics import CovGap
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import column_or_1d

from mapie.utils import (
    _check_alpha,
    _check_array_inf,
    _check_array_nan,
    _check_array_shape_classification,
    _check_array_shape_regression,
    _check_arrays_length,
    _transform_confidence_level_to_alpha,
)


def _check_one_coverage_input(
    y_intervals: Optional[ArrayLike],
    y_sets: Optional[ArrayLike],
) -> None:
    """
    Check that exactly one coverage input is provided.
    """
    if y_intervals is None and y_sets is None:
        raise ValueError("Either y_intervals or y_sets must be provided.")
    if y_intervals is not None and y_sets is not None:
        raise ValueError("Only one of y_intervals or y_sets can be provided.")


def _compute_cover_from_intervals(
    y: ArrayLike,
    y_intervals: ArrayLike,
) -> NDArray:
    """
    Compute binary coverage indicators from regression intervals.

    Parameters
    ----------
    y: ArrayLike of shape (n_samples,)
        True labels.
    y_intervals: ArrayLike of shape (n_samples, 2) or (n_samples, 2, 1)
        Regression prediction intervals.

    Returns
    -------
    NDArray of shape (n_samples,)
        Binary coverage values. A value of ``1`` means that the true label is
        covered by the interval.
    """
    y = cast(NDArray, column_or_1d(y))
    y_intervals = np.asarray(y_intervals)
    y_intervals = _check_array_shape_regression(y, y_intervals)
    if y_intervals.shape[-1] > 1:
        raise ValueError("y_intervals should contain only one confidence level.")
    _check_arrays_length(y, y_intervals)
    _check_array_nan(y)
    _check_array_inf(y)
    _check_array_nan(y_intervals)
    _check_array_inf(y_intervals)
    y_intervals = y_intervals.squeeze(axis=2)
    y_low = y_intervals[:, 0]
    y_high = y_intervals[:, 1]
    cover = (y >= y_low) & (y <= y_high)
    return cover.astype(int)


def _compute_cover_from_sets(
    y: ArrayLike,
    y_sets: ArrayLike,
) -> NDArray:
    """
    Compute binary coverage indicators from classification prediction sets.

    Parameters
    ----------
    y: ArrayLike of shape (n_samples,)
        True labels.
    y_sets: ArrayLike of shape (n_samples, n_classes) or (n_samples, n_classes, 1)
        Classification prediction sets.

    Returns
    -------
    NDArray of shape (n_samples,)
        Binary coverage values. A value of ``1`` means that the true label is
        covered by the prediction set.
    """
    y = cast(NDArray, column_or_1d(y))
    y_sets = np.asarray(y_sets, dtype=bool)
    y_sets = _check_array_shape_classification(y, y_sets)
    if y_sets.shape[-1] > 1:
        raise ValueError("y_sets should contain only one confidence level.")
    _check_arrays_length(y, y_sets)
    _check_array_nan(y)
    _check_array_inf(y)
    _check_array_nan(y_sets)
    _check_array_inf(y_sets)
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y should contain integer class labels.")
    if (np.min(y) < 0) or (np.max(y) >= y_sets.shape[1]):
        raise ValueError("y contains class labels outside y_sets.")
    y_sets = y_sets.squeeze(axis=2)
    y = np.expand_dims(y, axis=1)
    cover = np.take_along_axis(y_sets, y, axis=1).squeeze()
    return cover.squeeze().astype(int)


@overload
def coverage_gap(
    y: ArrayLike,
    groups: ArrayLike,
    confidence_level: float,
    *,
    y_intervals: ArrayLike,
    y_sets: None = None,
    weighted: bool = False,
) -> float: ...


@overload
def coverage_gap(
    y: ArrayLike,
    groups: ArrayLike,
    confidence_level: float,
    *,
    y_intervals: None = None,
    y_sets: ArrayLike,
    weighted: bool = False,
) -> float: ...


def coverage_gap(
    y: ArrayLike,
    groups: ArrayLike,
    confidence_level: float,
    *,
    y_intervals: Optional[ArrayLike] = None,
    y_sets: Optional[ArrayLike] = None,
    weighted: bool = False,
) -> float:
    """
    Compute the conditional coverage gap across groups.

    This metric wraps :class:`covmetrics.CovGap`. It first converts regression
    intervals or classification prediction sets into a binary coverage vector,
    where ``1`` means that the true label is covered, then evaluates the
    deviation from the target miscoverage level in each group.

    Ding, T., Angelopoulos, A., Bates, S., Jordan, M., and Tibshirani, R. J.
    Class-conditional conformal prediction with many classes. In Advances in
    Neural Information Processing Systems, 2023.

    Parameters
    ----------
    y: ArrayLike of shape (n_samples,)
        True labels.
    groups: ArrayLike of shape (n_samples,)
        Group membership of each sample, each group corresponding to a value.
    confidence_level: float
        Target coverage level.
    y_intervals: ArrayLike of shape (n_samples, 2) or (n_samples, 2, 1), optional
        Regression prediction intervals. Provide either ``y_intervals`` or
        ``y_sets``, but not both.
    y_sets: ArrayLike of shape (n_samples, n_classes) or (n_samples, n_classes, 1), optional
        Classification prediction sets. Provide either ``y_intervals`` or
        ``y_sets``, but not both.
    weighted: bool, optional
        Whether to compute the weighted version of the coverage gap, in which case each group's
        coverage gap will be be scaled by its number of members, by default ``False``.

    Returns
    -------
    float
        Conditional coverage gap.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.conditional import coverage_gap
    >>> y = np.array([0.0, 1.0, 2.0, 3.0])
    >>> groups = np.array([0, 0, 1, 1])
    >>> y_intervals = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [4.0, 5.0]])
    >>> coverage_gap(y, groups, 0.75, y_intervals=y_intervals)
    0.25
    """
    _check_one_coverage_input(y_intervals, y_sets)
    _check_alpha(confidence_level)
    groups = cast(NDArray, column_or_1d(groups))
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("groups should contain integer group labels.")

    if y_intervals is not None:
        cover = _compute_cover_from_intervals(y, y_intervals)
    else:
        assert y_sets is not None
        cover = _compute_cover_from_sets(y, y_sets)
    _check_arrays_length(cover, groups)
    alpha = _transform_confidence_level_to_alpha(confidence_level)
    return float(CovGap().evaluate(groups, cover, alpha, weighted))
