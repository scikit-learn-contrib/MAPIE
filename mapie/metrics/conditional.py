from typing import Any, Optional, cast, overload

import numpy as np
from covmetrics import CovGap, ERT, WSC
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
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


def _compute_cover_from_intervals(
    y: ArrayLike,
    y_intervals: ArrayLike,
) -> NDArray:
    """
    Compute binary coverage indicators from regression intervals.

    Parameters
    ----------
    y: ArrayLike of shape (n_samples,)
        True target values.
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
        True integer class labels.
    y_sets: ArrayLike of shape (n_samples, n_classes) or (n_samples, n_classes, 1)
        Boolean indicators of class membership in each prediction set.

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


def _compute_cover(
    y: ArrayLike,
    y_intervals: Optional[ArrayLike],
    y_sets: Optional[ArrayLike],
) -> NDArray:
    """
    Compute binary coverage indicators from intervals or prediction sets.
    """
    if y_intervals is None and y_sets is None:
        raise ValueError("Either y_intervals or y_sets must be provided.")
    elif y_intervals is not None and y_sets is None:
        return _compute_cover_from_intervals(y, y_intervals)
    elif y_intervals is None and y_sets is not None:
        return _compute_cover_from_sets(y, y_sets)
    else:
        raise ValueError("Only one of y_intervals or y_sets can be provided.")


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
    Compute the coverage gap across groups.

    This metric wraps :class:`covmetrics.CovGap`. It first converts regression
    intervals or classification prediction sets into a binary coverage vector,
    where ``1`` means that the true label is covered. It then computes the
    average absolute deviation between the empirical coverage of each group and
    the target coverage level.

    With ``weighted=False``, this function returns the unweighted coverage gap
    (CovGap), which gives every non-empty group the same weight. With
    ``weighted=True``, it returns the weighted coverage gap (WCovGap), which
    weights each group's gap by its sample proportion.

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
        Whether to compute WCovGap. If ``False``, all non-empty groups have the
        same weight. If ``True``, each group's coverage gap is weighted by its
        sample proportion, by default ``False``.

    Returns
    -------
    float
        Coverage gap across groups.

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
    _check_alpha(confidence_level)
    groups = cast(NDArray, column_or_1d(groups))
    if not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("groups should contain integer group labels.")

    cover = _compute_cover(y, y_intervals, y_sets)
    _check_arrays_length(cover, groups)
    alpha = _transform_confidence_level_to_alpha(confidence_level)
    return float(CovGap().evaluate(groups, cover, alpha, weighted))


def _check_worst_slab_coverage_inputs(
    x: ArrayLike,
    delta: float,
    n_directions: int,
) -> NDArray:
    """
    Check worst slab coverage inputs.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x should be a 2D array of shape (n_samples, n_features).")
    _check_array_nan(x)
    _check_array_inf(x)
    if not isinstance(delta, (float, int)):
        raise ValueError("delta should be a float in (0, 1).")
    if not 0 < delta < 1:
        raise ValueError("delta should be in (0, 1).")
    if not isinstance(n_directions, int):
        raise ValueError("n_directions should be an integer.")
    if n_directions < 1:
        raise ValueError("n_directions should be greater than or equal to 1.")
    return x


@overload
def worst_slab_coverage(
    x: ArrayLike,
    y: ArrayLike,
    *,
    y_intervals: ArrayLike,
    y_sets: None = None,
    delta: float = 0.1,
    n_directions: int = 1000,
    random_state: int = 42,
) -> float: ...


@overload
def worst_slab_coverage(
    x: ArrayLike,
    y: ArrayLike,
    *,
    y_intervals: None = None,
    y_sets: ArrayLike,
    delta: float = 0.1,
    n_directions: int = 1000,
    random_state: int = 42,
) -> float: ...


def worst_slab_coverage(
    x: ArrayLike,
    y: ArrayLike,
    *,
    y_intervals: Optional[ArrayLike] = None,
    y_sets: Optional[ArrayLike] = None,
    delta: float = 0.1,
    n_directions: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Compute the worst-case slab coverage.

    This metric wraps :class:`covmetrics.WSC`. It first converts regression
    intervals or classification prediction sets into a binary coverage vector,
    where ``1`` means that the true label is covered. It then samples random
    directions in the feature space and returns the lowest empirical coverage
    among slabs containing at least a fraction ``delta`` of the samples.

    Cauchois, M., Gupta, S., and Duchi, J. Knowing what You Know: valid and
    validated confidence sets in multiclass and multilabel prediction.
    Journal of Machine Learning Research, 22(81):1-42, 2021.

    Parameters
    ----------
    x: ArrayLike of shape (n_samples, n_features)
        Feature values used to define the geometric slabs.
    y: ArrayLike of shape (n_samples,)
        True labels.
    y_intervals: ArrayLike of shape (n_samples, 2) or (n_samples, 2, 1), optional
        Regression prediction intervals. Provide either ``y_intervals`` or
        ``y_sets``, but not both.
    y_sets: ArrayLike of shape (n_samples, n_classes) or (n_samples, n_classes, 1), optional
        Classification prediction sets. Provide either ``y_intervals`` or
        ``y_sets``, but not both.
    delta: float, optional
        Minimum fraction of samples required in each slab, by default ``0.1``.
    n_directions: int, optional
        Number of random directions sampled on the unit sphere, by default
        ``1000``.
    random_state: int, optional
        Seed used to sample random directions, by default ``42``.

    Returns
    -------
    float
        Worst-case slab coverage over the sampled directions.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.conditional import worst_slab_coverage
    >>> x = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([0.0, 1.0, 2.0, 3.0])
    >>> y_intervals = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 2.0], [4.0, 5.0]])
    >>> worst_slab_coverage(
    ...     x, y, y_intervals=y_intervals, delta=0.5, n_directions=5
    ... )
    0.5
    """
    x = _check_worst_slab_coverage_inputs(x, delta, n_directions)
    cover = _compute_cover(y, y_intervals, y_sets)
    _check_arrays_length(x, cover)
    return float(
        WSC().evaluate(
            x,
            cover,
            delta=delta,
            M=n_directions,
            seed=random_state,
        )
    )


def _check_excess_risk_target_coverage_inputs(
    x: ArrayLike,
    n_splits: int,
) -> NDArray:
    """
    Check excess risk target coverage inputs.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x should be a 2D array of shape (n_samples, n_features).")
    _check_array_nan(x)
    if np.isnan(x).any():
        raise ValueError("x should not contain NaN values.")
    _check_array_inf(x)
    if not isinstance(n_splits, int):
        raise ValueError("n_splits should be an integer.")
    if n_splits < 2:
        raise ValueError("n_splits should be greater than or equal to 2.")
    return x


@overload
def excess_risk_target_coverage(
    x: ArrayLike,
    y: ArrayLike,
    confidence_level: float,
    *,
    y_intervals: ArrayLike,
    y_sets: None = None,
    model_cls: Optional[type] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    loss: Optional[Any] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> float: ...


@overload
def excess_risk_target_coverage(
    x: ArrayLike,
    y: ArrayLike,
    confidence_level: float,
    *,
    y_intervals: None = None,
    y_sets: ArrayLike,
    model_cls: Optional[type] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    loss: Optional[Any] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> float: ...


def excess_risk_target_coverage(
    x: ArrayLike,
    y: ArrayLike,
    confidence_level: float,
    *,
    y_intervals: Optional[ArrayLike] = None,
    y_sets: Optional[ArrayLike] = None,
    model_cls: Optional[type] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    loss: Optional[Any] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> float:
    """
    Compute the excess risk of the target coverage.

    This metric wraps :class:`covmetrics.ERT`. It first converts regression
    intervals or classification prediction sets into a binary coverage vector,
    where ``1`` means that the true label is covered. It then trains a
    classifier to estimate conditional coverage from the features and compares
    the loss of this conditional coverage estimate to the loss of the constant
    target coverage predictor.

    The default classifier is :class:`sklearn.ensemble.HistGradientBoostingClassifier`.
    Pass ``model_cls`` and ``model_kwargs`` to use another classifier supporting
    ``fit`` and, preferably, ``predict_proba``.

    Braun, S., Holzmüller, D., Jordan, M. I., and Bach, F. Conditional Coverage
    Diagnostics for Conformal Prediction. arXiv:2512.11779, 2025.

    Parameters
    ----------
    x: ArrayLike of shape (n_samples, n_features)
        Feature values used to estimate conditional coverage.
    y: ArrayLike of shape (n_samples,)
        True labels.
    confidence_level: float
        Target coverage level.
    y_intervals: ArrayLike of shape (n_samples, 2) or (n_samples, 2, 1), optional
        Regression prediction intervals. Provide either ``y_intervals`` or
        ``y_sets``, but not both.
    y_sets: ArrayLike of shape (n_samples, n_classes) or (n_samples, n_classes, 1), optional
        Classification prediction sets. Provide either ``y_intervals`` or
        ``y_sets``, but not both.
    model_cls: type, optional
        Classifier class used by ``covmetrics.ERT`` to estimate conditional
        coverage, by default ``HistGradientBoostingClassifier``.
    model_kwargs: dict, optional
        Keyword arguments used to instantiate ``model_cls``. If ``model_cls`` is
        the default ``HistGradientBoostingClassifier`` and ``random_state`` is not
        provided in ``model_kwargs``, it is set to ``random_state``.
    n_splits: int, optional
        Number of cross-validation splits used to estimate ERT, by default ``5``.
    random_state: int, optional
        Seed used by the cross-validation splitter and, by default, by the
        default ``HistGradientBoostingClassifier``, by default ``42``.
    loss: callable, optional
        Loss function passed to ``covmetrics.ERT.evaluate``. If ``None``,
        ``covmetrics`` uses its default L1 miscoverage loss.
    fit_kwargs: dict, optional
        Extra keyword arguments passed to the classifier ``fit`` method.

    Returns
    -------
    float
        Excess risk of the target coverage.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyClassifier
    >>> from mapie.metrics.conditional import excess_risk_target_coverage
    >>> x = np.arange(12).reshape(-1, 1).astype(float)
    >>> y = np.arange(12).astype(float)
    >>> y_intervals = np.column_stack((y - 0.1, y + 0.1))
    >>> y_intervals[1::2, 0] = y[1::2] + 1.0
    >>> excess_risk_target_coverage(
    ...     x,
    ...     y,
    ...     0.5,
    ...     y_intervals=y_intervals,
    ...     model_cls=DummyClassifier,
    ...     model_kwargs={"strategy": "prior"},
    ...     n_splits=2,
    ... )
    -0.16666666666666666
    """
    _check_alpha(confidence_level)
    x = _check_excess_risk_target_coverage_inputs(x, n_splits)
    cover = _compute_cover(y, y_intervals, y_sets)
    _check_arrays_length(x, cover)
    alpha = _transform_confidence_level_to_alpha(confidence_level)

    if model_cls is None:
        model_cls = HistGradientBoostingClassifier
    model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
    if (
        model_cls is HistGradientBoostingClassifier
        and "random_state" not in model_kwargs
    ):
        model_kwargs["random_state"] = random_state
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs

    return float(
        ERT(model_cls=model_cls, **model_kwargs).evaluate(
            x,
            cover,
            alpha,
            n_splits=n_splits,
            random_state=random_state,
            loss=loss,
            **fit_kwargs,
        )
    )
