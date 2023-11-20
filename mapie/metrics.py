from typing import Optional, cast, Union, Tuple

import scipy
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, column_or_1d

from ._typing import ArrayLike, NDArray
from .utils import (calc_bins, check_alpha,
                    check_array_shape_classification,
                    check_array_shape_regression,
                    check_array_inf,
                    check_array_nan,
                    check_arrays_length,
                    check_binary_zero_one,
                    check_lower_upper_bounds,
                    check_nb_intervals_sizes,
                    check_nb_sets_sizes,
                    check_number_bins,
                    check_split_strategy)
from ._machine_precision import EPSILON


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
    y_true: ArrayLike of shape (n_samples,)
        True labels.
    y_pred_low: ArrayLike of shape (n_samples,)
        Lower bound of prediction intervals.
    y_pred_up: ArrayLike of shape (n_samples,)
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

    check_arrays_length(y_true, y_pred_low, y_pred_up)
    check_lower_upper_bounds(y_true, y_pred_low, y_pred_up)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_pred_low)
    check_array_inf(y_pred_low)
    check_array_nan(y_pred_up)
    check_array_inf(y_pred_up)

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
    y_true: ArrayLike of shape (n_samples,)
        True labels.
    y_pred_set: ArrayLike of shape (n_samples, n_class)
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

    check_arrays_length(y_true, y_pred_set)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_pred_set)
    check_array_inf(y_pred_set)

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
    y_pred_low: ArrayLike of shape (n_samples,)
        Lower bound of prediction intervals.
    y_pred_up: ArrayLike of shape (n_samples,)
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

    check_arrays_length(y_pred_low, y_pred_up)
    check_array_nan(y_pred_low)
    check_array_inf(y_pred_low)
    check_array_nan(y_pred_up)
    check_array_inf(y_pred_up)

    mean_width = np.abs(y_pred_up - y_pred_low).mean()
    return float(mean_width)


def classification_mean_width_score(y_pred_set: ArrayLike) -> float:
    """
    Mean width of prediction set output by
    :class:`~mapie.classification.MapieClassifier`.

    Parameters
    ----------
    y_pred_set: ArrayLike of shape (n_samples, n_class)
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
    check_array_nan(y_pred_set)
    check_array_inf(y_pred_set)
    mean_width = y_pred_set.sum(axis=1).mean()
    return float(mean_width)


def expected_calibration_error(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    num_bins: int = 50,
    split_strategy: Optional[str] = None,
) -> float:
    """
    The expected calibration error, which is the difference between
    the confidence scores and accuracy per bin [1].

    [1] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
    "Obtaining well calibrated probabilities using bayesian binning."
    Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        The target values for the calibrator.
    y_score: ArrayLike of shape (n_samples,) or (n_samples, n_classes)
        The predictions scores.
    num_bins: int
        Number of bins to make the split in the y_score. The allowed
        values are num_bins above 0.
    split_strategy: str
        The way of splitting the predictions into different bins.
        The allowed split strategies are "uniform", "quantile" and
        "array split".
    Returns
    -------
    float
        The score of ECE (Expected Calibration Error).
    """
    split_strategy = check_split_strategy(split_strategy)
    num_bins = check_number_bins(num_bins)
    y_true_ = check_binary_zero_one(y_true)
    y_scores = cast(NDArray, y_scores)

    check_arrays_length(y_true_, y_scores)
    check_array_nan(y_true_)
    check_array_inf(y_true_)
    check_array_nan(y_scores)
    check_array_inf(y_scores)

    if np.size(y_scores.shape) == 2:
        y_score = cast(
            NDArray, column_or_1d(np.nanmax(y_scores, axis=1))
        )
    else:
        y_score = cast(NDArray, column_or_1d(y_scores))

    _, bin_accs, bin_confs, bin_sizes = calc_bins(
        y_true_, y_score, num_bins, split_strategy
    )

    return np.divide(
        np.sum(bin_sizes * np.abs(bin_accs - bin_confs)),
        np.sum(bin_sizes)
    )


def top_label_ece(
    y_true: ArrayLike,
    y_scores: ArrayLike,
    y_score_arg: Optional[ArrayLike] = None,
    num_bins: int = 50,
    split_strategy: Optional[str] = None,
    classes: Optional[ArrayLike] = None,
) -> float:
    """
    The Top-Label ECE which is a method adapted to fit the
    ECE to a Top-Label setting [2].

    [2] Gupta, Chirag, and Aaditya K. Ramdas.
    "Top-label calibration and multiclass-to-binary reductions."
    arXiv preprint arXiv:2107.08353 (2021).

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        The target values for the calibrator.
    y_scores: ArrayLike of shape (n_samples, n_classes)
    or (n_samples,)
        The predictions scores, either the maximum score and the
        argmax needs to be inputted or in the form of the prediction
        probabilities.
    y_score_arg: Optional[ArrayLike] of shape (n_samples,)
        If only the maximum is provided in the y_scores, the argmax must
        be provided here. This is optional and could be directly infered
        from the y_scores.
    num_bins: int
        Number of bins to make the split in the y_score. The allowed
        values are num_bins above 0.
    split_strategy: str
        The way of splitting the predictions into different bins.
        The allowed split strategies are "uniform", "quantile" and
        "array split".
    classes: ArrayLike of shape (n_samples,)
        The different classes, in order of the indices that would be
        present in a pred_proba.

    Returns
    -------
    float
        The ECE score adapted in the top label setting.
    """
    y_scores = cast(NDArray, y_scores)
    y_true = cast(NDArray, y_true)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_scores)
    check_array_inf(y_scores)

    if y_score_arg is None:
        check_arrays_length(y_true, y_scores)
    else:
        y_score_arg = cast(NDArray, y_score_arg)
        check_array_nan(y_score_arg)
        check_array_inf(y_score_arg)
        check_arrays_length(y_true, y_scores, y_score_arg)

    ece = float(0.)
    split_strategy = check_split_strategy(split_strategy)
    num_bins = check_number_bins(num_bins)
    y_true = cast(NDArray, column_or_1d(y_true))
    if y_score_arg is None:
        y_score = cast(
            NDArray, column_or_1d(np.nanmax(y_scores, axis=1))
        )
        if classes is None:
            y_score_arg = cast(
                NDArray, column_or_1d(np.nanargmax(y_scores, axis=1))
            )
        else:
            classes = cast(NDArray, classes)
            y_score_arg = cast(
                NDArray, column_or_1d(classes[np.nanargmax(y_scores, axis=1)])
            )
    else:
        y_score = cast(NDArray, column_or_1d(y_scores))
        y_score_arg = cast(NDArray, column_or_1d(y_score_arg))
    labels = np.unique(y_score_arg)

    for label in labels:
        label_ind = np.where(label == y_score_arg)[0]
        y_true_ = np.array(y_true[label_ind] == label, dtype=int)
        ece += expected_calibration_error(
            y_true_,
            y_scores=y_score[label_ind],
            num_bins=num_bins,
            split_strategy=split_strategy
        )
    ece /= len(labels)
    return ece


def regression_coverage_score_v2(
    y_true: NDArray,
    y_intervals: NDArray,
) -> NDArray:
    """
    Effective coverage score obtained by the prediction intervals.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction intervals.

    It is different from ``regression_coverage_score`` because it uses
    directly the output of ``predict`` method and can compute the
    coverage for each alpha.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples, n_alpha) or (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha)
        Lower and upper bound of prediction intervals
        with different alpha risks.

    Returns
    -------
    NDArray of shape (n_alpha,)
        Effective coverage obtained by the prediction intervals.
    """
    check_arrays_length(y_true, y_intervals)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_intervals)
    check_array_inf(y_intervals)

    y_intervals = check_array_shape_regression(y_true, y_intervals)
    if len(y_true.shape) != 2:
        y_true = cast(NDArray, column_or_1d(y_true))
        y_true = np.expand_dims(y_true, axis=1)
    coverages = np.mean(
        np.logical_and(
            np.less_equal(y_intervals[:, 0, :], y_true),
            np.greater_equal(y_intervals[:, 1, :], y_true)
        ),
        axis=0
    )
    return coverages


def classification_coverage_score_v2(
    y_true: NDArray,
    y_pred_set: NDArray
) -> NDArray:
    """
    Effective coverage score obtained by the prediction sets.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction sets.

    It is different from ``classification_coverage_score`` because it uses
    directly the output of ``predict`` method and can compute the
    coverage for each alpha.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples, n_alpha) or (n_samples,)
        True labels.
    y_pred_set: NDArray of shape (n_samples, n_class, n_alpha)
        Prediction sets given by booleans of labels.

    Returns
    -------
    NDArray of shape (n_alpha,)
        Effective coverage obtained by the prediction sets.
    """
    check_arrays_length(y_true, y_pred_set)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_pred_set)
    check_array_inf(y_pred_set)

    y_pred_set = check_array_shape_classification(y_true, y_pred_set)
    if len(y_true.shape) != 2:
        y_true = cast(NDArray, column_or_1d(y_true))
        y_true = np.expand_dims(y_true, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    coverage = np.nanmean(
        np.take_along_axis(y_pred_set, y_true, axis=1),
        axis=0
    )
    return coverage[0]


def regression_ssc(
    y_true: NDArray,
    y_intervals: NDArray,
    num_bins: int = 3
) -> NDArray:
    """
    Compute Size-Stratified Coverage metrics proposed in [3] that is
    the conditional coverage conditioned by the size of the intervals.
    The intervals are ranked by their size (ascending) and then divided into
    num_bins groups: one value of coverage by groups is computed.

    Warning: This metric should be used only with non constant intervals
    (intervals of different sizes), with constant intervals the result
    may be misinterpreted.

    [3] Angelopoulos, A. N., & Bates, S. (2021).
    A gentle introduction to conformal prediction and
    distribution-free uncertainty quantification.
    arXiv preprint arXiv:2107.07511.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha) or (n_samples, 2)
        Prediction intervals given by booleans of labels.
    num_bins: int n
        Number of groups. Should be less than the number of different
        interval widths.

    Returns
    -------
    NDArray of shape (n_alpha, num_bins)

    Examples
    --------
    >>> from mapie.metrics import regression_ssc
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5])
    >>> y_intervals = np.array([
    ... [4, 6],
    ... [6.0, 9.0],
    ... [9, 10.0]
    ... ])
    >>> print(regression_ssc(y_true, y_intervals, num_bins=2))
    [[1. 1.]]
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_intervals = check_array_shape_regression(y_true, y_intervals)
    check_number_bins(num_bins)
    widths = np.abs(y_intervals[:, 1, :] - y_intervals[:, 0, :])
    check_nb_intervals_sizes(widths, num_bins)

    check_arrays_length(y_true, y_intervals)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_intervals)
    check_array_inf(y_intervals)

    indexes_sorted = np.argsort(widths, axis=0)
    indexes_bybins = np.array_split(indexes_sorted, num_bins, axis=0)
    coverages = np.zeros((y_intervals.shape[2], num_bins))
    for i, indexes in enumerate(indexes_bybins):
        intervals_binned = np.stack([
            np.take_along_axis(y_intervals[:, 0, :], indexes, axis=0),
            np.take_along_axis(y_intervals[:, 1, :], indexes, axis=0)
        ], axis=1)
        coverages[:, i] = regression_coverage_score_v2(y_true[indexes],
                                                       intervals_binned)

    return coverages


def regression_ssc_score(
    y_true: NDArray,
    y_intervals: NDArray,
    num_bins: int = 3
) -> NDArray:
    """
    Aggregate by the minimum for each alpha the Size-Stratified Coverage [3]:
    returns the maximum violation of the conditional coverage
    (with the groups defined).

    Warning: This metric should be used only with non constant intervals
    (intervals of different sizes), with constant intervals the result
    may be misinterpreted.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha) or (n_samples, 2)
        Prediction intervals given by booleans of labels.
    num_bins: int n
        Number of groups. Should be less than the number of different
        interval widths.

    Returns
    -------
    NDArray of shape (n_alpha,)

    Examples
    --------
    >>> from mapie.metrics import regression_ssc
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5])
    >>> y_intervals = np.array([
    ... [[4, 4], [6, 7.5]],
    ... [[6.0, 8], [9.0, 10]],
    ... [[9, 9], [10.0, 10.0]]
    ... ])
    >>> print(regression_ssc_score(y_true, y_intervals, num_bins=2))
    [1.  0.5]
    """
    return np.min(regression_ssc(y_true, y_intervals, num_bins), axis=1)


def classification_ssc(
    y_true: NDArray,
    y_pred_set: NDArray,
    num_bins: Union[int, None] = None
) -> NDArray:
    """
    Compute Size-Stratified Coverage metrics proposed in [3] that is
    the conditional coverage conditioned by the size of the predictions sets.
    The sets are ranked by their size (ascending) and then divided into
    num_bins groups: one value of coverage by groups is computed.

    [3] Angelopoulos, A. N., & Bates, S. (2021).
    A gentle introduction to conformal prediction and
    distribution-free uncertainty quantification.
    arXiv preprint arXiv:2107.07511.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_pred_set: NDArray of shape (n_samples, n_class, n_alpha)
    or (n_samples, n_class)
        Prediction sets given by booleans of labels.
    num_bins: int or None
        Number of groups. If None, one value of coverage by possible
        size of sets (n_classes +1) is computed. Should be less than the
        number of different set sizes.

    Returns
    -------
    NDArray of shape (n_alpha, num_bins)

    Examples
    --------
    >>> from mapie.metrics import classification_ssc
    >>> import numpy as np
    >>> y_true = y_true_class = np.array([3, 3, 1, 2, 2])
    >>> y_pred_set = np.array([
    ...    [True, True, True, True],
    ...    [False, True, False, True],
    ...    [True, True, True, False],
    ...    [False, False, True, True],
    ...    [True, True, False, True]])
    >>> print(classification_ssc(y_true, y_pred_set, num_bins=2))
    [[1.         0.66666667]]
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_set = check_array_shape_classification(y_true, y_pred_set)

    check_arrays_length(y_true, y_pred_set)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_pred_set)
    check_array_inf(y_pred_set)

    sizes = np.sum(y_pred_set, axis=1)
    n_classes = y_pred_set.shape[1]
    if num_bins is None:
        bins = list(range(n_classes + 1))
    else:
        check_nb_sets_sizes(sizes, num_bins)
        check_number_bins(num_bins)
        bins = [
            b[0] for b in np.array_split(range(n_classes + 1), num_bins)
        ]

    digitized_sizes = np.digitize(sizes, bins)
    coverages = np.zeros((y_pred_set.shape[2], len(bins)))
    for alpha in range(y_pred_set.shape[2]):
        indexes_bybins = [
            np.argwhere(digitized_sizes[:, alpha] == i)
            for i in range(1, len(bins)+1)
        ]

        for i, indexes in enumerate(indexes_bybins):
            coverages[alpha, i] = classification_coverage_score_v2(
                y_true[indexes],
                np.take_along_axis(
                    y_pred_set[:, :, alpha],
                    indexes,
                    axis=0
                )
            )
    return coverages


def classification_ssc_score(
    y_true: NDArray,
    y_pred_set: NDArray,
    num_bins: Union[int, None] = None
) -> NDArray:
    """
    Aggregate by the minimum for each alpha the Size-Stratified Coverage [3]:
    returns the maximum violation of the conditional coverage
    (with the groups defined).

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_pred_set: NDArray of shape (n_samples, n_class, n_alpha)
    or (n_samples, n_class)
        Prediction sets given by booleans of labels.
    num_bins: int or None
        Number of groups. If None, one value of coverage by possible
        size of sets (n_classes +1) is computed. Should be less than
        the number of different set sizes.

    Returns
    -------
    NDArray of shape (n_alpha,)

    Examples
    --------
    >>> from mapie.metrics import classification_ssc_score
    >>> import numpy as np
    >>> y_true = y_true_class = np.array([3, 3, 1, 2, 2])
    >>> y_pred_set = np.array([
    ...    [True, True, True, True],
    ...    [False, True, False, True],
    ...    [True, True, True, False],
    ...    [False, False, True, True],
    ...    [True, True, False, True]])
    >>> print(classification_ssc_score(y_true, y_pred_set, num_bins=2))
    [0.66666667]
    """
    check_arrays_length(y_true, y_pred_set)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_pred_set)
    check_array_inf(y_pred_set)

    return np.nanmin(classification_ssc(y_true, y_pred_set, num_bins), axis=1)


def _gaussian_kernel(
    x: NDArray,
    kernel_size: int
) -> NDArray:
    """
    Computes the gaussian kernel of x. (Used in hsic function)

    Parameters
    ----------
    x: NDArray
        The values from which to compute the gaussian kernel.
    kernel_size: int
        The variance (sigma), this coefficient controls the width of the curve.
    """
    norm_x = x ** 2
    dist = -2 * np.matmul(x, x.transpose((0, 2, 1))) \
        + norm_x + norm_x.transpose((0, 2, 1))
    return np.exp(-dist / kernel_size)


def hsic(
    y_true: NDArray,
    y_intervals: NDArray,
    kernel_sizes: ArrayLike = (1, 1)
) -> NDArray:
    """
    Compute the square root of the hsic coefficient. HSIC is Hilbert-Schmidt
    independence criterion that is a correlation measure. Here we use it as
    proposed in [4], to compute the correlation between the indicator of
    coverage and the interval size.

    If hsic is 0, the two variables (the indicator of coverage and the
    interval size) are independant.

    Warning: This metric should be used only with non constant intervals
    (intervals of different sizes), with constant intervals the result
    may be misinterpreted.

    [4] Feldman, S., Bates, S., & Romano, Y. (2021).
    Improving conditional coverage via orthogonal quantile regression.
    Advances in Neural Information Processing Systems, 34, 2060-2071.

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_intervals: NDArray of shape (n_samples, 2, n_alpha) or (n_samples, 2)
        Prediction sets given by booleans of labels.
    kernel_sizes: ArrayLike of size (2,)
        The variance (sigma) for each variable (the indicator of coverage and
        the interval size), this coefficient controls the width of the curve.

    Returns
    -------
    NDArray of shape (n_alpha,)
        One hsic correlation coefficient by alpha.

    Raises
    ------
    ValueError
        If kernel_sizes has a length different from 2
        and if it has negative or null values.

    Examples
    --------
    >>> from mapie.metrics import hsic
    >>> import numpy as np
    >>> y_true = np.array([9.5, 10.5, 12.5])
    >>> y_intervals = np.array([
    ... [[9, 9], [10.0, 10.0]],
    ... [[8.5, 9], [12.5, 12]],
    ... [[10.5, 10.5], [12.0, 12]]
    ... ])
    >>> print(hsic(y_true, y_intervals))
    [0.31787614 0.2962914 ]
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_intervals = check_array_shape_regression(y_true, y_intervals)

    check_arrays_length(y_true, y_intervals)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_intervals)
    check_array_inf(y_intervals)

    kernel_sizes = cast(NDArray, column_or_1d(kernel_sizes))
    if len(kernel_sizes) != 2:
        raise ValueError(
            "kernel_sizes should be an ArrayLike of length 2"
        )
    if (kernel_sizes <= 0).any():
        raise ValueError(
            "kernel_size should be positive"
        )
    n_samples, _, n_alpha = y_intervals.shape
    y_true_per_alpha = np.tile(y_true, (n_alpha, 1)).transpose()
    widths = np.expand_dims(
        np.abs(y_intervals[:, 1, :] - y_intervals[:, 0, :]).transpose(),
        axis=2
    )
    cov_ind = np.expand_dims(
        np.int_(
            ((y_intervals[:, 0, :] <= y_true_per_alpha) &
             (y_intervals[:, 1, :] >= y_true_per_alpha))
        ).transpose(),
        axis=2
    )

    k_mat = _gaussian_kernel(widths, kernel_sizes[0])
    l_mat = _gaussian_kernel(cov_ind, kernel_sizes[1])
    h_mat = np.eye(n_samples) - 1 / n_samples * np.ones((n_samples, n_samples))
    hsic_mat = np.matmul(l_mat, np.matmul(h_mat, np.matmul(k_mat, h_mat)))
    hsic_mat /= ((n_samples - 1) ** 2)
    coef_hsic = np.sqrt(np.matrix.trace(hsic_mat, axis1=1, axis2=2))

    return coef_hsic


def coverage_width_based(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike,
    eta: float,
    alpha: float
) -> float:
    """
    Coverage Width-based Criterion (CWC) obtained by the prediction intervals.

    The effective coverage score is a criterion used to evaluate the quality
    of prediction intervals (PIs) based on their coverage and width.

    Khosravi, Abbas, Saeid Nahavandi, and Doug Creighton.
    "Construction of optimal prediction intervals for load forecasting
    problems."
    IEEE Transactions on Power Systems 25.3 (2010): 1496-1503.

    Parameters
    ----------
    Coverage score : float
        Prediction interval coverage probability (Coverage score), which is
        the estimated fraction of true labels that lie within the prediction
        intervals.
    Mean Width Score : float
        Prediction interval normalized average width (Mean Width Score),
        calculated as the average width of the prediction intervals.
    eta : int
        A user-defined parameter that balances the contributions of
        Mean Width Score and Coverage score in the CWC calculation.
    alpha : float
        A user-defined parameter representing the designed confidence level of
        the PI.

    Returns
    -------
    float
        Effective coverage score (CWC) obtained by the prediction intervals.

    Notes
    -----
    The effective coverage score (CWC) is calculated using the following
    formula:
    CWC = (1 - Mean Width Score) * exp(-eta * (Coverage score - (1-alpha))**2)

    The CWC penalizes under- and overcoverage in the same way and summarizes
    the quality of the prediction intervals in a single value.

    High Eta (Large Positive Value):

    When eta is a high positive value, it will strongly
    emphasize the contribution of (1-Mean Width Score). This means that the
    algorithm will prioritize reducing the average width of the prediction
    intervals (Mean Width Score) over achieving a high coverage probability
    (Coverage score). The exponential term np.exp(-eta*(Coverage score -
    (1-alpha))**2) will have a sharp decline as Coverage score deviates
    from (1-alpha). So, achieving a high Coverage score becomes less important
    compared to minimizing Mean Width Score.
    The impact will be narrower prediction intervals on average, which may
    result in more precise but less conservative predictions.

    Low Eta (Small Positive Value):

    When eta is a low positive value, it will still
    prioritize reducing the average width of the prediction intervals
    (Mean Width Score) but with less emphasis compared to higher
    eta values.
    The exponential term will be less steep, meaning that deviations of
    Coverage score from (1-alpha) will have a moderate impact.
    You'll get a balance between prediction precision and coverage, but the
    exact balance will depend on the specific value of eta.

    Negative Eta (Any Negative Value):

    When eta is negative, it will have a different effect on the formula.
    Negative values of eta will cause the exponential term
    np.exp(-eta*(Coverage score - (1-alpha))**2) to become larger as
    Coverage score deviates from (1-alpha). This means that
    a negative eta prioritizes achieving a high coverage probability
    (Coverage score) over minimizing Mean Width Score.
    In this case, the algorithm will aim to produce wider prediction intervals
    to ensure a higher likelihood of capturing the true values within those
    intervals, even if it sacrifices precision.
    Negative eta values might be used in scenarios where avoiding errors or
    outliers is critical.

    Null Eta (Eta = 0):

    Specifically, when eta is zero, the CWC score becomes equal to
    (1 - Mean Width Score), which is equivalent to
    (1 - average width of the prediction intervals).
    Therefore, in this case, the CWC score is primarily based on the size of
    the prediction interval.

    Examples
    --------
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_preds_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_preds_up = np.array([6, 9, 10, 12.5, 12])
    >>> eta = 0.01
    >>> alpha = 0.1
    >>> cwb = coverage_width_based(y_true, y_preds_low, y_preds_up, eta, alpha)
    >>> print(np.round(cwb ,2))
    0.69
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_low = cast(NDArray, column_or_1d(y_pred_low))
    y_pred_up = cast(NDArray, column_or_1d(y_pred_up))

    check_alpha(1-alpha)

    coverage_score = regression_coverage_score(
        y_true,
        y_pred_low,
        y_pred_up
    )
    mean_width = regression_mean_width_score(
        y_pred_low,
        y_pred_up
    )
    ref_length = np.subtract(
        float(y_true.max()),
        float(y_true.min())
    )
    avg_length = mean_width / ref_length

    cwc = (1-avg_length)*np.exp(-eta*(coverage_score-(1-alpha))**2)

    return float(cwc)


def add_jitter(
    x: NDArray,
    noise_amplitude: float = 1e-8,
    random_state: Optional[Union[int, np.random.RandomState]] = None
) -> NDArray:
    """
    Add a tiny normal distributed perturbation to an array x.

    Parameters
    ----------
    x : NDArray
        The array to jitter.

    noise_amplitude : float, optional
        The tiny relative noise amplitude to add, by default 1e-8.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random sampling.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    NDArray
        The array x jittered.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics import add_jitter
    >>> x = np.array([0, 1, 2, 3, 4])
    >>> res = add_jitter(x, random_state=1)
    >>> res
    array([0.        , 0.99999999, 1.99999999, 2.99999997, 4.00000003])
    """
    n = len(x)
    random_state_np = check_random_state(random_state)
    noise = noise_amplitude * random_state_np.normal(size=n)
    x_jittered = x * (1 + noise)
    return x_jittered


def sort_xy_by_y(x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Sort two arrays x and y according to y values.

    Parameters
    ----------
    x : NDArray of size (n_samples,)
        The array to sort according to y.
    y : NDArray of size (n_samples,)
        The array to sort.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Both arrays sorted.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics import sort_xy_by_y
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([5, 4, 3, 1, 2])
    >>> x_sorted, y_sorted = sort_xy_by_y(x, y)
    >>> print(x_sorted)
    [4 5 3 2 1]
    >>> print(y_sorted)
    [1 2 3 4 5]
    """
    x = column_or_1d(x)
    y = column_or_1d(y)
    sort_index = np.argsort(y)
    x_sorted = x[sort_index]
    y_sorted = y[sort_index]
    return x_sorted, y_sorted


def cumulative_differences(
    y_true: NDArray,
    y_score: NDArray,
    noise_amplitude: float = 1e-8,
    random_state: Optional[Union[int, np.random.RandomState]] = 1
) -> NDArray:
    """
    Compute the cumulative difference between y_true and y_score, both ordered
    according to y_scores array.

    Parameters
    ----------
    y_true : NDArray of size (n_samples,)
        An array of ground truths.

    y_score : NDArray of size (n_samples,)
        An array of scores.

    noise_amplitude : float, optional
        The tiny relative noise amplitude to add, by default 1e-8.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random sampling.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    NDArray
        The mean cumulative difference between y_true and y_score.

    References
    ----------
    Arrieta-Ibarra I, Gujral P, Tannen J, Tygert M, Xu C.
    Metrics of calibration for probabilistic predictions.
    The Journal of Machine Learning Research.
    2022 Jan 1;23(1):15886-940.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics import cumulative_differences
    >>> y_true = np.array([1, 0, 0])
    >>> y_score = np.array([0.7, 0.3, 0.6])
    >>> cum_diff = cumulative_differences(y_true, y_score)
    >>> print(len(cum_diff))
    3
    >>> print(np.max(cum_diff) <= 1)
    True
    >>> print(np.min(cum_diff) >= -1)
    True
    >>> cum_diff
    array([-0.1, -0.3, -0.2])
    """
    check_arrays_length(y_true, y_score)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_score)
    check_array_inf(y_score)

    n = len(y_true)
    y_score_jittered = add_jitter(
        y_score,
        noise_amplitude=noise_amplitude,
        random_state=random_state
    )
    y_true_sorted, y_score_sorted = sort_xy_by_y(y_true, y_score_jittered)
    cumulative_differences = np.cumsum(y_true_sorted - y_score_sorted)/n
    return cumulative_differences


def length_scale(s: NDArray) -> float:
    """
    Compute the mean square root of the sum  of s * (1 - s).
    This is basically the standard deviation of the
    cumulative differences.

    Parameters
    ----------
    s : NDArray of shape (n_samples,)
        An array of scores.

    Returns
    -------
    float
        The length_scale array.

    References
    ----------
    Arrieta-Ibarra I, Gujral P, Tannen J, Tygert M, Xu C.
    Metrics of calibration for probabilistic predictions.
    The Journal of Machine Learning Research.
    2022 Jan 1;23(1):15886-940.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics import length_scale
    >>> s = np.array([0, 0, 0.4, 0.3, 0.8])
    >>> res = length_scale(s)
    >>> print(np.round(res, 2))
    0.16
    """
    n = len(s)
    length_scale = np.sqrt(np.sum(s * (1 - s)))/n
    return length_scale


def kolmogorov_smirnov_statistic(y_true: NDArray, y_score: NDArray) -> float:
    """
    Compute Kolmogorov-smirnov's statistic for calibration test.
    Also called ECCE-MAD
    (Estimated Cumulative Calibration Errors - Maximum Absolute Deviation).
    The closer to zero, the better the scores are calibrated.
    Indeed, if the scores are perfectly calibrated,
    the cumulative differences between ``y_true`` and ``y_score``
    should share the same properties of a standard Brownian motion
    asymptotically.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,)
        An array of ground truth.

    y_score : NDArray of shape (n_samples,)
        An array of scores..

    Returns
    -------
    float
        Kolmogorov-smirnov's statistic.

    References
    ----------
    Arrieta-Ibarra I, Gujral P, Tannen J, Tygert M, Xu C.
    Metrics of calibration for probabilistic predictions.
    The Journal of Machine Learning Research.
    2022 Jan 1;23(1):15886-940.

    Example
    -------
    >>> import numpy as np
    >>> from mapie.metrics import kolmogorov_smirnov_statistic
    >>> y_true = np.array([0, 1, 0, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.21, 0.9, 0.5])
    >>> print(np.round(kolmogorov_smirnov_statistic(y_true, y_score), 3))
    0.978
    """
    check_arrays_length(y_true, y_score)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_score)
    check_array_inf(y_score)

    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)

    cum_diff = cumulative_differences(y_true, y_score)
    sigma = length_scale(y_score)
    ks_stat = np.max(np.abs(cum_diff)) / sigma
    return ks_stat


def kolmogorov_smirnov_cdf(x: float) -> float:
    """
    Compute the Kolmogorov-smirnov cumulative distribution
    function (CDF) for the float x.
    This is interpreted as the CDF of the maximum absolute value
    of the standard Brownian motion over the unit interval [0, 1].
    The function is approximated by its power series, truncated so as to hit
    machine precision error.

    Parameters
    ----------
    x : float
        The float x to compute the cumulative distribution function on.

    Returns
    -------
    float
        The Kolmogorov-smirnov cumulative distribution function.

    References
    ----------
    Tygert M.
    Calibration of P-values for calibration and for deviation
    of a subpopulation from the full population.
    arXiv preprint arXiv:2202.00100.
    2022 Jan 31.

    D. A. Darling. A. J. F. Siegert.
    The First Passage Problem for a Continuous Markov Process.
    Ann. Math. Statist. 24 (4) 624 - 639, December,
    1953.

    Example
    -------
    >>> import numpy as np
    >>> from mapie.metrics import kolmogorov_smirnov_cdf
    >>> print(np.round(kolmogorov_smirnov_cdf(1), 4))
    0.3708
    """
    kmax = np.ceil(
        0.5 + x * np.sqrt(2) / np.pi * np.sqrt(np.log(4 / (np.pi*EPSILON)))
    )
    c = 0.0
    for k in range(int(kmax)):
        kplus = k + 1 / 2
        c += (-1)**k / kplus * np.exp(-kplus**2 * np.pi**2 / (2 * x**2))
    c *= 2 / np.pi
    return c


def kolmogorov_smirnov_p_value(y_true: NDArray, y_score: NDArray) -> float:
    """
    Compute Kolmogorov Smirnov p-value.
    Deduced from the corresponding statistic and CDF.
    It represents the probability of the observed statistic
    under the null hypothesis of perfect calibration.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,)
        An array of ground truth.

    y_score : NDArray of shape (n_samples,)
        An array of scores.

    Returns
    -------
    float
        The Kolmogorov Smirnov p-value.

    References
    ----------
    Tygert M.
    Calibration of P-values for calibration and for deviation
    of a subpopulation from the full population.
    arXiv preprint arXiv:2202.00100.
    2022 Jan 31.

    D. A. Darling. A. J. F. Siegert.
    The First Passage Problem for a Continuous Markov Process.
    Ann. Math. Statist. 24 (4) 624 - 639, December,
    1953.

    Example
    -------
    >>> import pandas as pd
    >>> from mapie.metrics import kolmogorov_smirnov_p_value
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_score = np.array([0.8, 0.3, 0.5, 0.5, 0.7, 0.1])
    >>> ks_p_value = kolmogorov_smirnov_p_value(y_true, y_score)
    >>> print(np.round(ks_p_value, 4))
    0.7857
    """
    check_arrays_length(y_true, y_score)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_score)
    check_array_inf(y_score)

    ks_stat = kolmogorov_smirnov_statistic(y_true, y_score)
    ks_p_value = 1 - kolmogorov_smirnov_cdf(ks_stat)
    return ks_p_value


def kuiper_statistic(y_true: NDArray, y_score: NDArray) -> float:
    """
    Compute Kuiper's statistic for calibration test.
    Also called ECCE-R (Estimated Cumulative Calibration Errors - Range).
    The closer to zero, the better the scores are calibrated.
    Indeed, if the scores are perfectly calibrated,
    the cumulative differences between ``y_true`` and ``y_score``
    should share the same properties of a standard Brownian motion
    asymptotically.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,)
        An array of ground truth.

    y_score : NDArray of shape (n_samples,)
        An array of scores.

    Returns
    -------
    float
        Kuiper's statistic.

    References
    ----------
    Arrieta-Ibarra I, Gujral P, Tannen J, Tygert M, Xu C.
    Metrics of calibration for probabilistic predictions.
    The Journal of Machine Learning Research.
    2022 Jan 1;23(1):15886-940.

    Example
    -------
    >>> import numpy as np
    >>> from mapie.metrics import kuiper_statistic
    >>> y_true = np.array([0, 1, 0, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.21, 0.9, 0.5])
    >>> print(np.round(kuiper_statistic(y_true, y_score), 3))
    0.857
    """
    check_arrays_length(y_true, y_score)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_score)
    check_array_inf(y_score)

    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    cum_diff = cumulative_differences(y_true, y_score)
    sigma = length_scale(y_score)
    ku_stat = (np.max(cum_diff) - np.min(cum_diff)) / sigma
    return ku_stat


def kuiper_cdf(x: float) -> float:
    """
    Compute the Kuiper cumulative distribution function (CDF) for the float x.
    This is interpreted as the CDF of the range
    of the standard Brownian motion over the unit interval [0, 1].
    The function is approximated by its power series, truncated so as to hit
    machine precision error.

    Parameters
    ----------
    x : float
        The float x to compute the cumulative distribution function.

    Returns
    -------
    float
        The Kuiper cumulative distribution function.

    References
    ----------
    Tygert M.
    Calibration of P-values for calibration and for deviation
    of a subpopulation from the full population.
    arXiv preprint arXiv:2202.00100.
    2022 Jan 31.

    William Feller.
    The Asymptotic Distribution of the Range of Sums of
    Independent Random Variables.
    Ann. Math. Statist. 22 (3) 427 - 432
    September, 1951.

    Example
    -------
    >>> import numpy as np
    >>> from mapie.metrics import kuiper_cdf
    >>> print(np.round(kuiper_cdf(1), 4))
    0.0634
    """
    kmax = np.ceil(
        (
            0.5 + x / (np.pi * np.sqrt(2)) *
            np.sqrt(
                np.log(
                    4 / (np.sqrt(2 * np.pi) * EPSILON) * (1 / x + x / np.pi**2)
                )
            )
        )
    )
    c = 0.0
    for k in range(int(kmax)):
        kplus = k + 1 / 2
        c += (
            (8 / x**2 + 2 / kplus**2 / np.pi**2) *
            np.exp(-2 * kplus**2 * np.pi**2 / x**2)
        )
    return c


def kuiper_p_value(y_true: NDArray, y_score: NDArray) -> float:
    """
    Compute Kuiper statistic p-value.
    Deduced from the corresponding statistic and CDF.
    It represents the probability of the observed statistic
    under the null hypothesis of perfect calibration.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,)
        An array of ground truth.

    y_score : NDArray of shape (n_samples,)
        An array of scores.

    Returns
    -------
    float
        The Kuiper p-value.

    References
    ----------
    Tygert M.
    Calibration of P-values for calibration and for deviation
    of a subpopulation from the full population.
    arXiv preprint arXiv:2202.00100.
    2022 Jan 31.

    William Feller.
    The Asymptotic Distribution of the Range of Sums of
    Independent Random Variables.
    Ann. Math. Statist. 22 (3) 427 - 432
    September, 1951.

    Example
    -------
    >>> import pandas as pd
    >>> from mapie.metrics import kuiper_p_value
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_score = np.array([0.8, 0.3, 0.5, 0.5, 0.7, 0.1])
    >>> ku_p_value = kuiper_p_value(y_true, y_score)
    >>> print(np.round(ku_p_value, 4))
    0.9684
    """
    check_arrays_length(y_true, y_score)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_score)
    check_array_inf(y_score)

    ku_stat = kuiper_statistic(y_true, y_score)
    ku_p_value = 1 - kuiper_cdf(ku_stat)
    return ku_p_value


def spiegelhalter_statistic(y_true: NDArray, y_score: NDArray) -> float:
    """
    Compute Spiegelhalter's statistic for calibration test.
    The closer to zero, the better the scores are calibrated.
    Indeed, if the scores are perfectly calibrated,
    the Brier score simplifies to an expression whose expectancy
    and variance are easy to compute. The statistic is no more that
    a z-score on this normalized expression.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,)
        An array of ground truth.

    y_score : NDArray of shape (n_samples,)
        An array of scores.

    Returns
    -------
    float
        Spiegelhalter's statistic.

    References
    ----------
    Spiegelhalter DJ.
    Probabilistic prediction in patient management and clinical trials.
    Statistics in medicine.
    1986 Sep;5(5):421-33.

    Example
    -------
    >>> import numpy as np
    >>> from mapie.metrics import spiegelhalter_statistic
    >>> y_true = np.array([0, 1, 0, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.21, 0.9, 0.5])
    >>> print(np.round(spiegelhalter_statistic(y_true, y_score), 3))
    -0.757
    """
    check_arrays_length(y_true, y_score)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_score)
    check_array_inf(y_score)

    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    numerator = np.sum(
        (y_true - y_score) * (1 - 2 * y_score)
    )
    denominator = np.sqrt(
        np.sum(
            (1 - 2 * y_score) ** 2 * y_score * (1 - y_score)
        )
    )
    sp_stat = numerator/denominator
    return sp_stat


def spiegelhalter_p_value(y_true: NDArray, y_score: NDArray) -> float:
    """
    Compute Spiegelhalter statistic p-value.
    Deduced from the corresponding statistic and CDF,
    which is no more than the normal distribution.
    It represents the probability of the observed statistic
    under the null hypothesis of perfect calibration.

    Parameters
    ----------
    y_true : NDArray of shape (n_samples,)
        An array of ground truth.

    y_score : NDArray of shape (n_samples,)
        An array of scores.

    Returns
    -------
    float
        The Spiegelhalter statistic p_value.

    References
    ----------
    Spiegelhalter DJ.
    Probabilistic prediction in patient management and clinical trials.
    Statistics in medicine.
    1986 Sep;5(5):421-33.

    Example
    -------
    >>> import numpy as np
    >>> from mapie.metrics import spiegelhalter_p_value
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_score = np.array([0.8, 0.3, 0.5, 0.5, 0.7, 0.1])
    >>> sp_p_value = spiegelhalter_p_value(y_true, y_score)
    >>> print(np.round(sp_p_value, 4))
    0.8486
    """
    check_arrays_length(y_true, y_score)
    check_array_nan(y_true)
    check_array_inf(y_true)
    check_array_nan(y_score)
    check_array_inf(y_score)
    sp_stat = spiegelhalter_statistic(y_true, y_score)
    sp_p_value = 1 - scipy.stats.norm.cdf(sp_stat)
    return sp_p_value
