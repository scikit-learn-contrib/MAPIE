from typing import Optional, cast, Union

import numpy as np
from sklearn.utils.validation import check_array, column_or_1d

from ._typing import ArrayLike, NDArray
from .utils import (calc_bins,
                    check_array_shape_classification,
                    check_array_shape_regression,
                    check_binary_zero_one,
                    check_nb_intervals_sizes,
                    check_nb_sets_sizes,
                    check_number_bins,
                    check_split_strategy)


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
    -------
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


def _picp(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike
) -> float:
    """
    Calculate the Prediction Interval Coverage Probability (PICP).

    The Prediction Interval Coverage Probability (PICP) is a measure used to
    estimate the fraction of true labels that lie within the prediction
    intervals.

    [5] Vilde Jensen, Filippo Maria Bianchi, Stian Norman Anfinsen (2022).
    Ensemble Conformalized Quantile Regression for Probabilistic Time Series
    Forecasting.

    Parameters
    ----------
    y_true : ArrayLike
        Array of true labels.
    y_pred_low : ArrayLike
        Array of lower bounds of prediction intervals.
    y_pred_up : ArrayLike
        Array of upper bounds of prediction intervals.

    Returns
    -------
    float
        Prediction Interval Coverage Probability (PICP).

    Notes
    -----
    The PICP is calculated as follows:
    1. Determine the number of true labels that lie within the prediction
    intervals.
       This is done by counting the number of elements that satisfy both
       conditions:
       (y_true >= y_pred_low) and (y_true <= y_pred_up). Let this count be
       'in_the_range'.
    2. Calculate the coverage by dividing 'in_the_range' by the total number
    of true labels.

    Examples
    --------
    >>> y_true = [5, 7.5, 9.5, 10.5, 12.5]
    >>> y_pred_low = [4, 6, 9, 8.5, 10.5]
    >>> y_pred_up = [6, 9, 10, 12.5, 12]
    >>> y_true = np.array(y_true)
    >>> y_pred_low = np.array(y_pred_low)
    >>> y_pred_up = np.array(y_pred_up)
    >>> print(_picp(y_true, y_pred_low, y_pred_up))
    0.8
    """

    # Ensure inputs are NumPy arrays for consistent operations
    y_true = np.asarray(y_true)
    y_pred_low = np.asarray(y_pred_low)
    y_pred_up = np.asarray(y_pred_up)

    in_the_range = np.sum((np.greater_equal(y_true, y_pred_low))
                          & (np.less_equal(y_true, y_pred_up)))

    coverage = in_the_range / np.prod(y_true.shape)

    return float(coverage)


def _pinaw(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike
) -> float:
    """
    Calculate the Prediction Interval Normalized Average Width (PINAW).

    The Prediction Interval Normalized Average Width (PINAW) is a measure
    used to evaluate the average width of prediction intervals (PIs) in
    relation to the range of the true labels.


    [5] Vilde Jensen, Filippo Maria Bianchi, Stian Norman Anfinsen (2022).
    Ensemble Conformalized Quantile Regression for Probabilistic Time Series
    Forecasting.

    Parameters
    ----------
    y_true : ArrayLike
        Array of true labels.
    y_pred_low : ArrayLike
        Array of lower bounds of prediction intervals.
    y_pred_up : ArrayLike
        Array of upper bounds of prediction intervals.

    Returns
    -------
    float
        Prediction Interval Normalized Average Width (PINAW).

    Notes
    -----
    The PINAW is calculated as follows:
    1. Calculate the average width of prediction intervals by taking the mean
       of the absolute differences between the upper and lower bounds:
       avg_length.
    2. Normalize avg_length by dividing it by the range of the true labels:
       (y_true.max() - y_true.min()).
    3. The resulting value is the PINAW.

    Examples
    --------
    >>> y_true = [5, 7.5, 9.5, 10.5, 12.5]
    >>> y_pred_low = [4, 6, 9, 8.5, 10.5]
    >>> y_pred_up = [6, 9, 10, 12.5, 12]
    >>> y_true = np.array(y_true)
    >>> y_pred_low = np.array(y_pred_low)
    >>> y_pred_up = np.array(y_pred_up)
    >>> print(np.round(_pinaw(y_true, y_pred_low, y_pred_up),2))
    0.31
    """
    # Convert y_true to a NumPy array of floats
    y_true = np.array(y_true, dtype=float)

    avg_length = np.mean(np.abs(np.subtract(y_pred_up, y_pred_low)))
    avg_length = avg_length / (np.subtract(float(y_true.max()),
                                           float(y_true.min())))

    return float(avg_length)


def cwc(
    y_true: ArrayLike,
    y_pred_low: ArrayLike,
    y_pred_up: ArrayLike,
    eta: float,
    mu: float
) -> float:
    """
    Coverage width-based criterion obtained by the prediction intervals.

    The effective coverage score is a criterion used to evaluate the quality
    of prediction intervals (PIs) based on their coverage and width.


    [5] Vilde Jensen, Filippo Maria Bianchi, Stian Norman Anfinsen (2022).
    Ensemble Conformalized Quantile Regression for Probabilistic Time Series
    Forecasting.

    Parameters
    ----------
    picp : float
        Prediction interval coverage probability (PICP), which is the estimated
        fraction of true labels that lie within the prediction intervals.
    pinaw : float
        Prediction interval normalized average width (PINAW), calculated as
        the average width of the prediction intervals.
    eta : int
        A user-defined parameter that balances the contributions of PINAW and
        PICP
        in the coverage width-based criterion (CWC) calculation.
    mu : float
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
    CWC = (1 - PINAW) * exp(-eta * (PICP - mu)**2)

    The CWC penalizes under- and overcoverage in the same way and summarizes
    the quality of the prediction intervals in a single value.

    High Eta (Large Positive Value):

    When eta is a high positive value, such as 10 or 100, it will strongly
    emphasize the contribution of (1-pinaw). This means that the algorithm
    will prioritize reducing the average width of the prediction intervals
    (pinaw) over achieving a high coverage probability (picp).
    The exponential term np.exp(-eta*(picp-mu)**2) will have a sharp decline
    as picp deviates from mu. So, achieving a high picp becomes less important
    compared to minimizing pinaw.
    The impact will be narrower prediction intervals on average, which may
    result in more precise but less conservative predictions.

    Low Eta (Small Positive Value):

    When eta is a low positive value, such as 0.01 or 0.1, it will still
    prioritize reducing the average width of the prediction intervals (pinaw)
    but with less emphasis compared to higher eta values.
    The exponential term will be less steep, meaning that deviations of picp
    from mu will have a moderate impact.
    You'll get a balance between prediction precision and coverage, but the
    exact balance will depend on the specific value of eta.

    Negative Eta (Any Negative Value):

    When eta is negative, it will have a different effect on the formula.
    Negative values of eta will cause the exponential term
    np.exp(-eta*(picp-mu)**2)
    to become larger as picp deviates from mu. This means that a negative eta
    prioritizes achieving a high coverage probability (picp) over minimizing
    pinaw.
    In this case, the algorithm will aim to produce wider prediction intervals
    to ensure a higher likelihood of capturing the true values within those
    intervals, even if it sacrifices precision.
    Negative eta values might be used in scenarios where avoiding errors or
    outliers is critical.

    Null Eta (Eta = 0):

    When eta is exactly zero, both pinaw and picp will have equal importance,
    as the exponential term becomes 1. This means there is a balance
    between minimizing the average width of prediction intervals (pinaw)
    and achieving a high coverage probability (picp).
    The algorithm will aim for a trade-off between precision and coverage,
    without giving preference to either one.
    In summary, the choice of eta determines how much importance you place on
    pinaw versus picp in your coverage width-based criterion.

    A high eta emphasizes precision, a low positive eta balances precision
    and coverage, a negative eta prioritizes coverage,
    and a null eta equally values both precision and coverage.
    The specific choice of eta should align with your objectives and the
    trade-offs that are acceptable in your particular application.

    Examples
    --------
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_preds_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_preds_up = np.array([6, 9, 10, 12.5, 12])
    >>> eta = 30
    >>> mu = 0.9
    >>> print(np.round(cwc(y_true, y_preds_low, y_preds_up, eta, mu),2))
    0.51
    """

    if 0 <= mu <= 1:
        # Mu is within the valid range
        picp = _picp(y_true, y_pred_low, y_pred_up)
        pinaw = _pinaw(y_true, y_pred_low, y_pred_up)
        cwc = (1-pinaw)*np.exp(-eta*(picp-mu)**2)

        return float(cwc)
    else:
        raise ValueError("mu must be between 0 and 1")
