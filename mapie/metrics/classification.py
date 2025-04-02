from typing import cast, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import column_or_1d, check_array

from mapie.utils import (
    check_arrays_length,
    check_array_nan,
    check_array_inf,
    check_array_shape_classification, check_nb_sets_sizes, check_number_bins,
)


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
    >>> from mapie.metrics.classification import classification_coverage_score
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
    >>> from mapie.metrics.classification import classification_mean_width_score
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
    >>> from mapie.metrics.classification import classification_ssc
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
    >>> from mapie.metrics.classification import classification_ssc_score
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
