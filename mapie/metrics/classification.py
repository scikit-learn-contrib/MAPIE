from typing import cast, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import column_or_1d

from mapie.utils import (
    _check_arrays_length,
    _check_array_nan,
    _check_array_inf,
    _check_array_shape_classification, _check_nb_sets_sizes, _check_number_bins,
)


def classification_mean_width_score(y_pred_set: ArrayLike) -> float:
    """
    Mean width of prediction set output by
    :class:`~mapie.classification._MapieClassifier`.

    Parameters
    ----------
    y_pred_set: NDArray of shape (n_samples, n_class, n_confidence_level)
        Prediction sets given by booleans of labels.

    Returns
    -------
    NDArray of shape (n_confidence_level,)
        Mean width of the prediction sets for each confidence level.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.classification import classification_mean_width_score
    >>> y_pred_set = np.array([
    ...     [[False, False], [False, True], [True, True]],
    ...     [[False, True], [True, False], [True, True]],
    ...     [[True, False], [True, True], [True, False]],
    ...     [[False, False], [True, True], [True, True]],
    ...     [[True, True], [False, True], [True, False]]
    ... ])
    >>> print(classification_mean_width_score(y_pred_set))
    [2.  1.8]
    """
    y_pred_set = np.asarray(y_pred_set, dtype=bool)
    _check_array_nan(y_pred_set)
    _check_array_inf(y_pred_set)
    width = y_pred_set.sum(axis=1)
    mean_width = width.mean(axis=0)
    return mean_width


def classification_coverage_score(
    y_true: NDArray,
    y_pred_set: NDArray
) -> NDArray:
    """
    Effective coverage score obtained by the prediction sets.

    The effective coverage is obtained by estimating the fraction
    of true labels that lie within the prediction sets.

    Prediction sets obtained by the ``predict`` method can be passed directly to the
    ``y_pred_set`` argument (see example below).

    Beside this intended use, this function also works with:

    - ``y_true`` of shape (n_sample,) and ``y_pred_set`` of shape (n_sample, n_class)
    - ``y_true`` of shape (n_sample, n) and ``y_pred_set`` of shape
      (n_sample, n_class, n)

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.

    y_pred_set: NDArray of shape (n_samples, n_class, n_confidence_level)
        Prediction sets with different confidence levels, given by booleans of labels
        with the ``predict`` method.

    Returns
    -------
    NDArray of shape (n_confidence_level,)
        Effective coverage obtained by the prediction sets for each confidence level.

    Examples
    --------
    >>> from mapie.metrics.classification import classification_coverage_score
    >>> from mapie.classification import SplitConformalClassifier
    >>> from mapie.utils import train_conformalize_test_split
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.neighbors import KNeighborsClassifier

    >>> X, y = make_classification(n_samples=500)
    >>> (
    ...     X_train, X_conformalize, X_test,
    ...     y_train, y_conformalize, y_test
    ... ) = train_conformalize_test_split(
    ...     X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
    ... )

    >>> mapie_classifier = SplitConformalClassifier(
    ...     estimator=KNeighborsClassifier(),
    ...     confidence_level=[0.9, 0.95, 0.99],
    ...     prefit=False,
    ... ).fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)

    >>> predicted_points, predicted_sets = mapie_classifier.predict_set(X_test)
    >>> coverage = classification_coverage_score(y_test, predicted_sets)[0]
    """
    _check_arrays_length(y_true, y_pred_set)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_pred_set)
    _check_array_inf(y_pred_set)

    y_pred_set = _check_array_shape_classification(y_true, y_pred_set)
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
    y_pred_set: NDArray of shape (n_samples, n_class, n_confidence_level)
    or (n_samples, n_class)
        Prediction sets given by booleans of labels.
    num_bins: int or None
        Number of groups. If None, one value of coverage by possible
        size of sets (n_classes +1) is computed. Should be less than the
        number of different set sizes.

    Returns
    -------
    NDArray of shape (n_confidence_level, num_bins)

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
    y_pred_set = _check_array_shape_classification(y_true, y_pred_set)

    _check_arrays_length(y_true, y_pred_set)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_pred_set)
    _check_array_inf(y_pred_set)

    sizes = np.sum(y_pred_set, axis=1)
    n_classes = y_pred_set.shape[1]
    if num_bins is None:
        bins = list(range(n_classes + 1))
    else:
        _check_nb_sets_sizes(sizes, num_bins)
        _check_number_bins(num_bins)
        bins = [
            b[0] for b in np.array_split(range(n_classes + 1), num_bins)
        ]

    digitized_sizes: NDArray = np.digitize(sizes, bins)
    coverages = np.zeros((y_pred_set.shape[2], len(bins)))
    for alpha in range(y_pred_set.shape[2]):
        indexes_bybins = [
            np.argwhere(digitized_sizes[:, alpha] == i)
            for i in range(1, len(bins)+1)
        ]

        for i, indexes in enumerate(indexes_bybins):
            coverages[alpha, i] = classification_coverage_score(
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
    Aggregate by the minimum for each confidence level the Size-Stratified Coverage [3]:
    returns the maximum violation of the conditional coverage
    (with the groups defined).

    Parameters
    ----------
    y_true: NDArray of shape (n_samples,)
        True labels.
    y_pred_set: NDArray of shape (n_samples, n_class, n_confidence_level)
    or (n_samples, n_class)
        Prediction sets given by booleans of labels.
    num_bins: int or None
        Number of groups. If None, one value of coverage by possible
        size of sets (n_classes +1) is computed. Should be less than
        the number of different set sizes.

    Returns
    -------
    NDArray of shape (n_confidence_level,)

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
    _check_arrays_length(y_true, y_pred_set)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_pred_set)
    _check_array_inf(y_pred_set)

    return np.nanmin(classification_ssc(y_true, y_pred_set, num_bins), axis=1)
