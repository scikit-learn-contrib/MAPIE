from typing import Optional, cast

import numpy as np
from sklearn.utils.validation import check_array, column_or_1d

from ._typing import ArrayLike, NDArray
from .utils import (calc_bins, check_binary_zero_one, check_number_bins,
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
    :class:`~mapie.classification.MapieClassifier`.

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
    y_true : ArrayLike of shape (n_samples,)
        The target values for the calibrator.
    y_score : ArrayLike of shape (n_samples,) or (n_samples, n_classes)
        The predictions scores.
    num_bins : int
        Number of bins to make the split in the y_score. The allowed
        values are num_bins above 0.
    split_strategy : str
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
    y_true : ArrayLike of shape (n_samples,)
        The target values for the calibrator.
    y_scores : ArrayLike of shape (n_samples, n_classes)
    or (n_samples,)
        The predictions scores, either the maximum score and the
        argmax needs to be inputted or in the form of the prediction
        probabilities.
    y_score_arg : Optional[ArrayLike] of shape (n_samples,)
        If only the maximum is provided in the y_scores, the argmax must
        be provided here. This is optional and could be directly infered
        from the y_scores.
    num_bins : int
        Number of bins to make the split in the y_score. The allowed
        values are num_bins above 0.
    split_strategy : str
        The way of splitting the predictions into different bins.
        The allowed split strategies are "uniform", "quantile" and
        "array split".
    classes : ArrayLike of shape (n_samples,)
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
