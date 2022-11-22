from typing import cast, Union, Optional, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
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


def get_binning_groups(
    y_score: NDArray,
    num_bins: int,
    strategy: str,
) -> Union[NDArray, NDArray]:
    """_summary_
    Parameters
    ----------
    y_score : _type_
        The scores given from the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.
    Returns
    -------
    _type_
        Returns the upper and lower bound values for the bins and the indices
        of the y_score that belong to each bins.
    """
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, num_bins)
        bins = np.percentile(y_score, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, num_bins)
    elif strategy == "array split":
        bin_groups = np.array_split(y_score, num_bins)
        bins = np.sort(
            np.array(
                [bin_group.max() for bin_group in bin_groups[:-1]]+[np.inf]
                )
        )
    else:
        ValueError("We don't have this strategy")
    bin_assignments = np.digitize(y_score, bins, right=True)
    return bins, bin_assignments


def calc_bins(
    y_score: NDArray,
    y_true: NDArray,
    num_bins: int,
    strategy: str,
) -> Union[NDArray, NDArray, NDArray, NDArray]:
    """
    For each bins, calculate the accuracy, average confidence and size.
    Parameters
    ----------
    y_score : _type_
        The scores given from the calibrator.
    y_true : _type_
        The "true" values, target for the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.
    Returns
    -------
    _type_
        Multiple arrays, the upper and lower bound of each bins,
        indices of y that belong to each bins, the accuracy,
        confidence and size of each bins.
    """
    bins, binned = get_binning_groups(y_score, num_bins, strategy)
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(y_score[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = np.divide(
                np.sum(y_true[binned == bin]),
                bin_sizes[bin],
            )
            bin_confs[bin] = np.divide(
                np.sum(y_score[binned == bin]),
                bin_sizes[bin],
            )
    return bins, bin_accs, bin_confs, bin_sizes


def check_split_strategy(
    strategy: Optional[str]
) -> str:
    valid_split_strategies = ["uniform", "quantile", "array split"]
    if strategy is None:
        strategy = "uniform"
    if strategy not in valid_split_strategies:
        raise ValueError(
            "Please provide a valid splitting strategy."
        )
    return strategy


def check_number_bins(
    num_bins: int
) -> None:
    if isinstance(num_bins, int) is False:
        raise ValueError(
            "Please provide a bin number as an integer."
        )
    elif num_bins < 1:
        raise ValueError(
            """
            Please provide a bin number greater than
            or equal to  1.
            """
        )


def expected_calibration_error(
    y_scores: ArrayLike,
    y_true: ArrayLike,
    num_bins: int = 50,
    split_strategy: Optional[str] = None,
) -> float:
    """
    Function to get the different metrics of interest.
    Parameters
    ----------
    y_score : _type_
        The predictions scores.
    y_true : _type_
        The "true" values, target for the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.
    Returns
    -------
    _type_
        The score of ECE (Expected Calibration Error)
    """
    split_strategy = check_split_strategy(split_strategy)
    check_number_bins(num_bins)
    y_true = cast(NDArray, column_or_1d(y_true))
    y_scores = cast(NDArray, column_or_1d(y_scores))

    if np.size(y_scores.shape) == 2:
        y_score = cast(
            NDArray, column_or_1d(np.max(y_scores, axis=1))
        )
    else:
        y_score = cast(NDArray, column_or_1d(y_scores))

    _, bin_accs, bin_confs, bin_sizes = calc_bins(
        y_score, y_true, num_bins, split_strategy
    )

    return np.divide(
        np.sum(bin_sizes * np.abs(bin_accs - bin_confs)),
        np.sum(bin_sizes)
    )


def top_label_ece(
    y_scores: ArrayLike,
    y_true: ArrayLike,
    num_bins: int = 50,
    split_strategy: Optional[str] = None,
) -> float:
    ece = float(0)
    split_strategy = check_split_strategy(split_strategy)
    check_number_bins(num_bins)
    y_true = cast(NDArray, column_or_1d(y_true))
    y_score = cast(
        NDArray, column_or_1d(np.max(y_scores, axis=1))
    )
    y_score_arg = cast(
        NDArray, column_or_1d(np.argmax(y_scores, axis=1))
    )
    labels = np.unique(y_score_arg)

    for label in labels:
        label_ind = np.where(label == y_score_arg)[0]
        ece += expected_calibration_error(
            y_scores=y_score[label_ind],
            y_true=np.array(
                y_true[label_ind] == (label + 1),
                dtype=int
            ),
            num_bins=num_bins,
            split_strategy=split_strategy
        )
    ece /= len(labels)
    return ece


def draw_reliability_graph(
    y_scores: ArrayLike,
    y_true: ArrayLike,
    num_bins: int = 50,
    split_strategy: str = "uniform",
    title: str = None,
    axs: plt.Axes = None,
):
    """
    Plotting the accuracy and confidence per bins and showing
    the values of ECE and MCE.
    Parameters
    ----------
    y_score : _type_
        The scores given from the calibrator.
    y_true : _type_
        The "true" values, target for the calibrator.
    num_bins : _type_
        Number of bins to make the split in the y_score.
    strategy : _type_
        The way of splitting the predictions into different bins.
    title : _type_
        Title to give to the graph
    axs : _type_, optional
        If you want to plot multiple graph next to one another, by default None
    """
    split_strategy = check_split_strategy(split_strategy)
    check_number_bins(num_bins)
    y_true = cast(NDArray, column_or_1d(y_true))
    y_score = cast(
        NDArray, column_or_1d(np.max(y_scores, axis=1))
    )

    bins, bin_accs, _, _ = calc_bins(
        y_score, y_true, num_bins, split_strategy
    )

    if axs is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        ax = axs

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Accuracy')
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    ax.bar(
        bins, bins, width=1/(bins.shape[0]+1),
        alpha=0.3, edgecolor='black', color='r', hatch='\\'
    )
    ax.bar(
        bins, bin_accs, width=1/(bins.shape[0]+1),
        alpha=1, edgecolor='black', color='b'
    )
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)
    ax.set_title(title)

    if axs is None:
        plt.show()
