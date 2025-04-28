import scipy
from numpy.typing import ArrayLike
from sklearn.utils import check_random_state
from mapie._machine_precision import EPSILON
from numpy.typing import NDArray
from mapie.utils import (
    _calc_bins,
    _check_array_inf,
    _check_array_nan,
    _check_arrays_length,
    _check_binary_zero_one,
    _check_number_bins,
    _check_split_strategy,
)


import numpy as np
from sklearn.utils.validation import column_or_1d


from typing import Tuple, cast, Optional, Union


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
    y_scores: ArrayLike of shape (n_samples,) or (n_samples, n_classes)
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
    split_strategy = _check_split_strategy(split_strategy)
    num_bins = _check_number_bins(num_bins)
    y_true_ = _check_binary_zero_one(y_true)
    y_scores = cast(NDArray, y_scores)

    _check_arrays_length(y_true_, y_scores)
    _check_array_nan(y_true_)
    _check_array_inf(y_true_)
    _check_array_nan(y_scores)
    _check_array_inf(y_scores)

    if np.size(y_scores.shape) == 2:
        y_score = cast(
            NDArray, column_or_1d(np.nanmax(y_scores, axis=1))
        )
    else:
        y_score = cast(NDArray, column_or_1d(y_scores))

    _, bin_accs, bin_confs, bin_sizes = _calc_bins(
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
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_scores)
    _check_array_inf(y_scores)

    if y_score_arg is None:
        _check_arrays_length(y_true, y_scores)
    else:
        y_score_arg = cast(NDArray, y_score_arg)
        _check_array_nan(y_score_arg)
        _check_array_inf(y_score_arg)
        _check_arrays_length(y_true, y_scores, y_score_arg)

    ece = float(0.)
    split_strategy = _check_split_strategy(split_strategy)
    num_bins = _check_number_bins(num_bins)
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
    >>> from mapie.metrics.calibration import add_jitter
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
    >>> from mapie.metrics.calibration import sort_xy_by_y
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
    >>> from mapie.metrics.calibration import cumulative_differences
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
    _check_arrays_length(y_true, y_score)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_score)
    _check_array_inf(y_score)

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
    >>> from mapie.metrics.calibration import length_scale
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

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.calibration import kolmogorov_smirnov_statistic
    >>> y_true = np.array([0, 1, 0, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.21, 0.9, 0.5])
    >>> print(np.round(kolmogorov_smirnov_statistic(y_true, y_score), 3))
    0.978
    """
    _check_arrays_length(y_true, y_score)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_score)
    _check_array_inf(y_score)

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

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.calibration import kolmogorov_smirnov_cdf
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

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mapie.metrics.calibration import kolmogorov_smirnov_p_value
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_score = np.array([0.8, 0.3, 0.5, 0.5, 0.7, 0.1])
    >>> ks_p_value = kolmogorov_smirnov_p_value(y_true, y_score)
    >>> print(np.round(ks_p_value, 4))
    0.7857
    """
    _check_arrays_length(y_true, y_score)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_score)
    _check_array_inf(y_score)

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

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.calibration import kuiper_statistic
    >>> y_true = np.array([0, 1, 0, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.21, 0.9, 0.5])
    >>> print(np.round(kuiper_statistic(y_true, y_score), 3))
    0.857
    """
    _check_arrays_length(y_true, y_score)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_score)
    _check_array_inf(y_score)

    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    cum_diff = cumulative_differences(y_true, y_score)
    sigma = length_scale(y_score)
    ku_stat = (np.max(cum_diff) - np.min(cum_diff)) / sigma  # type: ignore
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

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.calibration import kuiper_cdf
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

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mapie.metrics.calibration import kuiper_p_value
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_score = np.array([0.8, 0.3, 0.5, 0.5, 0.7, 0.1])
    >>> ku_p_value = kuiper_p_value(y_true, y_score)
    >>> print(np.round(ku_p_value, 4))
    0.9684
    """
    _check_arrays_length(y_true, y_score)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_score)
    _check_array_inf(y_score)

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

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.calibration import spiegelhalter_statistic
    >>> y_true = np.array([0, 1, 0, 1, 0])
    >>> y_score = np.array([0.1, 0.9, 0.21, 0.9, 0.5])
    >>> print(np.round(spiegelhalter_statistic(y_true, y_score), 3))
    -0.757
    """
    _check_arrays_length(y_true, y_score)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_score)
    _check_array_inf(y_score)

    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    numerator: float = np.sum(
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

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.calibration import spiegelhalter_p_value
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_score = np.array([0.8, 0.3, 0.5, 0.5, 0.7, 0.1])
    >>> sp_p_value = spiegelhalter_p_value(y_true, y_score)
    >>> print(np.round(sp_p_value, 4))
    0.8486
    """
    _check_arrays_length(y_true, y_score)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_score)
    _check_array_inf(y_score)
    sp_stat = spiegelhalter_statistic(y_true, y_score)
    sp_p_value = 1 - scipy.stats.norm.cdf(sp_stat)
    return sp_p_value
