from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import column_or_1d

from mapie.utils import (
    _check_arrays_length,
    _check_array_nan,
    _check_array_inf,
    _check_array_shape_regression,
    _check_number_bins,
    _check_nb_intervals_sizes,
    _check_alpha,
)


def regression_mean_width_score(
    y_intervals: NDArray
) -> NDArray:
    """
    Effective mean width score obtained by the prediction intervals.

    Parameters
    ----------
    y_intervals: NDArray of shape (n_samples, 2, n_confidence_level)
        Lower and upper bound of prediction intervals
        with different confidence levels, given by the ``predict_interval`` method

    Returns
    ---------
    NDArray of shape (n_confidence_level,)
        Effective mean width of the prediction intervals for each confidence level.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.metrics.regression import regression_mean_width_score
    >>> y_intervals = np.array([[[4, 6, 8], [6, 9, 11]],
    ...                    [[9, 10, 11], [10, 12, 14]],
    ...                    [[8.5, 9.5, 10], [12.5, 12, 13]],
    ...                    [[7, 8, 9], [8.5, 9.5, 10]],
    ...                    [[5, 6, 7], [6.5, 8, 9]]])
    >>> print(regression_mean_width_score(y_intervals))
    [2.  2.2 2.4]
    """
    y_intervals = np.asarray(y_intervals, dtype=float)

    _check_array_nan(y_intervals)
    _check_array_inf(y_intervals)

    width = np.abs(y_intervals[:, 1, :] - y_intervals[:, 0, :])
    mean_width = width.mean(axis=0)
    return mean_width


def regression_coverage_score(
    y_true: NDArray,
    y_intervals: NDArray,
) -> NDArray:
    """
    Effective coverage obtained by the prediction intervals.

    Intervals given by the ``predict_interval`` method can be passed directly
    to the ``y_intervals`` argument (see example below).

    Beside this intended use, this function also works with:

    - ``y_true`` of shape (n_sample,) and ``y_intervals`` of shape (n_sample, 2)
    - ``y_true`` of shape (n_sample, n) and `y_intervals` of shape
      (n_sample, 2, n)

    The effective coverage is obtained by computing the fraction
    of true labels that lie within the prediction intervals.

    Parameters
    ------------
    y_true: NDArray of shape (n_samples,)
        True labels.

    y_intervals: NDArray of shape (n_samples, 2, n_confidence_level)
        Lower and upper bound of prediction intervals
        with different confidence levels, given by the ``predict_interval`` method

    Returns
    ---------
    NDArray of shape (n_confidence_level,)
        Effective coverage obtained by the prediction intervals
        for each confidence level.

    Examples
    ---------
    >>> from mapie.metrics.regression import regression_coverage_score
    >>> from mapie.regression import SplitConformalRegressor
    >>> from mapie.utils import train_conformalize_test_split
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import Ridge

    >>> X, y = make_regression(n_samples=500, n_features=2, noise=1.0)
    >>> (
    ...     X_train, X_conformalize, X_test,
    ...     y_train, y_conformalize, y_test
    ... ) = train_conformalize_test_split(
    ...     X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
    ... )

    >>> mapie_regressor = SplitConformalRegressor(
    ...     estimator=Ridge(),
    ...     confidence_level=0.95,
    ...     prefit=False,
    ... ).fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)

    >>> predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)
    >>> coverage = regression_coverage_score(y_test, predicted_intervals)[0]
    """
    _check_arrays_length(y_true, y_intervals)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_intervals)
    _check_array_inf(y_intervals)

    y_intervals = _check_array_shape_regression(y_true, y_intervals)
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
    y_intervals: NDArray of shape (n_samples, 2, n_confidence_level) or (n_samples, 2)
        Prediction intervals given by booleans of labels.
    num_bins: int n
        Number of groups. Should be less than the number of different
        interval widths.

    Returns
    -------
    NDArray of shape (n_confidence_level, num_bins)

    Examples
    --------
    >>> from mapie.metrics.regression import regression_ssc
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
    y_intervals = _check_array_shape_regression(y_true, y_intervals)
    _check_number_bins(num_bins)
    widths = np.abs(y_intervals[:, 1, :] - y_intervals[:, 0, :])
    _check_nb_intervals_sizes(widths, num_bins)

    _check_arrays_length(y_true, y_intervals)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_intervals)
    _check_array_inf(y_intervals)

    indexes_sorted = np.argsort(widths, axis=0)
    indexes_bybins = np.array_split(indexes_sorted, num_bins, axis=0)
    coverages = np.zeros((y_intervals.shape[2], num_bins))
    for i, indexes in enumerate(indexes_bybins):
        intervals_binned = np.stack([
            np.take_along_axis(y_intervals[:, 0, :], indexes, axis=0),
            np.take_along_axis(y_intervals[:, 1, :], indexes, axis=0)
        ], axis=1)
        coverages[:, i] = regression_coverage_score(y_true[indexes], intervals_binned)

    return coverages


def regression_ssc_score(
    y_true: NDArray,
    y_intervals: NDArray,
    num_bins: int = 3
) -> NDArray:
    """
    Aggregate by the minimum for each confidence level the Size-Stratified Coverage [3]:
    returns the maximum violation of the conditional coverage
    (with the groups defined).

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
    y_intervals: NDArray of shape (n_samples, 2, n_confidence_level) or (n_samples, 2)
        Prediction intervals given by booleans of labels.
    num_bins: int n
        Number of groups. Should be less than the number of different
        interval widths.

    Returns
    -------
    NDArray of shape (n_confidence_level,)

    Examples
    --------
    >>> from mapie.metrics.regression import regression_ssc_score
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
    y_intervals: NDArray of shape (n_samples, 2, n_confidence_level) or (n_samples, 2)
        Prediction sets given by booleans of labels.
    kernel_sizes: ArrayLike of size (2,)
        The variance (sigma) for each variable (the indicator of coverage and
        the interval size), this coefficient controls the width of the curve.

    Returns
    -------
    NDArray of shape (n_confidence_level,)
        One hsic correlation coefficient by confidence level.

    Raises
    ------
    ValueError
        If kernel_sizes has a length different from 2
        and if it has negative or null values.

    Examples
    --------
    >>> from mapie.metrics.regression import hsic
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
    y_intervals = _check_array_shape_regression(y_true, y_intervals)

    _check_arrays_length(y_true, y_intervals)
    _check_array_nan(y_true)
    _check_array_inf(y_true)
    _check_array_nan(y_intervals)
    _check_array_inf(y_intervals)

    kernel_sizes = cast(NDArray, column_or_1d(kernel_sizes))
    if len(kernel_sizes) != 2:
        raise ValueError(
            "kernel_sizes should be an ArrayLike of length 2"
        )
    if (kernel_sizes <= 0).any():
        raise ValueError(
            "kernel_size should be positive"
        )
    n_samples, _, n_confidence_level = y_intervals.shape
    y_true_per_alpha = np.tile(y_true, (n_confidence_level, 1)).transpose()
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
    confidence_level: float
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
    confidence_level : float
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
    >>> from mapie.metrics.regression import coverage_width_based
    >>> import numpy as np
    >>> y_true = np.array([5, 7.5, 9.5, 10.5, 12.5])
    >>> y_preds_low = np.array([4, 6, 9, 8.5, 10.5])
    >>> y_preds_up = np.array([6, 9, 10, 12.5, 12])
    >>> eta = 0.01
    >>> confidence_level = 0.9
    >>> cwb = coverage_width_based(
    ... y_true, y_preds_low, y_preds_up, eta, confidence_level
    ... )
    >>> print(np.round(cwb ,2))
    0.69
    """
    y_true = cast(NDArray, column_or_1d(y_true))
    y_pred_low = cast(NDArray, column_or_1d(y_pred_low))
    y_pred_up = cast(NDArray, column_or_1d(y_pred_up))

    _check_alpha(confidence_level)

    coverage_score = regression_coverage_score(
        y_true,
        np.column_stack((y_pred_low, y_pred_up)),
    )[0]
    mean_width = regression_mean_width_score(
        np.column_stack((y_pred_low, y_pred_up))[:, :, np.newaxis]
    )[0]
    ref_length = np.subtract(
        float(y_true.max()),
        float(y_true.min())
    )
    avg_length = mean_width / ref_length

    cwc = (1-avg_length)*np.exp(-eta * (coverage_score - confidence_level) ** 2)

    return float(cwc)


def regression_mwi_score(
        y_true: NDArray,
        y_pis: NDArray,
        confidence_level: float
) -> float:
    """
    The Winkler score, proposed by Winkler (1972), is a measure used to
    evaluate prediction intervals, combining the length of the interval
    with a penalty that increases proportionally to the distance of an
    observation outside the interval.

    Parameters
    ----------
    y_true: ArrayLike of shape (n_samples,)
        Ground truth values
    y_pis: ArrayLike of shape (n_samples, 2, 1)
        Lower and upper bounds of prediction intervals
        output from a MAPIE regressor
    confidence_level: float
        The value of confidence_level

    Returns
    -------
    float
        The mean Winkler interval score

    References
    ----------
    [1] Robert L. Winkler
    "A Decision-Theoretic Approach to Interval Estimation",
    Journal of the American Statistical Association,
    volume 67, pages 187-191 (1972)
    (https://doi.org/10.1080/01621459.1972.10481224)
    [2] Tilmann Gneiting and Adrian E Raftery
    "Strictly Proper Scoring Rules, Prediction, and Estimation",
    Journal of the American Statistical Association,
    volume 102, pages 359-378 (2007)
    (https://doi.org/10.1198/016214506000001437) (Section 6.2)
    """

    # Undo any possible quantile crossing
    y_pred_low = np.minimum(y_pis[:, 0, 0], y_pis[:, 1, 0])
    y_pred_up = np.maximum(y_pis[:, 0, 0], y_pis[:, 1, 0])

    _check_arrays_length(y_true, y_pred_low, y_pred_up)

    # Checking for NaN and inf values
    for array in (y_true, y_pred_low, y_pred_up):
        _check_array_nan(array)
        _check_array_inf(array)

    width = np.sum(y_pred_up) - np.sum(y_pred_low)  # type: ignore
    error_above: float = np.sum((y_true - y_pred_up)[y_true > y_pred_up])
    error_below: float = np.sum((y_pred_low - y_true)[y_true < y_pred_low])
    total_error = error_above + error_below
    mwi = (width + total_error * 2 / (1 - confidence_level)) / len(y_true)
    return mwi
