"""
Testing for metrics module.
"""
from typing import Union

import numpy as np
import pytest
from numpy.random import RandomState
from typing_extensions import TypedDict

from numpy.typing import ArrayLike, NDArray
from mapie.metrics.calibration import (spiegelhalter_p_value)
from mapie.metrics.calibration import (
    add_jitter,
    cumulative_differences,
    expected_calibration_error,
    kolmogorov_smirnov_cdf,
    kolmogorov_smirnov_p_value,
    kolmogorov_smirnov_statistic,
    kuiper_cdf,
    kuiper_p_value,
    kuiper_statistic,
    length_scale,
    sort_xy_by_y,
    spiegelhalter_statistic,
    top_label_ece,
)
from mapie.metrics.classification import (
    classification_mean_width_score,
    classification_coverage_score,
    classification_ssc, classification_ssc_score,
)
from mapie.metrics.regression import (
    regression_mean_width_score,
    regression_coverage_score,
    regression_ssc,
    regression_ssc_score, hsic, coverage_width_based, regression_mwi_score,
)

y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5])
y_preds = np.array([
    [5, 4, 6],
    [7.5, 6.0, 9.0],
    [9.5, 9, 10.0],
    [10.5, 8.5, 12.5],
    [11.5, 10.5, 12.0],
])
intervals = np.array([
    [[4, 4], [6, 7.5]],
    [[6.0, 8], [9.0, 10]],
    [[9, 9], [10.0, 10.0]],
    [[8.5, 9], [12.5, 12]],
    [[10.5, 10.5], [12.0, 12]]
])

y_true_class = np.array([3, 3, 1, 2, 2])
y_pred_set = np.array(
    [
        [False, False, True, True],
        [False, True, False, True],
        [False, True, True, False],
        [False, False, True, True],
        [False, True, False, True],
    ]
)
y_pred_set_2alphas = np.array([
    [
        [False, False],
        [False, True],
        [False, True],
        [False, False],
    ],
    [
        [False, False],
        [True, True],
        [True, True],
        [True, True]
    ],
    [
        [False, False],
        [True, False],
        [True, False],
        [False, False]
    ],
    [
        [True, False],
        [True, False],
        [True, True],
        [True, False],
    ],
    [
        [False, False],
        [True, True],
        [False, True],
        [True, True]
    ]
])

Params_ssc_reg = TypedDict(
    "Params_ssc_reg",
    {
        "y_intervals": NDArray,
        "num_bins": int
    },
)
Params_ssc_classif = TypedDict(
    "Params_ssc_classif",
    {
        "y_pred_set": NDArray,
        "num_bins": Union[int, None]
    },
)
SSC_REG = {
    "1alpha_base": Params_ssc_reg(
        y_intervals=intervals[:, :, 0],
        num_bins=2
    ),
    "1alpha_3sp": Params_ssc_reg(
        y_intervals=intervals[:, :, 0],
        num_bins=3
    ),
    "2alpha_base": Params_ssc_reg(
        y_intervals=intervals,
        num_bins=2
    ),
    "2alpha_3sp": Params_ssc_reg(
        y_intervals=intervals,
        num_bins=3
    ),
}
SSC_CLASSIF = {
    "1alpha_base": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas[:, :, 0],
        num_bins=2
    ),
    "1alpha_3sp": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas[:, :, 0],
        num_bins=3
    ),
    "1alpha_None": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas[:, :, 0],
        num_bins=None
    ),
    "2alpha_base": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas,
        num_bins=2,
    ),
    "2alpha_3sp": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas,
        num_bins=3
    ),
    "2alpha_None": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas,
        num_bins=None
    ),
}
SSC_REG_COVERAGES = {
    "1alpha_base": np.array([[2/3, 1.]]),
    "1alpha_3sp": np.array([[0.5, 1., 1.]]),
    "2alpha_base": np.array([[2/3, 1.], [1/3, 1.]]),
    "2alpha_3sp": np.array([[0.5, 1., 1.], [0.5, 0.5, 1.]]),
}
SSC_REG_COVERAGES_SCORE = {
    "1alpha_base": np.array([2/3]),
    "1alpha_3sp": np.array([0.5]),
    "2alpha_base": np.array([2/3, 1/3]),
    "2alpha_3sp": np.array([0.5, 0.5]),
}
SSC_CLASSIF_COVERAGES = {
    "1alpha_base": np.array([[1/3, 1.]]),
    "1alpha_3sp": np.array([[0., 2/3, 1.]]),
    "1alpha_None": np.array([[0., np.nan, 0.5, 1., 1.]]),
    "2alpha_base": np.array([[1/3, 1.], [1/3, 1.]]),
    "2alpha_3sp": np.array([[0., 2/3, 1.], [0.5, 2/3, np.nan]]),
    "2alpha_None": np.array([[0., np.nan, 0.5, 1., 1.],
                             [0., 1., 0., 1., np.nan]]),
}
SSC_CLASSIF_COVERAGES_SCORE = {
    "1alpha_base": np.array([1 / 3]),
    "1alpha_3sp": np.array([0.]),
    "1alpha_None": np.array([0.]),
    "2alpha_base": np.array([1 / 3, 1 / 3]),
    "2alpha_3sp": np.array([0., 0.5]),
    "2alpha_None": np.array([0., 0.]),
}

prng = RandomState(1234567890)
y_score = prng.random(51)
y_scores = prng.random((51, 5))
y_true = prng.randint(0, 2, 51)


def test_regression_ypredlow_shape() -> None:
    """Test shape of y_pred_low."""
    with pytest.raises(ValueError):
        coverage_width_based(
            y_toy, y_preds[:1], y_preds[:, 2], eta=30, confidence_level=0.9
        )


def test_regression_ypredup_shape() -> None:
    """Test shape of y_pred_up."""
    with pytest.raises(ValueError):
        coverage_width_based(
            y_toy, y_preds[:, 1], y_preds[:1], eta=30, confidence_level=0.9
        )


def test_regression_intervals_invalid_shape() -> None:
    """Test invalid shape of intervals raises an error"""
    with pytest.raises(ValueError, match=r".*should be a 3D array*"):
        regression_ssc(y_toy, y_preds[0])
    with pytest.raises(ValueError, match=r".*should be a 3D array*"):
        regression_ssc_score(y_toy, y_preds[0])
    with pytest.raises(ValueError, match=r".*should be a 3D array*"):
        hsic(y_toy, y_preds[0])


def test_regression_ytrue_invalid_shape() -> None:
    """Test invalid shape of y_true raises an error"""
    with pytest.raises(ValueError):
        regression_ssc(np.tile(y_toy, 2).reshape(5, 2), y_preds)
    with pytest.raises(ValueError):
        regression_ssc_score(np.tile(y_toy, 2).reshape(5, 2), y_preds)
    with pytest.raises(ValueError):
        hsic(np.tile(y_toy, 2).reshape(5, 2), y_preds)
    with pytest.raises(ValueError):
        coverage_width_based(
            np.tile(y_toy, 2).reshape(5, 2), y_preds[:, 1], y_preds[:, 2],
            eta=30, confidence_level=0.9
        )


def test_regression_valid_input_shape() -> None:
    """Test valid shape of intervals raises no error"""
    regression_ssc(y_toy, intervals)
    regression_ssc_score(y_toy, intervals)
    hsic(y_toy, intervals)
    coverage_width_based(
        y_toy, y_preds[:, 1], y_preds[:, 2], eta=0, confidence_level=0.9
    )
    regression_mean_width_score(intervals)


def test_regression_same_length() -> None:
    """Test when y_true and y_preds have different lengths."""
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        regression_ssc(y_toy, intervals[:-1, ])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        regression_ssc_score(y_toy, intervals[:-1, ])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        hsic(y_toy, intervals[:-1, ])
    with pytest.raises(ValueError):
        coverage_width_based(
            y_toy, y_preds[:-1, 1], y_preds[:, 2], eta=0, confidence_level=0.9
        )


def test_regression_toydata_coverage_score() -> None:
    """Test coverage_score for toy data."""
    scr = regression_coverage_score(y_toy, y_preds[:, 1:])[0]
    assert scr == 0.8


def test_classification_y_true_shape() -> None:
    """Test shape of y_true."""
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        classification_ssc(np.tile(y_true_class, (2, 1)), y_pred_set_2alphas)
    with pytest.raises(ValueError, match=r".*are arrays with different len*"):
        classification_ssc_score(np.tile(y_true_class, (2, 1)),
                                 y_pred_set_2alphas)


def test_classification_y_pred_set_shape() -> None:
    """Test shape of y_pred_set."""
    with pytest.raises(ValueError, match=r".*should be a 3D array*"):
        classification_ssc(y_true_class, y_pred_set[:, 0])
    with pytest.raises(ValueError, match=r".*should be a 3D array*"):
        classification_ssc_score(y_true_class, y_pred_set[:, 0])


def test_classification_same_length() -> None:
    """Test when y_true and y_pred_set have different lengths."""
    with pytest.raises(ValueError, match=r".*are arrays with different len*"):
        classification_coverage_score(y_true_class, y_pred_set[:-1, :])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        classification_ssc(y_true_class, y_pred_set_2alphas[:-1, :, :])
    with pytest.raises(ValueError, match=r".*are arrays with different len*"):
        classification_ssc_score(y_true_class, y_pred_set_2alphas[:-1, :, :])


def test_classification_valid_input_shape() -> None:
    """Test that valid inputs shape raise no error."""
    classification_ssc(y_true_class, y_pred_set_2alphas)
    classification_ssc_score(y_true_class, y_pred_set_2alphas)
    classification_mean_width_score(y_pred_set_2alphas)


def test_classification_toydata() -> None:
    """Test coverage_score for toy data."""
    assert classification_coverage_score(y_true_class, y_pred_set)[0] == 0.8


def test_classification_mean_width_score_toydata() -> None:
    """Test classification_mean_width_score for toy data."""
    scr = classification_mean_width_score(y_pred_set_2alphas)
    np.testing.assert_allclose(scr, [2.2, 1.8], rtol=1e-2, atol=1e-2)


def test_regression_mean_width_score_toydata() -> None:
    """Test regression_mean_width_score for toy data."""
    scr = regression_mean_width_score(intervals)
    np.testing.assert_allclose(scr, [2.3, 2.2], rtol=1e-2, atol=1e-2)


def test_ece_score() -> None:
    """
    Test the expected calibration score for
    dataset if score is list of max scores.
    """
    scr = expected_calibration_error(y_true, y_score)
    assert np.round(scr, 4) == 0.4471


def test_ece_scores() -> None:
    """
    Test the expected calibration score for
    dataset if score probability output.
    """
    scr = expected_calibration_error(y_true, y_scores)
    assert np.round(scr, 4) == 0.5363


def test_top_label_ece() -> None:
    """Test that score is """
    scr = top_label_ece(y_true, y_scores)
    assert np.round(scr, 4) == 0.6997


def test_top_label_same_result() -> None:
    """
    Test that we have the same results if the input contains
    the maximum with the argmax values or if it is the probabilities
    """
    pred_proba_ = np.array(
        [
            [0.2, 0.2, 0.4],
            [0.5, 0.3, 0.2],
            [0, 0.4, 0.6],
            [0.1, 0.7, 0.2]
        ]
    )
    y_true_ = np.array([1, 0, 2, 1])
    pred_max_ = np.max(pred_proba_, axis=1)
    pred_argmax_ = np.argmax(pred_proba_, axis=1)

    scr1 = top_label_ece(y_true_, pred_proba_)
    scr2 = top_label_ece(
        y_true_,
        pred_max_,
        y_score_arg=pred_argmax_
    )

    classes = np.unique([y_true_+1])
    scr3 = top_label_ece(
        y_true_+1,
        pred_proba_,
        classes=classes,
    )

    scr4 = top_label_ece(
        y_true_+1,
        np.max(pred_proba_, axis=1),
        classes[np.argmax(pred_proba_, axis=1)]
    )
    assert scr1 == scr2
    assert scr1 == scr3
    assert scr1 == scr4


@pytest.mark.parametrize("num_bins", [10, 0, -1, 5])
def test_invalid_splits_regression_ssc(num_bins: int) -> None:
    """Test that invalid number of bins for ssc raise errors."""
    with pytest.raises(ValueError):
        regression_ssc(y_toy, intervals, num_bins=num_bins)


@pytest.mark.parametrize("num_bins", [10, 0, -1])
def test_invalid_splits_regression_ssc_score(num_bins: int) -> None:
    """Test that invalid number of bins for ssc raise errors."""
    with pytest.raises(ValueError):
        regression_ssc_score(y_toy, intervals, num_bins=num_bins)


@pytest.mark.parametrize("num_bins", [1, 2, 3])
def test_valid_splits_regression_ssc(num_bins: int) -> None:
    """Test that valid number of bins for ssc raise no error."""
    regression_ssc(y_toy, intervals, num_bins=num_bins)


@pytest.mark.parametrize("num_bins", [1, 2, 3])
def test_valid_splits_regression_ssc_score(num_bins: int) -> None:
    """Test that valid number of bins for ssc score raise no error."""
    regression_ssc_score(y_toy, intervals, num_bins=num_bins)


@pytest.mark.parametrize("params", [*SSC_REG])
def test_regression_ssc_return_shape(params: str) -> None:
    """Test that the array returned by ssc metric has the correct shape."""
    cond_cov = regression_ssc(y_toy, **SSC_REG[params])
    assert cond_cov.shape == SSC_REG_COVERAGES[params].shape


@pytest.mark.parametrize("params", [*SSC_REG])
def test_regression_ssc_score_return_shape(params: str) -> None:
    """Test that the array returned by ssc score has the correct shape."""
    cond_cov_min = regression_ssc_score(y_toy, **SSC_REG[params])
    assert cond_cov_min.shape == SSC_REG_COVERAGES_SCORE[params].shape


@pytest.mark.parametrize("params", [*SSC_REG])
def test_regression_ssc_coverage_values(params: str):
    """Test that the conditional coverage values returned are correct."""
    cond_cov = regression_ssc(y_toy, **SSC_REG[params])
    np.testing.assert_allclose(cond_cov, SSC_REG_COVERAGES[params])


@pytest.mark.parametrize("params", [*SSC_REG])
def test_regression_ssc_score_coverage_values(params: str):
    """Test that the conditional coverage values returned are correct."""
    cond_cov_min = regression_ssc_score(y_toy, **SSC_REG[params])
    np.testing.assert_allclose(cond_cov_min, SSC_REG_COVERAGES_SCORE[params])


@pytest.mark.parametrize("num_bins", [10, 0, -1, 4])
def test_invalid_splits_classification_ssc(num_bins: int) -> None:
    """Test that invalid number of bins for ssc raise errors."""
    with pytest.raises(ValueError):
        classification_ssc(y_true_class, y_pred_set_2alphas, num_bins=num_bins)


@pytest.mark.parametrize("num_bins", [10, 0, -1])
def test_invalid_splits_classification_ssc_score(num_bins: int) -> None:
    """Test that invalid number of bins for ssc raise errors."""
    with pytest.raises(ValueError):
        classification_ssc_score(y_true_class, y_pred_set_2alphas,
                                 num_bins=num_bins)


@pytest.mark.parametrize("num_bins", [3, 2, None])
def test_valid_splits_classification_ssc(num_bins: int) -> None:
    """Test that valid number of bins for ssc raise no error."""
    classification_ssc(y_true_class, y_pred_set_2alphas, num_bins=num_bins)


@pytest.mark.parametrize("num_bins", [3, 2, None])
def test_valid_splits_classification_ssc_score(num_bins: int) -> None:
    """Test that valid number of bins for ssc raise no error."""
    classification_ssc_score(y_true_class, y_pred_set_2alphas,
                             num_bins=num_bins)


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_return_shape(params: str) -> None:
    """Test that the arrays returned by ssc metrics have the correct shape."""
    cond_cov = classification_ssc(y_true_class, **SSC_CLASSIF[params])
    assert cond_cov.shape == SSC_CLASSIF_COVERAGES[params].shape


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_score_return_shape(params: str) -> None:
    """Test that the arrays returned by ssc metrics have the correct shape."""
    cond_cov_min = classification_ssc_score(y_true_class,
                                            **SSC_CLASSIF[params])
    assert cond_cov_min.shape == SSC_CLASSIF_COVERAGES_SCORE[params].shape


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_coverage_values(params: str) -> None:
    """Test that the conditional coverage values returned are correct."""
    cond_cov = classification_ssc(y_true_class, **SSC_CLASSIF[params])
    np.testing.assert_allclose(cond_cov, SSC_CLASSIF_COVERAGES[params])


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_score_coverage_values(params: str) -> None:
    """Test that the conditional coverage values returned are correct."""
    cond_cov_min = classification_ssc_score(y_true_class,
                                            **SSC_CLASSIF[params])
    np.testing.assert_allclose(
        cond_cov_min, SSC_CLASSIF_COVERAGES_SCORE[params]
    )


@pytest.mark.parametrize("kernel_sizes", [[1], 2, [[1, 2]], [-1, 1], [1, -1]])
def test_hsic_invalid_kernel_sizes(kernel_sizes: ArrayLike) -> None:
    """Test that invalid kernel sizes raises an error"""
    with pytest.raises(ValueError):
        hsic(y_toy, intervals, kernel_sizes=kernel_sizes)


@pytest.mark.parametrize("kernel_sizes", [(1, 1), [2, 2], (3, 1)])
def test_hsic_valid_kernel_sizes(kernel_sizes: ArrayLike) -> None:
    """Test that valid kernel sizes raises no errors"""
    hsic(y_toy, intervals, kernel_sizes=kernel_sizes)


@pytest.mark.parametrize("kernel_sizes", [(1, 1), (2, 2), (3, 1)])
def test_hsic_return_shape(kernel_sizes: ArrayLike) -> None:
    """Test that the arrau returned by hsic has the good shape"""
    coef = hsic(y_toy, intervals, kernel_sizes=kernel_sizes)
    assert coef.shape == (2,)


def test_hsic_correlation_value() -> None:
    """Test that the values returned by hsic are correct"""
    coef = hsic(y_toy, intervals, kernel_sizes=(1, 1))
    np.testing.assert_allclose(coef, np.array([0.16829506, 0.3052798]))


def test_regression_coverage_score_ytrue_valid_shape() -> None:
    """Test that no error is raised if y_true has the shape (n_samples,)."""
    regression_coverage_score(y_toy, intervals)


def test_regression_coverage_score_intervals_invalid_shape() -> None:
    """Test that an error is raised if intervals has not the good shape."""
    with pytest.raises(ValueError):
        regression_coverage_score(
            np.expand_dims(y_toy, 1), intervals[:, 0, 0]
        )


def test_classification_coverage_score_ytrue_valid_shape() -> None:
    """Test that no error is raised if y_true has a shape (n_samples,)."""
    classification_coverage_score(y_true_class, y_pred_set_2alphas)


def test_classification_coverage_score_ypredset_invalid_shape() -> None:
    """Test that an error is raised if y_pred_set has not the good shape."""
    with pytest.raises(ValueError):
        classification_coverage_score(
            np.expand_dims(y_true_class, axis=1), y_pred_set[:, 0]
        )


def test_alpha_invalid_cwc_score() -> None:
    """Test a non-valid value of mu in cwc score."""
    with pytest.raises(ValueError):
        coverage_width_based(
            y_preds[:, 0], y_preds[:, 1], y_preds[:, 2], eta=30, confidence_level=-1
        )


def test_valid_eta() -> None:
    """Test different values of eta in cwc metric."""
    y, y_low, y_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]
    cwb = coverage_width_based(y, y_low, y_up, eta=30, confidence_level=0.9)
    np.testing.assert_allclose(cwb, 0.48, rtol=1e-2)

    cwb = coverage_width_based(y, y_low, y_up, eta=0.01, confidence_level=0.9)
    np.testing.assert_allclose(cwb, 0.65, rtol=1e-2)

    cwb = coverage_width_based(y, y_low, y_up, eta=-1, confidence_level=0.9)
    np.testing.assert_allclose(cwb, 0.65, rtol=1e-2)

    cwb = coverage_width_based(y, y_low, y_up, eta=0, confidence_level=0.9)
    np.testing.assert_allclose(cwb, 0.65, rtol=1e-2)


@pytest.mark.parametrize("amplitude", [0.1, 0.01, 0.001])
def test_add_jitter_amplitude(amplitude: float) -> None:
    """Test that noise perturbation is consistent with required amplitude"""
    x = np.array([0, 1, 2, 3, 4])
    x_jittered = add_jitter(x, noise_amplitude=amplitude, random_state=1)
    np.testing.assert_allclose(x, x_jittered, rtol=5*amplitude)


def test_sort_xy_by_y() -> None:
    """
    Test that sorting two reversed arrays by one of them
    does reverse the arrays
    """
    x = np.linspace(-3, 3, 20)
    y = np.linspace(3, -3, 20)
    x_sorted, y_sorted = sort_xy_by_y(x, y)
    np.testing.assert_allclose(x_sorted, y)
    np.testing.assert_allclose(y_sorted, x)


@pytest.mark.parametrize("random_state", [1, 2, 3])
def test_cumulative_differences(random_state: int) -> None:
    """Test that cumulative differences are always between -1 and 1"""
    generator = RandomState(random_state)
    y_true = generator.choice([0, 1], size=100)
    y_score = generator.uniform(size=100)
    cum_diff = cumulative_differences(y_true, y_score)
    assert np.max(cum_diff) <= 1
    assert np.min(cum_diff) >= -1


def test_length_scale() -> None:
    """Test that length scale are well computed"""
    generator = RandomState(1)
    y_score = generator.uniform(size=100)
    scale = length_scale(y_score)
    np.testing.assert_allclose(scale, 0.040389, atol=1e-6)


def test_kolmogorov_smirnov_statistic() -> None:
    """Test that Kolmogorov-Smirnov's statistics are well computed"""
    generator = RandomState(1)
    y_true = generator.choice([0, 1], size=100)
    y_score = generator.uniform(size=100)
    ks_stat = kolmogorov_smirnov_statistic(y_true, y_score)
    np.testing.assert_allclose(ks_stat, 5.607741, atol=1e-6)


def test_kolmogorov_smirnov_cdf() -> None:
    """Test that Kolmogorov-Smirnov's statistics are well computed"""
    np.testing.assert_allclose(kolmogorov_smirnov_cdf(1), 0.370777, atol=1e-6)
    np.testing.assert_allclose(kolmogorov_smirnov_cdf(2), 0.908999, atol=1e-6)
    np.testing.assert_allclose(kolmogorov_smirnov_cdf(3), 0.9946, atol=1e-6)


def test_kolmogorov_smirnov_p_value_non_calibrated() -> None:
    """
    Test that Kolmogorov-Smirnov's p-values are well computed
    for uncalibrated data.
    """
    generator = RandomState(1)
    y_true = generator.choice([0, 1], size=100)
    y_score = generator.uniform(size=100)
    ks_stat = kolmogorov_smirnov_p_value(y_true, y_score)
    np.testing.assert_allclose(ks_stat, 0.0, atol=1e-6)


def test_kolmogorov_smirnov_p_value_calibrated() -> None:
    """
    Test that Kolmogorov-Smirnov's p-values are well computed
    for calibrated data.
    """
    generator = RandomState(1)
    y_score = generator.uniform(size=100)
    uniform = generator.uniform(size=len(y_score))
    y_true = (uniform <= y_score).astype(float)
    ks_stat = kolmogorov_smirnov_p_value(y_true, y_score)
    np.testing.assert_allclose(ks_stat, 0.2148, atol=1e-6)


def test_kuiper_statistic() -> None:
    """Test that Kuiper's statistics are well computed"""
    generator = RandomState(1)
    y_true = generator.choice([0, 1], size=100)
    y_score = generator.uniform(size=100)
    ku_stat = kuiper_statistic(y_true, y_score)
    np.testing.assert_allclose(ku_stat, 5.354395, atol=1e-6)


def test_kuiper_cdf() -> None:
    """Test that Kuiper's statistics are well computed"""
    np.testing.assert_allclose(kuiper_cdf(1), 0.063365, atol=1e-6)
    np.testing.assert_allclose(kuiper_cdf(2), 0.818506, atol=1e-6)
    np.testing.assert_allclose(kuiper_cdf(3), 0.989201, atol=1e-6)


def test_kuiper_p_value_non_calibrated() -> None:
    """
    Test that Kuiper's p-values are well computed
    for uncalibrated data
    """
    generator = RandomState(1)
    y_true = generator.choice([0, 1], size=100)
    y_score = generator.uniform(size=100)
    ks_stat = kuiper_p_value(y_true, y_score)
    np.testing.assert_allclose(ks_stat, 0.0, atol=1e-6)


def test_kuiper_p_value_calibrated() -> None:
    """
    Test that Kuiper's p-values are well computed
    for calibrated data.
    """
    generator = RandomState(1)
    y_score = generator.uniform(size=100)
    uniform = generator.uniform(size=len(y_score))
    y_true = (uniform <= y_score).astype(float)
    ks_stat = kuiper_p_value(y_true, y_score)
    np.testing.assert_allclose(ks_stat, 0.313006, atol=1e-6)


def test_spiegelhalter_statistic() -> None:
    """Test that Spiegelhalter's statistics are well computed"""
    generator = RandomState(1)
    y_true = generator.choice([0, 1], size=100)
    y_score = generator.uniform(size=100)
    sp_stat = spiegelhalter_statistic(y_true, y_score)
    np.testing.assert_allclose(sp_stat, 13.906833, atol=1e-6)


def test_spiegelhalter_p_value_non_calibrated() -> None:
    """
    Test that Spiegelhalter's p-values are well computed
    for uncalibrated data
    """
    generator = RandomState(1)
    y_true = generator.choice([0, 1], size=100)
    y_score = generator.uniform(size=100)
    ks_stat = spiegelhalter_p_value(y_true, y_score)
    np.testing.assert_allclose(ks_stat, 0.0, atol=1e-6)


def test_spiegelhalter_p_value_calibrated() -> None:
    """
    Test that Spiegelhalter's p-values are well computed
    for calibrated data.
    """
    generator = RandomState(1)
    y_score = generator.uniform(size=100)
    uniform = generator.uniform(size=len(y_score))
    y_true = (uniform <= y_score).astype(float)
    ks_stat = spiegelhalter_p_value(y_true, y_score)
    np.testing.assert_allclose(ks_stat, 0.174832, atol=1e-6)


def test_regression_mwi_score() -> None:
    """
    Test the mean Winkler interval score.
    There are four predictions in y_pis.
    For each the ground truth value is 10.0.
    The first interval covers the true value.
    The second interval is above the true value.
    The third interval has lower > upper, i.e.
    quantile crossing.
    The fourth interval is below the true value.
    """

    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pis = np.array([
        [[5.0],
            [15.0]],
        [[15.0],
            [25.0]],
        [[12.0],
            [8.0]],
        [[-5.0],
            [0.0]]])

    alpha = 0.1

    mwi_score = regression_mwi_score(y_true, y_pis, 1 - alpha)
    np.testing.assert_allclose(mwi_score, 82.25, rtol=1e-2)
