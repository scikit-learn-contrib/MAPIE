"""
Testing for metrics module.
"""
import numpy as np
import pytest
from numpy.random import RandomState

from typing_extensions import TypedDict
from mapie._typing import ArrayLike
from mapie.metrics import (classification_coverage_score,
                           classification_mean_width_score,
                           expected_calibration_error,
                           regression_coverage_score,
                           regression_mean_width_score, top_label_ece,
                           regression_ssc,
                           regression_ssc_score,
                           classification_ssc,
                           classification_ssc_score,
                           hsic)

y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5])
y_preds = np.array(
    [
        [5, 4, 6],
        [7.5, 6.0, 9.0],
        [9.5, 9, 10.0],
        [10.5, 8.5, 12.5],
        [11.5, 10.5, 12.0],
    ]
)
intervals = np.array(
    [
        [
            [4, 4], [6, 7.5]
        ],
        [
            [6.0, 8], [9.0, 10]
        ],
        [
            [9, 9], [10.0, 10.0]
        ],
        [
            [8.5, 9], [12.5, 12]
        ],
        [
            [10.5, 10.5], [12.0, 12]
        ]
    ]
)

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
        [False, True],
        [True, True],
        [True, True],
        [True, True]
    ],
    [
        [False, False],
        [True, False],
        [False, False],
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
        "intervals": ArrayLike,
        "n_splits": int,
        "method": str
    },
)
Params_ssc_classif = TypedDict(
    "Params_ssc_classif",
    {
        "y_pred_set": ArrayLike,
        "n_splits": int,
        "method": str
    },
)
SSC_REG = {
    "1alpha_base": Params_ssc_reg(
        intervals=intervals[:, :, 0],
        n_splits=2,
        method="uniform"
    ),
    "1alpha_3sp": Params_ssc_reg(
        intervals=intervals[:, :, 0],
        n_splits=3,
        method="uniform"
    ),
    "1alpha_3sp_quant": Params_ssc_reg(
        intervals=intervals[:, :, 0],
        n_splits=3,
        method="quantile"
    ),
    "1alpha_quant": Params_ssc_reg(
        intervals=intervals[:, :, 0],
        n_splits=2,
        method="quantile"
    ),
    "2alpha_base": Params_ssc_reg(
        intervals=intervals,
        n_splits=2,
        method="uniform"
    ),
    "2alpha_3sp": Params_ssc_reg(
        intervals=intervals,
        n_splits=3,
        method="uniform"
    ),
    "2alpha_3sp_quant": Params_ssc_reg(
        intervals=intervals,
        n_splits=3,
        method="quantile"
    ),
    "2alpha_quant": Params_ssc_reg(
        intervals=intervals,
        n_splits=2,
        method="quantile"
    ),
}
SSC_CLASSIF = {
    "1alpha_base": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas[:, :, 0],
        n_splits=2,
        method="uniform"
    ),
    "1alpha_3sp": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas[:, :, 0],
        n_splits=3,
        method="uniform"
    ),
    "1alpha_3sp_quant": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas[:, :, 0],
        n_splits=3,
        method="quantile"
    ),
    "1alpha_quant": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas[:, :, 0],
        n_splits=2,
        method="quantile"
    ),
    "2alpha_base": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas,
        n_splits=2,
        method="uniform"
    ),
    "2alpha_3sp": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas,
        n_splits=3,
        method="uniform"
    ),
    "2alpha_3sp_quant": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas,
        n_splits=3,
        method="quantile"
    ),
    "2alpha_quant": Params_ssc_classif(
        y_pred_set=y_pred_set_2alphas,
        n_splits=2,
        method="quantile"
    ),
}
SSC_REG_COVERAGES = {
    "1alpha_base": np.array([[2/3, 1.]]),
    "1alpha_3sp": np.array([[0.5, 1., 1.]]),
    "1alpha_3sp_quant": np.array([[0.5, 1., 1.]]),
    "1alpha_quant": np.array([[2/3, 1.]]),
    "2alpha_base": np.array([[2/3, 1.], [1/3, 1.]]),
    "2alpha_3sp": np.array([[0.5, 1., 1.], [0.5, 0.5, 1.]]),
    "2alpha_3sp_quant": np.array([[0.5, 1., 1.], [0.5, 0., 1.]]),
    "2alpha_quant": np.array([[2/3, 1.], [1/3, 1.]])
}
SSC_REG_COVERAGES_SCORE = {
    "1alpha_base": np.array([2/3]),
    "1alpha_3sp": np.array([0.5]),
    "1alpha_3sp_quant": np.array([0.5]),
    "1alpha_quant": np.array([2/3]),
    "2alpha_base": np.array([2/3, 1/3]),
    "2alpha_3sp": np.array([0.5, 0.5]),
    "2alpha_3sp_quant": np.array([0.5, 0.]),
    "2alpha_quant": np.array([2/3, 1/3])
}
SSC_CLASSIF_COVERAGES = {
    "1alpha_base": np.array([[1/3, 1.]]),
    "1alpha_3sp": np.array([[0.5, 0.5, 1.]]),
    "1alpha_3sp_quant": np.array([[0.5, 0., 1.]]),
    "1alpha_quant": np.array([[1/3, 1.]]),
    "2alpha_base": np.array([[1/3, 1.], [1/3, 1.]]),
    "2alpha_3sp": np.array([[0.5, 0.5, 1.], [0.5, 0.5, 1.]]),
    "2alpha_3sp_quant": np.array([[0.5, 0., 1.], [0.5, 0., 1.]]),
    "2alpha_quant": np.array([[1/3, 1.], [1/3, 1.]]),
}
SSC_CLASSIF_COVERAGES_SCORE = {
    "1alpha_base": np.array([1 / 3]),
    "1alpha_3sp": np.array([0.5]),
    "1alpha_3sp_quant": np.array([0.]),
    "1alpha_quant": np.array([1 / 3]),
    "2alpha_base": np.array([1 / 3, 1 / 3]),
    "2alpha_3sp": np.array([0.5, 0.5]),
    "2alpha_3sp_quant": np.array([0., 0.]),
    "2alpha_quant": np.array([1 / 3, 1 / 3]),
}

prng = RandomState(1234567890)
y_score = prng.random(51)
y_scores = prng.random((51, 5))
y_true = prng.randint(0, 2, 51)


def test_regression_ypredlow_shape() -> None:
    """Test shape of y_pred_low."""
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_coverage_score(y_toy, y_preds[:, :2], y_preds[:, 2])
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_mean_width_score(y_preds[:, :2], y_preds[:, 2])


def test_regression_ypredup_shape() -> None:
    """Test shape of y_pred_up."""
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_coverage_score(y_toy, y_preds[:, 1], y_preds[:, 1:])
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_mean_width_score(y_preds[:, :2], y_preds[:, 2])


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
    with pytest.raises(ValueError, match=r".*should be a 1d array*"):
        regression_ssc(np.tile(y_toy, 2).reshape(5, 2), y_preds)
    with pytest.raises(ValueError, match=r".*should be a 1d array*"):
        regression_ssc_score(np.tile(y_toy, 2).reshape(5, 2), y_preds)
    with pytest.raises(ValueError, match=r".*should be a 1d array*"):
        hsic(np.tile(y_toy, 2).reshape(5, 2), y_preds)


def test_regression_valid_input_shape() -> None:
    """Test valid shape of intervals raises no error"""
    regression_ssc(y_toy, intervals)
    regression_ssc_score(y_toy, intervals)
    hsic(y_toy, intervals)


def test_regression_same_length() -> None:
    """Test when y_true and y_preds have different lengths."""
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        regression_coverage_score(y_toy, y_preds[:-1, 1], y_preds[:-1, 2])
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_mean_width_score(y_preds[:, :2], y_preds[:, 2])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        regression_ssc(y_toy, intervals[:-1, ])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        regression_ssc_score(y_toy, intervals[:-1, ])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        hsic(y_toy, intervals[:-1, ])


def test_regression_toydata_coverage_score() -> None:
    """Test coverage_score for toy data."""
    scr = regression_coverage_score(y_toy, y_preds[:, 1], y_preds[:, 2])
    assert scr == 0.8


def test_regression_ytrue_type_coverage_score() -> None:
    """Test that list(y_true) gives right coverage."""
    scr = regression_coverage_score(list(y_toy), y_preds[:, 1], y_preds[:, 2])
    assert scr == 0.8


def test_regression_ypredlow_type_coverage_score() -> None:
    """Test that list(y_pred_low) gives right coverage."""
    scr = regression_coverage_score(y_toy, list(y_preds[:, 1]), y_preds[:, 2])
    assert scr == 0.8


def test_regression_ypredup_type_coverage_score() -> None:
    """Test that list(y_pred_up) gives right coverage."""
    scr = regression_coverage_score(y_toy, y_preds[:, 1], list(y_preds[:, 2]))
    assert scr == 0.8


def test_classification_y_true_shape() -> None:
    """Test shape of y_true."""
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        classification_coverage_score(
            np.tile(y_true_class, (2, 1)), y_pred_set
        )
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        classification_ssc(np.tile(y_true_class, (2, 1)), y_pred_set_2alphas)
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        classification_ssc_score(np.tile(y_true_class, (2, 1)),
                                 y_pred_set_2alphas)


def test_classification_y_pred_set_shape() -> None:
    """Test shape of y_pred_set."""
    with pytest.raises(ValueError, match=r".*Expected 2D array*"):
        classification_coverage_score(y_true_class, y_pred_set[:, 0])
    with pytest.raises(ValueError, match=r".*should be a 3D array*"):
        classification_ssc(y_true_class, y_pred_set[:, 0])
    with pytest.raises(ValueError, match=r".*should be a 3D array*"):
        classification_ssc_score(y_true_class, y_pred_set[:, 0])


def test_classification_same_length() -> None:
    """Test when y_true and y_pred_set have different lengths."""
    with pytest.raises(IndexError, match=r".*shape mismatch*"):
        classification_coverage_score(y_true_class, y_pred_set[:-1, :])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        classification_ssc(y_true_class, y_pred_set_2alphas[:-1, :, :])
    with pytest.raises(ValueError, match=r".*shape mismatch*"):
        classification_ssc_score(y_true_class, y_pred_set_2alphas[:-1, :, :])


def test_classification_valid_input_shape() -> None:
    """Test that valid inputs shape raise no error."""
    classification_ssc(y_true_class, y_pred_set_2alphas)
    classification_ssc_score(y_true_class, y_pred_set_2alphas)


def test_classification_toydata() -> None:
    """Test coverage_score for toy data."""
    assert classification_coverage_score(y_true_class, y_pred_set) == 0.8


def test_classification_ytrue_type() -> None:
    """Test that list(y_true_class) gives right coverage."""
    scr = classification_coverage_score(list(y_true_class), y_pred_set)
    assert scr == 0.8


def test_classification_y_pred_set_type() -> None:
    """Test that list(y_pred_set) gives right coverage."""
    scr = classification_coverage_score(y_true_class, list(y_pred_set))
    assert scr == 0.8


@pytest.mark.parametrize("pred_set", [y_pred_set, list(y_pred_set)])
def test_classification_toydata_width(pred_set: ArrayLike) -> None:
    """Test width mean for toy data."""
    assert classification_mean_width_score(pred_set) == 2.0


def test_classification_y_pred_set_width_shape() -> None:
    """Test shape of y_pred_set in classification_mean_width_score."""
    with pytest.raises(ValueError, match=r".*Expected 2D array*"):
        classification_mean_width_score(y_pred_set[:, 0])


def test_regression_toydata_mean_width_score() -> None:
    """Test mean_width_score for toy data."""
    scr = regression_mean_width_score(y_preds[:, 1], y_preds[:, 2])
    assert scr == 2.3


def test_regression_ypredlow_type_mean_width_score() -> None:
    """Test that list(y_pred_low) gives right coverage."""
    scr = regression_mean_width_score(list(y_preds[:, 1]), y_preds[:, 2])
    assert scr == 2.3


def test_regression_ypredup_type_mean_width_score() -> None:
    """Test that list(y_pred_up) gives right coverage."""
    scr = regression_mean_width_score(y_preds[:, 1], list(y_preds[:, 2]))
    assert scr == 2.3


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


def test_top_lable_ece() -> None:
    """Test that score is """
    scr = top_label_ece(y_true, y_scores)
    assert np.round(scr, 4) == 0.6997


def test_top_label_same_result() -> None:
    """
    Test that we have the same results if the input contais
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


@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_valid_method_regression_ssc(method: str) -> None:
    """Test that valid method for ssc splits raise no errors."""
    regression_ssc(y_toy, intervals, method=method)


@pytest.mark.parametrize("method", ["equal", 5, [6, 7]])
def test_invalid_method_regression_ssc(method: str) -> None:
    """Test that invalid method for ssc splits raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid method for ssc splits*"):
        regression_ssc(y_toy, intervals, method=method)


@pytest.mark.parametrize("n_splits", [10, 0, 1])
def test_invalid_splits_regression_ssc(n_splits: int) -> None:
    """Test that invalid number of splits for ssc raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid number of splits*"):
        regression_ssc(y_toy, intervals, n_splits=n_splits)


@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_valid_method_regression_ssc_score(method: str) -> None:
    """Test that valid method for ssc splits raise no errors."""
    regression_ssc_score(y_toy, intervals, method=method)


@pytest.mark.parametrize("method", ["equal", 5, [6, 7]])
def test_invalid_method_regression_ssc_score(method: str) -> None:
    """Test that invalid method for ssc splits raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid method for ssc splits*"):
        regression_ssc_score(y_toy, intervals, method=method)


@pytest.mark.parametrize("n_splits", [10, 0, 1])
def test_invalid_splits_regression_ssc_score(n_splits: int) -> None:
    """Test that invalid number of splits for ssc raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid number of splits*"):
        regression_ssc_score(y_toy, intervals, n_splits=n_splits)


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


@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_valid_method_classification_ssc(method: str) -> None:
    """Test that valid method for ssc splits raise no errors."""
    classification_ssc(y_true_class, y_pred_set_2alphas, method=method)


@pytest.mark.parametrize("method", ["equal", 5, [6, 7]])
def test_invalid_method_classification_ssc(method: str) -> None:
    """Test that invalid method for ssc splits raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid method for ssc splits*"):
        classification_ssc(y_true_class, y_pred_set_2alphas, method=method)


@pytest.mark.parametrize("n_splits", [10, 0, 1])
def test_invalid_splits_classification_ssc(n_splits: int) -> None:
    """Test that invalid number of splits for ssc raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid number of splits*"):
        classification_ssc(y_true_class, y_pred_set_2alphas, n_splits=n_splits)


@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_valid_method_classification_ssc_score(method: str) -> None:
    """Test that valid method for ssc splits raise no errors."""
    classification_ssc_score(y_true_class, y_pred_set_2alphas, method=method)


@pytest.mark.parametrize("method", ["equal", 5, [6, 7]])
def test_invalid_method_classification_ssc_score(method: str) -> None:
    """Test that invalid method for ssc splits raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid method for ssc splits*"):
        classification_ssc_score(
            y_true_class, y_pred_set_2alphas, method=method
        )


@pytest.mark.parametrize("n_splits", [10, 0, 1])
def test_invalid_splits_classification_ssc_score(n_splits: int) -> None:
    """Test that invalid number of splits for ssc raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid number of splits*"):
        classification_ssc_score(
            y_true_class, y_pred_set_2alphas, n_splits=n_splits
        )


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_return_shape(params: str) -> None:
    """Test that the arrays returned by ssc metrics have the correct shape."""
    cond_cov = classification_ssc(y_true_class, **SSC_CLASSIF[params])
    assert cond_cov.shape == SSC_CLASSIF_COVERAGES[params].shape


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_score_return_shape(params: str) -> None:
    """Test that the arrays returned by ssc metrics have the correct shape."""
    cond_cov_min = classification_ssc_score(
        y_true_class, **SSC_CLASSIF[params]
    )
    assert cond_cov_min.shape == SSC_CLASSIF_COVERAGES_SCORE[params].shape


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_return_shape_n_classes(params: str) -> None:
    """
    Test that the arrays returned by ssc metrics have the correct shape,
    when splits=None, there is n_classes+1 groups.
    """
    cond_cov = classification_ssc(y_true_class, **SSC_CLASSIF[params])
    assert cond_cov.shape == SSC_CLASSIF_COVERAGES[params].shape


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_score_return_shape_n_classes(params: str) -> None:
    """
    Test that the arrays returned by ssc metrics have the correct shape,
    when splits=None, there is n_classes+1 groups.
    """
    cond_cov_min = classification_ssc_score(
        y_true_class, **SSC_CLASSIF[params]
    )
    assert cond_cov_min.shape == SSC_CLASSIF_COVERAGES_SCORE[params]


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_coverage_values(params: str):
    """Test that the conditional coverage values returned are correct."""
    cond_cov = classification_ssc(y_true_class, **SSC_CLASSIF[params])
    np.testing.assert_allclose(cond_cov, SSC_CLASSIF_COVERAGES[params])


@pytest.mark.parametrize("params", [*SSC_CLASSIF])
def test_classification_ssc_score_coverage_values(params: str):
    """Test that the conditional coverage values returned are correct."""
    cond_cov_min = classification_ssc_score(
        y_true_class, **SSC_CLASSIF[params]
    )
    np.testing.assert_allclose(
        cond_cov_min, SSC_CLASSIF_COVERAGES_SCORE[params]
    )


@pytest.mark.parametrize("kernel_sizes", [(1, 2, 1), [1], 2, [[1, 2]]])
def test_hsic_invalid_kernel_sizes(kernel_sizes: ArrayLike):
    with pytest.raises(TypeError):
        hsic(y_toy, intervals, kernel_sizes=kernel_sizes)


@pytest.mark.parametrize("kernel_sizes", [(1, 1), (2, 2), (3, 1)])
def test_hsic_valid_kernel_sizes(kernel_sizes: ArrayLike):
    hsic(y_toy, intervals, kernel_sizes=kernel_sizes)


@pytest.mark.parametrize("kernel_sizes", [(1, 1), (2, 2), (3, 1)])
def test_hsic_return_shape(kernel_sizes: ArrayLike):
    coef = hsic(y_toy, intervals, kernel_sizes=kernel_sizes)
    assert coef.shape == (2,)


def test_hsic_correlation_value():
    coef = hsic(y_toy, intervals, kernel_sizes=(1, 1))
    np.testing.assert_allclose(coef, np.array([0.16829506, 0.3052798]))
