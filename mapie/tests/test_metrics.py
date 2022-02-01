"""
Testing for metrics module.
"""
import pytest
import numpy as np
from mapie.metrics import (
    classification_coverage_score,
    classification_mean_width_score,
    regression_coverage_score,
    regression_mean_width_score,
)
from mapie._typing import ArrayLike


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


def test_regression_ypredlow_shape() -> None:
    "Test shape of y_pred_low."
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_coverage_score(y_toy, y_preds[:, :2], y_preds[:, 2])
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_mean_width_score(y_preds[:, :2], y_preds[:, 2])


def test_regression_ypredup_shape() -> None:
    "Test shape of y_pred_up."
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_coverage_score(y_toy, y_preds[:, 1], y_preds[:, 1:])
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_mean_width_score(y_preds[:, :2], y_preds[:, 2])


def test_regression_same_length() -> None:
    "Test when y_true and y_preds have different lengths."
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        regression_coverage_score(y_toy, y_preds[:-1, 1], y_preds[:-1, 2])
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_mean_width_score(y_preds[:, :2], y_preds[:, 2])


def test_regression_toydata_coverage_score() -> None:
    "Test coverage_score for toy data."
    scr = regression_coverage_score(y_toy, y_preds[:, 1], y_preds[:, 2])
    assert scr == 0.8


def test_regression_ytrue_type_coverage_score() -> None:
    "Test that list(y_true) gives right coverage."
    scr = regression_coverage_score(list(y_toy), y_preds[:, 1], y_preds[:, 2])
    assert scr == 0.8


def test_regression_ypredlow_type_coverage_score() -> None:
    "Test that list(y_pred_low) gives right coverage."
    scr = regression_coverage_score(y_toy, list(y_preds[:, 1]), y_preds[:, 2])
    assert scr == 0.8


def test_regression_ypredup_type_coverage_score() -> None:
    "Test that list(y_pred_up) gives right coverage."
    scr = regression_coverage_score(y_toy, y_preds[:, 1], list(y_preds[:, 2]))
    assert scr == 0.8


def test_classification_y_true_shape() -> None:
    "Test shape of y_true."
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        classification_coverage_score(
            np.tile(y_true_class, (2, 1)), y_pred_set
        )


def test_classification_y_pred_set_shape() -> None:
    "Test shape of y_pred_set."
    with pytest.raises(ValueError, match=r".*Expected 2D array*"):
        classification_coverage_score(y_true_class, y_pred_set[:, 0])


def test_classification_same_length() -> None:
    "Test when y_true and y_pred_set have different lengths."
    with pytest.raises(IndexError, match=r".*shape mismatch*"):
        classification_coverage_score(y_true_class, y_pred_set[:-1, :])


def test_classification_toydata() -> None:
    "Test coverage_score for toy data."
    assert classification_coverage_score(y_true_class, y_pred_set) == 0.8


def test_classification_ytrue_type() -> None:
    "Test that list(y_true_class) gives right coverage."
    scr = classification_coverage_score(list(y_true_class), y_pred_set)
    assert scr == 0.8


def test_classification_y_pred_set_type() -> None:
    "Test that list(y_pred_set) gives right coverage."
    scr = classification_coverage_score(y_true_class, list(y_pred_set))
    assert scr == 0.8


@pytest.mark.parametrize("pred_set", [y_pred_set, list(y_pred_set)])
def test_classification_toydata_width(pred_set: ArrayLike) -> None:
    "Test width mean for toy data."
    assert classification_mean_width_score(pred_set) == 2.0


def test_classification_y_pred_set_width_shape() -> None:
    "Test shape of y_pred_set in classification_mean_width_score."
    with pytest.raises(ValueError, match=r".*Expected 2D array*"):
        classification_mean_width_score(y_pred_set[:, 0])


def test_regression_toydata_mean_width_score() -> None:
    "Test mean_width_score for toy data."
    scr = regression_mean_width_score(y_preds[:, 1], y_preds[:, 2])
    assert scr == 2.3


def test_regression_ypredlow_type_mean_width_score() -> None:
    "Test that list(y_pred_low) gives right coverage."
    scr = regression_mean_width_score(list(y_preds[:, 1]), y_preds[:, 2])
    assert scr == 2.3


def test_regression_ypredup_type_mean_width_score() -> None:
    "Test that list(y_pred_up) gives right coverage."
    scr = regression_mean_width_score(y_preds[:, 1], list(y_preds[:, 2]))
    assert scr == 2.3
