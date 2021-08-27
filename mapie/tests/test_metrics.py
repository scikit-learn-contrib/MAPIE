"""
Testing for metrics module.
"""
import pytest
import numpy as np
from mapie.metrics import regression_coverage_score
from mapie.metrics import classification_coverage_score


X_toy = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5])
y_preds = np.array([
    [5, 4, 6],
    [7.5, 6., 9.],
    [9.5, 9, 10.],
    [10.5, 8.5, 12.5],
    [11.5, 10.5, 12.]
])

y_true_class = np.array([3, 3, 1, 2, 2])
y_pred_set = np.array([
    [False, False, False,  True],
    [False, False, False,  True],
    [False,  True, False, False],
    [False, False,  True, False],
    [False,  True, False, False]
])


def test_ypredlow_shape() -> None:
    "Test shape of y_pred_low."
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_coverage_score(y_toy, y_preds[:, :2], y_preds[:, 2])


def test_ypredup_shape() -> None:
    "Test shape of y_pred_up."
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        regression_coverage_score(y_toy, y_preds[:, 1], y_preds[:, 1:])


def test_same_length() -> None:
    "Test when y_true and y_preds have different lengths."
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        regression_coverage_score(y_toy, y_preds[:-1, 1], y_preds[:-1, 2])


def test_toydata() -> None:
    "Test coverage_score for toy data"
    assert regression_coverage_score(
        y_toy, y_preds[:, 1], y_preds[:, 2]
    ) == 0.8


def test_ytrue_type() -> None:
    "Test that list(y_true) gives right coverage."
    assert regression_coverage_score(
        list(y_toy), y_preds[:, 1], y_preds[:, 2]
    ) == 0.8


def test_ypredlow_type() -> None:
    "Test that list(y_pred_low) gives right coverage."
    assert regression_coverage_score(
        y_toy, list(y_preds[:, 1]), y_preds[:, 2]
    ) == 0.8


def test_ypredup_type() -> None:
    "Test that list(y_pred_up) gives right coverage."
    assert regression_coverage_score(
        y_toy, y_preds[:, 1], list(y_preds[:, 2])
    ) == 0.8


def test_same_length_y_pred_set__y_true_class() -> None:
    "Test when y_true_class and y_pred_set have different lengths."
    with pytest.raises(IndexError, match=r".*index 4 is out of bounds*"):
        classification_coverage_score(y_true_class, y_pred_set[:-1, :])


def test_classification_coverage_score() -> None:
    "Test coverage_score for y_true_class"
    assert classification_coverage_score(y_true_class, y_pred_set) == 0.8
