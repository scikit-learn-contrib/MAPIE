"""
Testing for metrics module.
"""
import pytest
import numpy as np
from mapie.metrics import coverage_score


X_toy = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5])
y_preds = np.array([
    [5, 4, 6],
    [7.5, 6., 9.],
    [9.5, 9, 10.],
    [10.5, 8.5, 12.5],
    [11.5, 10.5, 12.]
])


def test_ypredlow_shape() -> None:
    "Test shape of y_pred_low."
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        coverage_score(y_toy, y_preds[:, :2], y_preds[:, 2])


def test_ypredup_shape() -> None:
    "Test shape of y_pred_low."
    with pytest.raises(ValueError, match=r".*y should be a 1d array*"):
        coverage_score(y_toy, y_preds[:, 1], y_preds[:, 1:])


def test_same_length() -> None:
    "Test when y_true and y_preds have different lengths."
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        coverage_score(y_toy, y_preds[:-1, 1], y_preds[:-1, 2])


def test_toydata() -> None:
    "Test coverage_score for toy data"
    assert coverage_score(y_toy, y_preds[:, 1], y_preds[:, 2]) == 0.8


def test_ytrue_type() -> None:
    "Test that list(y_true) gives right coverage."
    assert coverage_score(list(y_toy), y_preds[:, 1], y_preds[:, 2]) == 0.8


def test_ypredlow_type() -> None:
    "Test that list(y_pred_low) gives right coverage."
    assert coverage_score(y_toy, list(y_preds[:, 1]), y_preds[:, 2]) == 0.8


def test_ypredup_type() -> None:
    "Test that list(y_pred_up) gives right coverage."
    assert coverage_score(y_toy, y_preds[:, 1], list(y_preds[:, 2])) == 0.8
