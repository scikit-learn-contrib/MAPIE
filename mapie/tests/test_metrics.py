"""
Testing for metrics module.
"""
import pytest
import numpy as np
from mapie.metrics import coverage


X_toy = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5])
y_toy_preds = np.array([
    [5, 4, 6],
    [7.5, 6., 9.],
    [9.5, 9, 10.],
    [10.5, 8.5, 12.5],
    [11.5, 10.5, 12.]
])


def test_same_length() -> None:
    "Test when y_true and y_preds have different lengths."
    with pytest.raises(ValueError, match=r".*different lengths*"):
        coverage(y_toy, y_toy_preds[:-1, :])


def test_ypreds_shape() -> None:
    "Test that y_preds.shape[1] is equal to 3."
    with pytest.raises(ValueError, match=r".*not equal to 3*"):
        coverage(y_toy, y_toy_preds[:, :2])


def test_toydata() -> None:
    "Test coverage for toy data"
    assert (coverage(y_toy, y_toy_preds) == 0.8)


def test_ytrue_type() -> None:
    "Test that list(y_true) gives right coverage."
    assert (coverage(list(y_toy), y_toy_preds) == 0.8)


def test_ypreds_type() -> None:
    "Test that list(y_preds) gives right coverage."
    assert (coverage(y_toy, list(y_toy_preds)) == 0.8)
