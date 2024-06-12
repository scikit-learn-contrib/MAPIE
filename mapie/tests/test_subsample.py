from __future__ import annotations

import numpy as np
import pytest

from mapie.subsample import BlockBootstrap, Subsample


def test_default_parameters_SubSample() -> None:
    """Test default values of Subsample."""
    cv = Subsample()
    assert cv.n_resamplings == 30
    assert cv.n_samples is None
    assert cv.random_state is None


def test_get_n_splits_SubSample() -> None:
    """Test get_n_splits method of Subsample."""
    cv = Subsample(n_resamplings=3)
    assert cv.get_n_splits() == 3


def test_split_SubSample() -> None:
    """Test outputs of subsamplings."""
    X = np.array([0, 1, 2, 3])
    cv = Subsample(n_resamplings=2, random_state=1)
    trains = np.concatenate([x[0] for x in cv.split(X)])
    tests = np.concatenate([x[1] for x in cv.split(X)])
    trains_expected = np.array([1, 3, 0, 0, 3, 1, 3, 1])
    tests_expected = np.array([2, 0, 2])
    np.testing.assert_equal(trains, trains_expected)
    np.testing.assert_equal(tests, tests_expected)


def test_split_SubSample_n_samples() -> None:
    """Test outputs of subsamplings."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv1 = Subsample(n_resamplings=2, random_state=0,
                    n_samples=0.8, replace=False)
    cv2 = Subsample(n_resamplings=2, random_state=0,
                    n_samples=6, replace=False)
    train1 = np.concatenate([x[0] for x in cv1.split(X)])
    test1 = np.concatenate([x[1] for x in cv1.split(X)])
    train2 = np.concatenate([x[0] for x in cv2.split(X)])
    test2 = np.concatenate([x[1] for x in cv2.split(X)])
    train1_expected = np.array([2, 8, 4, 9, 1, 6, 7,
                                3, 3, 5, 1, 2, 9, 8, 0, 6])
    test1_expected = np.array([0, 5, 4, 7])
    train2_expected = np.array([2, 8, 4, 9, 1, 6, 3, 5, 1, 2, 9, 8])
    test2_expected = np.array([0, 3, 5, 7, 0, 4, 6, 7])
    expected_n_samples_cv1 = int(np.floor(0.8 * X.shape[0]))
    assert len(train1) == 2 * expected_n_samples_cv1
    assert len(train2) == 2 * 6
    np.testing.assert_equal(train1, train1_expected)
    np.testing.assert_equal(test1, test1_expected)
    np.testing.assert_equal(train2, train2_expected)
    np.testing.assert_equal(test2, test2_expected)


def test_default_parameters_BlockBootstrap() -> None:
    """Test default values of Subsample."""
    cv = BlockBootstrap()
    assert cv.n_resamplings == 30
    assert cv.length is None
    assert cv.n_blocks is None
    assert not cv.overlapping
    assert cv.random_state is None


def test_get_n_splits_BlockBootstrap() -> None:
    """Test get_n_splits method of Subsample."""
    cv = BlockBootstrap(n_resamplings=3)
    assert cv.get_n_splits() == 3


def test_split_BlockBootstrap() -> None:
    """Test outputs of subsamplings."""
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = BlockBootstrap(
        n_resamplings=1, length=2, overlapping=False, random_state=1
    )
    trains = np.concatenate([x[0] for x in cv.split(X)])
    tests = np.concatenate([x[1] for x in cv.split(X)])
    trains_expected = np.array([7, 8, 9, 10, 1, 2, 3, 4, 7, 8, 1, 2])
    tests_expected = np.array([5, 6])
    np.testing.assert_equal(trains, trains_expected)
    np.testing.assert_equal(tests, tests_expected)

    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = BlockBootstrap(
        n_resamplings=1, length=2, overlapping=True, random_state=1
    )
    trains = np.concatenate([x[0] for x in cv.split(X)])
    tests = np.concatenate([x[1] for x in cv.split(X)])
    trains_expected = np.array([5, 6, 8, 9, 9, 10, 5, 6, 0, 1, 0, 1])
    tests_expected = np.array([2, 3, 4, 7])
    np.testing.assert_equal(trains, trains_expected)
    np.testing.assert_equal(tests, tests_expected)


def test_split_BlockBootstrap_error_below_zero() -> None:
    """Test outputs of subsamplings for length block below 0."""
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = BlockBootstrap(length=20)
    with pytest.raises(ValueError, match=r".*The length of blocks is <= 0 *"):
        next(cv.split(X))


def test_split_BlockBootstrap_error() -> None:
    """Test outputs of subsamplings."""
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = BlockBootstrap()
    with pytest.raises(ValueError, match=r".*Exactly one argument*"):
        next(cv.split(X))
