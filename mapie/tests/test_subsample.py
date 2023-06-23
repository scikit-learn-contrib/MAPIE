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
      
def test_split_samples_are_different()->None:
    """Test that subsamples are different """
    X = np.array([0,1,2,3])
    cv = Subsample(n_resamplings=2, random_state=1)
    trains = [x[0] for x in cv.split(X)]
    tests = [x[1] for x in cv.split(X)]
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(trains[0], trains[1])
        np.testing.assert_equal(trains[1], trains[2])
        np.testing.assert_equal(trains[2], trains[3])

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(tests[0], tests[1])
        np.testing.assert_equal(tests[1], tests[2])
        np.testing.assert_equal(tests[2], tests[3])

def test_split_blockbootstraps_are_different()->None:
    """Test that BlockBootstrap outputs are different """
    X = np.array([0,1,2,3])
    cv = BlockBootstrap(n_blocks=2, n_resamplings=2, random_state=1)
    trains = [x[0] for x in cv.split(X)]
    tests = [x[1] for x in cv.split(X)]
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(trains[0], trains[1])
        np.testing.assert_equal(trains[1], trains[2])
        np.testing.assert_equal(trains[2], trains[3])

    with np.testing.assert_raises(AssertionError):
        np.testing.assert_equal(tests[0], tests[1])
        np.testing.assert_equal(tests[1], tests[2])
        np.testing.assert_equal(tests[2], tests[3])  