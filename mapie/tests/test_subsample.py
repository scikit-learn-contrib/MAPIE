from __future__ import annotations

from itertools import combinations, product
from typing import Union

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


@pytest.mark.parametrize("n_samples", [4, 6, 8, 10])
@pytest.mark.parametrize("n_resamplings", [1, 2, 3])
def test_n_samples_int(n_samples: int,
                       n_resamplings: int) -> None:
    """Test outputs of subsamplings when n_samples is a int"""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = Subsample(n_resamplings=n_resamplings, random_state=0,
                   n_samples=n_samples, replace=False)
    train_set = np.concatenate([x[0] for x in cv.split(X)])
    val_set = np.concatenate([x[1] for x in cv.split(X)])
    assert len(train_set) == n_samples*n_resamplings
    assert len(val_set) == (X.shape[0] - n_samples)*n_resamplings


@pytest.mark.parametrize("n_samples", [0.4, 0.6, 0.8, 0.9])
@pytest.mark.parametrize("n_resamplings", [1, 2, 3])
def test_n_samples_float(n_samples: float,
                         n_resamplings: int) -> None:
    """Test outputs of subsamplings when n_samples is a
    float between 0 and 1."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = Subsample(n_resamplings=n_resamplings, random_state=0,
                   n_samples=n_samples, replace=False)
    train_set = np.concatenate([x[0] for x in cv.split(X)])
    val_set = np.concatenate([x[1] for x in cv.split(X)])
    assert len(train_set) == int(np.floor(n_samples*X.shape[0]))*n_resamplings
    assert len(val_set) == (
        (X.shape[0] - int(np.floor(n_samples * X.shape[0]))) *
        n_resamplings
    )


@pytest.mark.parametrize("n_resamplings", [1, 2, 3])
def test_n_samples_none(n_resamplings: int) -> None:
    """Test outputs of subsamplings when n_samples is None."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = Subsample(n_resamplings=n_resamplings, random_state=0,
                   replace=False)
    train_set = np.concatenate([x[0] for x in cv.split(X)])
    val_set = np.concatenate([x[1] for x in cv.split(X)])
    assert len(train_set) == X.shape[0]*n_resamplings
    assert len(val_set) == 0


@pytest.mark.parametrize("n_samples", [0.4, 0.6, 3, 6])
@pytest.mark.parametrize("n_resamplings", [2, 3, 4])
def test_split_samples_Subsample(n_resamplings: int,
                                 n_samples: Union[int, float]) -> None:
    """Test that outputs of subsamplings are all different."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv = Subsample(n_resamplings=n_resamplings,
                   n_samples=n_samples, replace=False, random_state=0)
    trains = [x[0] for x in cv.split(X)]
    tests = [x[1] for x in cv.split(X)]
    for (train1, train2), (test1, test2) in product(
            combinations(trains, 2), combinations(tests, 2)):
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_equal(train1, train2)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_equal(test1, test2)


@pytest.mark.parametrize("n_samples", [0.4, 0.6, 3, 6])
@pytest.mark.parametrize("n_resamplings", [2, 3, 4])
def test_reproductibility_samples_Subsample(
        n_resamplings: int,
        n_samples: Union[int, float]
) -> None:
    """This test ensures that each split between
    two instances is the same for a given seed."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    cv1 = Subsample(n_resamplings=n_resamplings,
                    n_samples=n_samples, replace=False, random_state=0)
    trains1 = [x[0] for x in cv1.split(X)]
    tests1 = [x[1] for x in cv1.split(X)]
    cv2 = Subsample(n_resamplings=n_resamplings,
                    n_samples=n_samples, replace=False, random_state=0)
    trains2 = [x[0] for x in cv2.split(X)]
    tests2 = [x[1] for x in cv2.split(X)]
    np.testing.assert_array_equal(trains1, trains2)
    np.testing.assert_array_equal(tests1, tests2)


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


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("n_resamplings", [2, 3, 4])
def test_split_samples_BlockBootstrap(n_resamplings: int,
                                      length: int) -> None:
    """Test that outputs of subsamplings are all different."""
    X = np.arange(31)
    cv = BlockBootstrap(n_resamplings=n_resamplings,
                        length=length, random_state=0)
    trains = [x[0] for x in cv.split(X)]
    tests = [x[1] for x in cv.split(X)]
    for (train1, train2), (test1, test2) in product(
            combinations(trains, 2), combinations(tests, 2)):
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_equal(train1, train2)
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_equal(test1, test2)


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("n_resamplings", [2, 3, 4])
def test_reproductibility_samples_BlockBootstrap(
        n_resamplings: int,
        length: int) -> None:
    """This test ensures that each split between
    two instances is the same for a given seed."""
    X = np.arange(15)
    cv1 = BlockBootstrap(
        n_resamplings=n_resamplings,
        length=length,
        random_state=42
    )
    trains1 = [x[0] for x in list(cv1.split(X))]
    tests1 = [x[1] for x in list(cv1.split(X))]
    cv2 = BlockBootstrap(
        n_resamplings=n_resamplings,
        length=length,
        random_state=42
    )
    trains2 = [x[0] for x in list(cv2.split(X))]
    tests2 = [x[1] for x in list(cv2.split(X))]
    np.testing.assert_equal(trains1, trains2)
    np.testing.assert_equal(tests1, tests2)
