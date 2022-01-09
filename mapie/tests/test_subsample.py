from __future__ import annotations

import numpy as np

from mapie.subsample import Subsample


def test_default_parameters() -> None:
    """Test default values of Subsample."""
    cv = Subsample()
    assert cv.n_resamplings == 30
    assert cv.n_samples is None
    assert cv.random_state is None


def test_get_n_splits() -> None:
    """Test get_n_splits method of Subsample."""
    cv = Subsample(n_resamplings=3)
    assert cv.get_n_splits() == 3


def test_split() -> None:
    """Test outputs of subsamplings."""
    X = np.array([0, 1, 2, 3])
    cv = Subsample(n_resamplings=2, random_state=1)
    trains = np.concatenate([x[0] for x in cv.split(X)])
    tests = np.concatenate([x[1] for x in cv.split(X)])
    trains_expected = np.array([1, 3, 0, 0, 3, 1, 3, 1])
    tests_expected = np.array([2, 0, 2])
    np.testing.assert_equal(trains, trains_expected)
    np.testing.assert_equal(tests, tests_expected)
