from __future__ import annotations

import pytest
from mapie.subsample import Subsample


def test_invalid_randomstates() -> None:
    """
    Test that wrong list of random states in Subsample
    class raise errors.
    """

    with pytest.raises(
        ValueError, match=r".*Incoherent number of random states*"
    ):
        Subsample(
            n_resamplings=30, replace=True, random_states=[0, 1],
        )


def test_default_Subsample() -> None:
    """Test default values of Subsample."""
    cv = Subsample(n_resamplings=30)
    assert cv.n_resamplings == 30
    assert cv.n_samples is None
    assert cv.random_states is None


def test_get_n_splits() -> None:
    """Test get_n_splits method of Subsample."""
    cv = Subsample(n_resamplings=30)
    assert cv.get_n_splits() == 30
