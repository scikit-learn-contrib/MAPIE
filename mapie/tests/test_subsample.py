from __future__ import annotations

from mapie.subsample import Subsample


def test_default_Subsample() -> None:
    """Test default values of Subsample."""
    cv = Subsample(n_resamplings=30)
    assert cv.n_resamplings == 30
    assert cv.n_samples is None
    assert cv.random_state is None


def test_get_n_splits() -> None:
    """Test get_n_splits method of Subsample."""
    cv = Subsample(n_resamplings=30)
    assert cv.get_n_splits() == 30
