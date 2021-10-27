import numpy as np

from mapie.aggregation_functions import phi1D, phi2D


def test_phi1D() -> None:
    """Test the result of phi1D."""
    x = np.array([1, 2, 3, 4, 5])
    B = np.array([[1, 1, 1, np.nan, np.nan], [np.nan, np.nan, np.nan, 1, 1]])
    res = phi1D(x, B, fun=lambda x: np.nanmean(x, axis=1))
    assert res[0] == 2.0


def test_phi2D() -> None:
    """Test the result of phi2D."""
    A = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    B = np.array([[1, 1, 1, np.nan, np.nan], [np.nan, np.nan, np.nan, 1, 1]])
    res = phi2D(A, B, fun=lambda x: np.nanmean(x, axis=1))
    assert res[0, 0] == 2.0
    assert res[1, 0] == 7.0
