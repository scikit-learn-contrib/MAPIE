import numpy as np
from mapie.aggregation_functions import phi1D


def test_phi1D_normal() -> None:
    x = np.array([1, 2, 3, 4, 5])
    B = np.array([[1, 1, 1, np.nan, np.nan], [np.nan, np.nan, np.nan, 1, 1]])
    res = phi1D(x, B, fun=lambda x: np.nanmean(x, axis=1))
    assert res[0] == 2.0
