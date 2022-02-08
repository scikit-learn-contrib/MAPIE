import numpy as np
from sklearn.linear_model import LinearRegression

from mapie.time_series_regression import MapieTimeSeriesRegressor


def test_MapieTimeSeriesRegressor_partial_fit_ensemble_T() -> None:
    """Test ``residuals_`` when ``ensemble`` is True."""
    X_toy = np.array([[0], [1], [2], [3], [4], [5]])
    y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    mapie_ts_reg = MapieTimeSeriesRegressor(LinearRegression(), cv=-1).fit(
        X_toy, y_toy
    )
    assert round(mapie_ts_reg.residuals_[-1], 2) == round(np.abs(15 - 14.4), 2)
    mapie_ts_reg = mapie_ts_reg.partial_fit(
        X=np.array([[6]]), y=np.array([17.5]), ensemble=True
    )
    assert round(mapie_ts_reg.residuals_[-1], 2) == round(
        np.abs(17.5 - 16.56), 2
    )


def test_MapieTimeSeriesRegressor_partial_fit_ensemble_F() -> None:
    """Test ``residuals_`` when ``ensemble`` is False."""
    X_toy = np.array([[0], [1], [2], [3], [4], [5]])
    y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    mapie_ts_reg = MapieTimeSeriesRegressor(LinearRegression(), cv=-1).fit(
        X_toy, y_toy
    )
    assert round(mapie_ts_reg.residuals_[-1], 2) == round(np.abs(15 - 14.4), 2)
    mapie_ts_reg = mapie_ts_reg.partial_fit(
        X=np.array([[6]]), y=np.array([17.5]), ensemble=False
    )
    assert round(mapie_ts_reg.residuals_[-1], 2) == round(
        np.abs(17.5 - 16.6), 2
    )
