from typing import NoReturn

from mapie.utils import _raise_removed_v0_name_error

from .quantile_regression import ConformalizedQuantileRegressor
from .regression import (
    SplitConformalRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
)
from .time_series_regression import TimeSeriesRegressor

__all__ = [
    "TimeSeriesRegressor",
    "SplitConformalRegressor",
    "CrossConformalRegressor",
    "JackknifeAfterBootstrapRegressor",
    "ConformalizedQuantileRegressor",
]

_V0_TO_V1_NAMES = {
    "MapieRegressor": (
        "SplitConformalRegressor, CrossConformalRegressor "
        "or JackknifeAfterBootstrapRegressor"
    ),
    "MapieQuantileRegressor": "ConformalizedQuantileRegressor",
    "MapieTimeSeriesRegressor": "TimeSeriesRegressor",
}


def __getattr__(name: str) -> NoReturn:
    _raise_removed_v0_name_error(name, __name__, _V0_TO_V1_NAMES)
