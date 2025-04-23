from sklearn.utils import deprecated

from mapie.regression import TimeSeriesRegressor as NewClass


@deprecated(
    "WARNING: Deprecated path to import TimeSeriesRegressor. "
    "Please prefer the new path: "
    "[from mapie.regression import TimeSeriesRegressor]."
)
class TimeSeriesRegressor(NewClass):
    pass
