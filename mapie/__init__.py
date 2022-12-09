from . import classification, metrics, regression
from ._version import __version__

__all__ = [
    "regression",
    "classification",
    "quantile_regression",
    "time_series_regression",
    "metrics",
    "__version__"
]
