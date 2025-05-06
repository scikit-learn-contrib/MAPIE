from . import (
    classification,
    metrics,
    regression,
    utils,
    multi_label_classification,
    calibration,
    subsample,
)
from ._version import __version__

__all__ = [
    "regression",
    "classification",
    "multi_label_classification",
    "calibration",
    "metrics",
    "utils",
    "subsample",
    "__version__"
]
