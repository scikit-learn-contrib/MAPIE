from importlib.metadata import PackageNotFoundError, version

from . import (
    classification,
    metrics,
    regression,
    utils,
    risk_control,
    calibration,
    subsample,
)

try:
    __version__ = version("mapie")
except PackageNotFoundError:  # pragma: no cover
    # Fallback for source-only usage without installed metadata.
    __version__ = "0+unknown"

__all__ = [
    "regression",
    "classification",
    "risk_control",
    "calibration",
    "metrics",
    "utils",
    "subsample",
    "__version__",
]
