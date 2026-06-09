"""
Test that accessing v0 class names removed in MAPIE v1 raises a helpful
error pointing to the migration guide, while unknown names keep raising
a plain AttributeError.
"""

from __future__ import annotations

import re
from types import ModuleType

import pytest

import mapie.calibration
import mapie.classification
import mapie.regression

MIGRATION_GUIDE_URL = (
    "https://contrib.scikit-learn.org/MAPIE/stable/getting-started/v1-release-notes/"
)


@pytest.mark.parametrize(
    "module, v0_name, v1_names",
    [
        (
            mapie.regression,
            "MapieRegressor",
            "SplitConformalRegressor, CrossConformalRegressor "
            "or JackknifeAfterBootstrapRegressor",
        ),
        (
            mapie.regression,
            "MapieQuantileRegressor",
            "ConformalizedQuantileRegressor",
        ),
        (
            mapie.regression,
            "MapieTimeSeriesRegressor",
            "TimeSeriesRegressor",
        ),
        (
            mapie.classification,
            "MapieClassifier",
            "SplitConformalClassifier or CrossConformalClassifier",
        ),
        (
            mapie.calibration,
            "MapieCalibrator",
            "TopLabelCalibrator",
        ),
    ],
)
def test_removed_v0_name_raises_migration_aware_error(
    module: ModuleType, v0_name: str, v1_names: str
) -> None:
    expected_message = (
        f"{v0_name} was removed in MAPIE v1. "
        f"Use {v1_names} instead. "
        f"See the migration guide: {MIGRATION_GUIDE_URL}"
    )
    with pytest.raises(ImportError, match=re.escape(expected_message)):
        getattr(module, v0_name)


def test_from_import_of_removed_v0_name_raises_migration_aware_error() -> None:
    with pytest.raises(ImportError, match="MapieRegressor was removed in MAPIE v1"):
        from mapie.regression import MapieRegressor  # noqa: F401


@pytest.mark.parametrize(
    "module", [mapie.regression, mapie.classification, mapie.calibration]
)
def test_unknown_name_raises_plain_attribute_error(module: ModuleType) -> None:
    with pytest.raises(
        AttributeError,
        match=f"module {module.__name__!r} has no attribute 'unknown_name'",
    ):
        _ = module.unknown_name
