from typing import Union

import numpy as np
from covmetrics import CovGap

from mapie.classification import CrossConformalClassifier, SplitConformalClassifier
from mapie.regression import (
    ConformalizedQuantileRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    SplitConformalRegressor,
    TimeSeriesRegressor,
)

MapieRegressors = Union[
    SplitConformalRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    ConformalizedQuantileRegressor,
    TimeSeriesRegressor,
]

MapieClassifiers = Union[
    SplitConformalClassifier,
    CrossConformalClassifier,
]


def _compute_cover(mapie_estimator, x, y):
    if isinstance(mapie_estimator, MapieRegressors):
        _, y_intervals = mapie_estimator.predict_interval(x)
        y_low = y_intervals[:, 0, :]
        y_high = y_intervals[:, 1, :]
        n_confidence_level = y_high.shape[1]
        y_per_alpha = np.tile(y, (n_confidence_level, 1)).transpose()
        return ((y_per_alpha >= y_low) & (y_per_alpha <= y_high)).astype(int)


def coverage_gap(mapie_estimator, x, y, groups, alpha, weighted=False):
    cover = compute_cover(mapie_estimator, x, y)
    return CovGap().evaluate(groups, cover, alpha, weighted)
