import numpy as np
from covmetrics import CovGap

from mapie.utils import _transform_confidence_level_to_alpha


def _compute_cover(y, y_intervals=None, y_sets=None):
    if y_intervals is not None:
        if y_intervals.ndim == 3 and y_intervals.shape[-1] > 1:
            raise ValueError
        y_intervals = y_intervals.squeeze()
        y_low = y_intervals[:, 0]
        y_high = y_intervals[:, 1]
        cover = (y >= y_low) & (y <= y_high)
        return cover.astype(int)
    if y_sets is not None:
        if y_sets.ndim == 3 and y_sets.shape[-1] > 1:
            raise ValueError
        y_sets = y_sets.squeeze()
        y = np.expand_dims(y, axis=1)
        cover = np.take_along_axis(y_sets, y, axis=1).squeeze()
        return cover.squeeze().astype(int)


def coverage_gap(
    y, groups, confidence_level, y_intervals=None, y_sets=None, weighted=False
):
    cover = _compute_cover(y, y_intervals, y_sets)
    alpha = _transform_confidence_level_to_alpha(confidence_level)
    return CovGap().evaluate(groups, cover, alpha, weighted)
