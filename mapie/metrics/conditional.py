from covmetrics import CovGap

MapieEstimator = Union[
    SplitConformalClassifier,
    SplitConformalRegressor,
]

def compute_cover(mapie_estimator, x, y):
    if isinstance(mapie_estimator, SplitConformalRegressor):
        _ (y_low, y_high) = mapie_estimator.predict_interval(x)
    return (y >= y_low | y <= y_high)

def coverage_gap(mapie_estimator, y, groups, alpha, weighted=False):
    cover = compute_cover(mapie_estimator, y)
    return CovGap().evaluate(groups, cover, alpha, weighted)
