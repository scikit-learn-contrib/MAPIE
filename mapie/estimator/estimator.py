from sklearn.utils import deprecated

from mapie.estimator.regressor import EnsembleRegressor as NewEnsembleRegressor


@deprecated(
    "WARNING: Deprecated path to import EnsembleRegressor. "
    "Please prefer the new path: "
    "[from mapie.estimator.regressor import EnsembleRegressor]."
)
class EnsembleRegressor(NewEnsembleRegressor):
    pass
