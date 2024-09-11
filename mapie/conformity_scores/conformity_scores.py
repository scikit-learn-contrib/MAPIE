import warnings
warnings.warn(
    "Conformity score class is depreciated. Prefer import " +
    "BaseRegressionScore from mapie.conformity_scores",
    DeprecationWarning
)

from .regression import (  # noqa: F401, E402
    BaseRegressionScore as ConformityScore
)
