from .regression import BaseRegressionScore as ConformityScore  # noqa: F401

import warnings
warnings.warn(
    "Conformity score class is depreciated. Prefer import " +
    "BaseRegressionScore from mapie.conformity_scores",
    DeprecationWarning
)
