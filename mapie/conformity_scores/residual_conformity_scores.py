from sklearn.utils import deprecated

from .bounds import (
    AbsoluteConformityScore as NewAbsoluteConformityScore,
    GammaConformityScore as NewGammaConformityScore,
    ResidualNormalisedScore as NewResidualNormalisedScore
)


@deprecated(
    "WARNING: Deprecated path to import AbsoluteConformityScore. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores.bounds import AbsoluteConformityScore]."
)
class AbsoluteConformityScore(NewAbsoluteConformityScore):
    pass


@deprecated(
    "WARNING: Deprecated path to import GammaConformityScore. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores.bounds import GammaConformityScore]."
)
class GammaConformityScore(NewGammaConformityScore):
    pass


@deprecated(
    "WARNING: Deprecated path to import ResidualNormalisedScore. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores.bounds import ResidualNormalisedScore]."
)
class ResidualNormalisedScore(NewResidualNormalisedScore):
    pass
