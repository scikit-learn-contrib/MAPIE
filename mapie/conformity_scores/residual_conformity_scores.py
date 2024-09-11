from sklearn.utils import deprecated

from .bounds import (
    AbsoluteConformityScore as OldAbsoluteConformityScore,
    GammaConformityScore as OldGammaConformityScore,
    ResidualNormalisedScore as OldResidualNormalisedScore
)


@deprecated(
    "WARNING: Deprecated path to import AbsoluteConformityScore. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores.bounds import AbsoluteConformityScore]."
)
class AbsoluteConformityScore(OldAbsoluteConformityScore):
    pass


@deprecated(
    "WARNING: Deprecated path to import GammaConformityScore. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores.bounds import GammaConformityScore]."
)
class GammaConformityScore(OldGammaConformityScore):
    pass


@deprecated(
    "WARNING: Deprecated path to import ResidualNormalisedScore. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores.bounds import ResidualNormalisedScore]."
)
class ResidualNormalisedScore(OldResidualNormalisedScore):
    pass
