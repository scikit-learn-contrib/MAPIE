from typing import Union
from mapie.conformity_scores import (
    BaseRegressionScore,
    AbsoluteConformityScore,
    GammaConformityScore,
    ResidualNormalisedScore,
)


def select_conformity_score(
    conformity_score: Union[str, BaseRegressionScore]
) -> BaseRegressionScore:
    if isinstance(conformity_score, BaseRegressionScore):
        return conformity_score
    elif conformity_score == "absolute":
        return AbsoluteConformityScore()
    elif conformity_score == "gamma":
        return GammaConformityScore()
    elif conformity_score == "residualsNorm":
        return ResidualNormalisedScore()
    else:
        raise ValueError("Invalid conformity_score type")
