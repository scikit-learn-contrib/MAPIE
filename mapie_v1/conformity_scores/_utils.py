from typing import Union, TypeVar, Type
from mapie.conformity_scores.interface import BaseConformityScore
from . import CONFORMITY_SCORES_STRING_MAP

RegressionOrClassificationScore = TypeVar(
    'RegressionOrClassificationScore',
    bound=BaseConformityScore
)


def check_and_select_conformity_score(
    conformity_score: Union[str, RegressionOrClassificationScore],
    conformity_score_type: Type[RegressionOrClassificationScore]
) -> RegressionOrClassificationScore:
    if isinstance(conformity_score, conformity_score_type):
        return conformity_score
    elif conformity_score in CONFORMITY_SCORES_STRING_MAP[conformity_score_type]:
        return CONFORMITY_SCORES_STRING_MAP[conformity_score_type][conformity_score]()
    else:
        raise ValueError("Invalid conformity_score parameter")
