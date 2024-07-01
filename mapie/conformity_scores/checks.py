from typing import Optional

from .conformity_scores import ConformityScore
from .residual_conformity_scores import AbsoluteConformityScore


def check_conformity_score(
    conformity_score: Optional[ConformityScore],
    sym: bool = True,
) -> ConformityScore:
    """
    Check parameter ``conformity_score``.

    Raises
    ------
    ValueError
        If parameter is not valid.

    Examples
    --------
    >>> from mapie.conformity_scores.checks import check_conformity_score
    >>> try:
    ...     check_conformity_score(1)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Invalid conformity_score argument.
    Must be None or a ConformityScore instance.
    """
    if conformity_score is None:
        return AbsoluteConformityScore(sym=sym)
    elif isinstance(conformity_score, ConformityScore):
        return conformity_score
    else:
        raise ValueError(
            "Invalid conformity_score argument.\n"
            "Must be None or a ConformityScore instance."
        )
