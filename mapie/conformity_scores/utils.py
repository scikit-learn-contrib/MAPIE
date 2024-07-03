from typing import Optional

from .regression import BaseRegressionScore
from .classification import BaseClassificationScore
from .bounds import AbsoluteConformityScore
from .sets import APS, LAC, Naive, RAPS, TopK


def check_regression_conformity_score(
    conformity_score: Optional[BaseRegressionScore],
    sym: bool = True,
) -> BaseRegressionScore:
    """
    Check parameter ``conformity_score`` for regression task.

    Raises
    ------
    ValueError
        If parameters are not valid.

    Examples
    --------
    >>> from mapie.conformity_scores.utils import (
    ...     check_regression_conformity_score
    ... )
    >>> try:
    ...     check_regression_conformity_score(1)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Invalid conformity_score argument.
    Must be None or a ConformityScore instance.
    """
    if conformity_score is None:
        return AbsoluteConformityScore(sym=sym)
    elif isinstance(conformity_score, BaseRegressionScore):
        return conformity_score
    else:
        raise ValueError(
            "Invalid conformity_score argument.\n"
            "Must be None or a ConformityScore instance."
        )


def check_classification_conformity_score(
    conformity_score: Optional[BaseClassificationScore] = None,
    method: Optional[str] = None,
) -> BaseClassificationScore:
    """
    Check parameter ``conformity_score`` for classification task.

    Raises
    ------
    ValueError
        If parameters are not valid.

    Examples
    --------
    >>> from mapie.conformity_scores.utils import (
    ...     check_classification_conformity_score
    ... )
    >>> try:
    ...     check_classification_conformity_score(1)
    ... except Exception as exception:
    ...     print(exception)
    ...
    Invalid conformity_score argument.
    Must be None or a ConformityScore instance.
    """
    allowed_methods = ['lac', 'naive', 'aps', 'raps', 'top_k']
    deprecated_methods = ['score', 'cumulated_score']
    if method is not None:
        if method in ['score', 'lac']:
            return LAC()
        if method in ['cumulated_score', 'aps']:
            return APS()
        if method in ['naive']:
            return Naive()
        if method in ['raps']:
            return RAPS()
        if method in ['top_k']:
            return TopK()
        else:
            raise ValueError(
                f"Invalid method. Allowed values are {allowed_methods}. "
                f"Deprecated values are {deprecated_methods}. "
            )
    elif isinstance(conformity_score, BaseClassificationScore):
        return conformity_score
    elif conformity_score is None:
        return LAC()
    else:
        raise ValueError(
            "Invalid conformity_score argument.\n"
            "Must be None or a ConformityScore instance."
        )
