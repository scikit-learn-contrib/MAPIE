from typing import Optional, no_type_check

from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
)

from .regression import BaseRegressionScore
from .classification import BaseClassificationScore
from .bounds import (
    AbsoluteConformityScore,
    GammaConformityScore,
    ResidualNormalisedScore,
)
from .sets import (
    LACConformityScore,
    TopKConformityScore,
    APSConformityScore,
    RAPSConformityScore,
)

from numpy.typing import ArrayLike


CONFORMITY_SCORES_STRING_MAP = {
    BaseRegressionScore: {
        "absolute": AbsoluteConformityScore,
        "gamma": GammaConformityScore,
        "residual_normalized": ResidualNormalisedScore,
    },
    BaseClassificationScore: {
        "lac": LACConformityScore,
        "top_k": TopKConformityScore,
        "aps": APSConformityScore,
        "raps": RAPSConformityScore,
    },
}


@no_type_check  # Cumbersome to type
def check_and_select_conformity_score(conformity_score, conformity_score_type):
    if isinstance(conformity_score, conformity_score_type):
        return conformity_score
    elif conformity_score in CONFORMITY_SCORES_STRING_MAP[conformity_score_type]:
        return CONFORMITY_SCORES_STRING_MAP[conformity_score_type][conformity_score]()
    else:
        raise ValueError("Invalid conformity_score parameter")


def check_regression_conformity_score(
    conformity_score: Optional[BaseRegressionScore],
    sym: bool = True,
) -> BaseRegressionScore:
    """
    Check parameter ``conformity_score`` for regression task.
    By default, return a AbsoluteConformityScore instance.

    Parameters
    ----------
    conformity_score: BaseClassificationScore
        Conformity score function.

        By default, `None`.

    sym: bool
        Whether to use symmetric bounds.

        By default, `True`.

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
    Must be None or a BaseRegressionScore instance.
    """
    if conformity_score is None:
        return AbsoluteConformityScore(sym=sym)
    elif isinstance(conformity_score, BaseRegressionScore):
        return conformity_score
    else:
        raise ValueError(
            "Invalid conformity_score argument.\n"
            "Must be None or a BaseRegressionScore instance."
        )


def check_target(
    conformity_score: BaseClassificationScore,
    y: ArrayLike
) -> None:
    """
    Check that if the type of target is binary,
    (then the method have to be ``"lac"``), or multi-class.

    Parameters
    ----------
    conformity_score: BaseClassificationScore
        Conformity score function.

    y: NDArray of shape (n_samples,)
        Training labels.

    Raises
    ------
    ValueError
        If type of target is binary and method is not ``"lac"``
        or ``"score"`` or if type of target is not multi-class.
    """
    check_classification_targets(y)
    if (
        type_of_target(y) == "binary" and
        not isinstance(conformity_score, LACConformityScore)
    ):
        raise ValueError(
            "Invalid conformity score for binary target. "
            "The only valid score is 'lac'."
        )


def check_classification_conformity_score(
    conformity_score: Optional[BaseClassificationScore] = None,
) -> BaseClassificationScore:
    """
    Check parameter ``conformity_score`` for classification task.
    By default, return a LACConformityScore instance.

    Parameters
    ----------
    conformity_score: BaseClassificationScore
        Conformity score function.

        By default, `None`.

    Raises
    ------
    ValueError
        If conformity_score is not valid.
    """
    if conformity_score is not None:
        if isinstance(conformity_score, BaseClassificationScore):
            return conformity_score
        else:
            raise ValueError(
                "Invalid conformity_score argument.\n"
                "Must be None or a BaseClassificationScore instance."
            )
    else:
        return LACConformityScore()
