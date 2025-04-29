from typing import Optional
import warnings

from sklearn.utils.multiclass import (check_classification_targets,
                                      type_of_target)

from .regression import BaseRegressionScore
from .classification import BaseClassificationScore
from .bounds import AbsoluteConformityScore
from .sets import LACConformityScore

from numpy.typing import ArrayLike


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


def check_depreciated_size_raps(
    size_raps: Optional[float]
) -> None:
    """
    Check if the parameter ``size_raps`` is used. If so, raise a warning.

    Raises
    ------
    Warning
        If ``size_raps`` is not ``None``.
    """
    if not (size_raps is None):
        warnings.warn(
            "WARNING: Deprecated parameter. "
            "The parameter `size_raps` is deprecated. "
            "In the next release, `RAPSConformityScore` takes precedence over "
            "`_MapieClassifier` for setting the size used. "
            "Prefer to define `size_raps` in `RAPSConformityScore` rather "
            "than in the `fit` method of `_MapieClassifier`.",
            DeprecationWarning
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
