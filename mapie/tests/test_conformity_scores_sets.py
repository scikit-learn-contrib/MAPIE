from typing import Optional

import pytest

# from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import BaseClassificationScore
from mapie.conformity_scores.sets import APS, LAC, TopK
from mapie.conformity_scores.utils import check_classification_conformity_score


cs_list = [None, LAC(), APS(), TopK()]
method_list = [None, 'naive', 'aps', 'raps', 'lac', 'top_k']
wrong_method_list = ['naive_', 'aps_', 'raps_', 'lac_', 'top_k_']


def test_error_mother_class_initialization() -> None:
    with pytest.raises(TypeError):
        BaseClassificationScore()  # type: ignore


@pytest.mark.parametrize("conformity_score", cs_list)
def test_check_classification_conformity_score(
    conformity_score: Optional[BaseClassificationScore]
) -> None:
    assert isinstance(
        check_classification_conformity_score(conformity_score),
        BaseClassificationScore
    )


@pytest.mark.parametrize("method", method_list)
def test_check_classification_method(
    method: Optional[str]
) -> None:
    assert isinstance(
        check_classification_conformity_score(method=method),
        BaseClassificationScore
    )


@pytest.mark.parametrize("method", wrong_method_list)
def test_check_wrong_classification_method(
    method: Optional[str]
) -> None:
    with pytest.raises(ValueError, match="Invalid method.*"):
        check_classification_conformity_score(method=method)
