from typing import List

import numpy as np
import pytest

from mapie.conformity_scores import (
    AbsoluteConformityScore,
    BaseRegressionScore,
    GammaConformityScore,
    LACConformityScore,
    BaseClassificationScore,
    TopKConformityScore,
)
from mapie.conformity_scores.sets.utils import get_true_label_position
from numpy.typing import NDArray

from mapie.conformity_scores.utils import check_and_select_conformity_score


class TestCheckAndSelectConformityScore:
    @pytest.mark.parametrize(
        "score, score_type, expected_class",
        [
            (AbsoluteConformityScore(), BaseRegressionScore, AbsoluteConformityScore),
            ("gamma", BaseRegressionScore, GammaConformityScore),
            (LACConformityScore(), BaseClassificationScore, LACConformityScore),
            ("top_k", BaseClassificationScore, TopKConformityScore),
        ],
    )
    def test_with_valid_inputs(self, score, score_type, expected_class):
        result = check_and_select_conformity_score(score, score_type)
        assert isinstance(result, expected_class)

    @pytest.mark.parametrize(
        "score_type", [BaseRegressionScore, BaseClassificationScore]
    )
    def test_with_invalid_input(self, score_type):
        with pytest.raises(ValueError):
            check_and_select_conformity_score("I'm not a valid input :(", score_type)


Y_TRUE_PROBA_PLACE = [
    [
        np.array([2, 0]),
        np.array([[0.1, 0.3, 0.6], [0.2, 0.7, 0.1]]),
        np.array([[0], [1]]),
    ],
    [
        np.array([1, 0]),
        np.array([[0.7, 0.12, 0.18], [0.5, 0.24, 0.26]]),
        np.array([[2], [0]]),
    ],
]


def test_shape_get_true_label_position() -> None:
    """
    Check the shape returned by the function
    """
    y_pred_proba = np.random.rand(5, 3)
    y = np.random.randint(0, 3, size=(5, 1))
    position = get_true_label_position(y_pred_proba, y)
    assert position.shape == y.shape


@pytest.mark.parametrize("y_true_proba_place", Y_TRUE_PROBA_PLACE)
def test_get_true_label_position(y_true_proba_place: List[NDArray]) -> None:
    """
    Check that the returned true label position the good.
    """
    y_true = y_true_proba_place[0]
    y_pred_proba = y_true_proba_place[1]
    place = y_true_proba_place[2]

    found_place = get_true_label_position(y_pred_proba, y_true)

    assert (found_place == place).all()
