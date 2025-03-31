from typing import List

import numpy as np
import pytest

from mapie.conformity_scores.sets.utils import get_true_label_position
from numpy.typing import NDArray

Y_TRUE_PROBA_PLACE = [
    [
        np.array([2, 0]),
        np.array([
            [.1, .3, .6],
            [.2, .7, .1]
        ]),
        np.array([[0], [1]])
    ],
    [
        np.array([1, 0]),
        np.array([
            [.7, .12, .18],
            [.5, .24, .26]
        ]),
        np.array([[2], [0]])
    ]
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
def test_get_true_label_position(
    y_true_proba_place: List[NDArray]
) -> None:
    """
    Check that the returned true label position the good.
    """
    y_true = y_true_proba_place[0]
    y_pred_proba = y_true_proba_place[1]
    place = y_true_proba_place[2]

    found_place = get_true_label_position(y_pred_proba, y_true)

    assert (found_place == place).all()
