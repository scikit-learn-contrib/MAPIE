import numpy as np
from mapie.conformity_scores.utils_classification_conformity_scores import (
    get_true_label_position,
)


def test_get_true_label_position() -> None:
    y_pred_proba = np.array(
        [[0.1, 0.5, 0.4], [0.3, 0.2, 0.5], [0.2, 0.8, 0.0], [0.4, 0.35, 0.25]]
    )
    y = np.array([1, 2, 0, 1])
    y = np.reshape(
        y, (-1, 1)
    )  # add in order to have shape of form (4,1) instead of (4,)

    position = get_true_label_position(y_pred_proba, y)

    expected_position = np.array([[0], [0], [1], [1]])

    assert np.array_equal(position, expected_position)
    assert position.shape == y.shape
