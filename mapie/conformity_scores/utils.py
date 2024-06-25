import numpy as np
from mapie._typing import NDArray


def get_true_label_position(y_pred_proba: NDArray, y: NDArray) -> NDArray:
    """
    Return the sorted position of the true label in the
    prediction

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_classes)
        Model prediction.

    y: NDArray of shape (n_samples)
        Labels.

    Returns
    -------
    NDArray of shape (n_samples, 1)
        Position of the true label in the prediction.
    """
    index = np.argsort(np.fliplr(np.argsort(y_pred_proba, axis=1)))
    position = np.take_along_axis(index, y.reshape(-1, 1), axis=1)

    return position
