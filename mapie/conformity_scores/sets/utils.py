from typing import Optional, Tuple, Union, cast
import numpy as np
from sklearn.calibration import label_binarize

from mapie._typing import ArrayLike, NDArray
from mapie._machine_precision import EPSILON


def get_true_label_position(
    y_pred_proba: NDArray,
    y: NDArray
) -> NDArray:
    """
    Return the sorted position of the true label in the prediction

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


def get_true_label_cumsum_proba(
    y: ArrayLike,
    y_pred_proba: NDArray,
    classes: ArrayLike
) -> Tuple[NDArray, NDArray]:
    """
    Compute the cumsumed probability of the true label.

    Parameters
    ----------
    y: NDArray of shape (n_samples, )
        Array with the labels.

    y_pred_proba: NDArray of shape (n_samples, n_classes)
        Predictions of the model.

    classes: NDArray of shape (n_classes, )
        Array with the classes.

    Returns
    -------
    Tuple[NDArray, NDArray] of shapes (n_samples, 1) and (n_samples, ).
        The first element is the cumsum probability of the true label.
        The second is the sorted position of the true label.
    """
    y_true = label_binarize(y=y, classes=classes)
    index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
    y_pred_sorted = np.take_along_axis(y_pred_proba, index_sorted, axis=1)
    y_true_sorted = np.take_along_axis(y_true, index_sorted, axis=1)
    y_pred_sorted_cumsum = np.cumsum(y_pred_sorted, axis=1)
    cutoff = np.argmax(y_true_sorted, axis=1)
    true_label_cumsum_proba = np.take_along_axis(
        y_pred_sorted_cumsum, cutoff.reshape(-1, 1), axis=1
    )

    return true_label_cumsum_proba, cutoff + 1


def check_include_last_label(
    include_last_label: Optional[Union[bool, str]]
) -> Optional[Union[bool, str]]:
    """
    Check if ``include_last_label`` is a boolean or a string.
    Else raise error.

    Parameters
    ----------
    include_last_label: Optional[Union[bool, str]]
        Whether or not to include last label in
        prediction sets for the ``"aps"`` method. Choose among:

        - ``False``, does not include label whose cumulated score is just
            over the quantile.

        - ``True``, includes label whose cumulated score is just over the
            quantile, unless there is only one label in the prediction set.

        - ``"randomized"``, randomly includes label whose cumulated score
            is just over the quantile based on the comparison of a uniform
            number and the difference between the cumulated score of the last
            label and the quantile.

    Returns
    -------
    Optional[Union[bool, str]]

    Raises
    ------
    ValueError
        "Invalid include_last_label argument. "
        "Should be a boolean or 'randomized'."
    """
    if (
        (not isinstance(include_last_label, bool)) and
        (not include_last_label == "randomized")
    ):
        raise ValueError(
            "Invalid include_last_label argument. "
            "Should be a boolean or 'randomized'."
        )
    else:
        return include_last_label


def check_proba_normalized(
    y_pred_proba: ArrayLike,
    axis: int = 1
) -> NDArray:
    """
    Check if for all the samples the sum of the probabilities is equal to one.

    Parameters
    ----------
    y_pred_proba: ArrayLike of shape (n_samples, n_classes) or
    (n_samples, n_train_samples, n_classes)
        Softmax output of a model.

    Returns
    -------
    ArrayLike of shape (n_samples, n_classes)
        Softmax output of a model if the scores all sum to one.

    Raises
    ------
    ValueError
        If the sum of the scores is not equal to one.
    """
    sum_proba = np.sum(y_pred_proba, axis=axis)
    err_msg = "The sum of the scores is not equal to one."
    np.testing.assert_allclose(sum_proba, 1, err_msg=err_msg, rtol=1e-5)
    y_pred_proba = cast(NDArray, y_pred_proba).astype(np.float64)

    return y_pred_proba


def get_last_index_included(
    y_pred_proba_cumsum: NDArray,
    threshold: NDArray,
    include_last_label: Optional[Union[bool, str]]
) -> NDArray:
    """
    Return the index of the last included sorted probability
    depending if we included the first label over the quantile
    or not.

    Parameters
    ----------
    y_pred_proba_cumsum: NDArray of shape (n_samples, n_classes)
        Cumsumed probabilities in the original order.

    threshold: NDArray of shape (n_alpha,) or shape (n_samples_train,)
        Threshold to compare with y_proba_last_cumsum, can be either:

        - the quantiles associated with alpha values when
            ``cv`` == "prefit", ``cv`` == "split"
            or ``agg_scores`` is "mean"

        - the conformity score from training samples otherwise
            (i.e., when ``cv`` is a CV splitter and
            ``agg_scores`` is "crossval")

    include_last_label: Union[bool, str]
        Whether or not include the last label. If 'randomized',
        the last label is included.

    Returns
    -------
    NDArray of shape (n_samples, n_alpha)
        Index of the last included sorted probability.
    """
    if include_last_label or include_last_label == 'randomized':
        y_pred_index_last = (
            np.ma.masked_less(
                y_pred_proba_cumsum
                - threshold[np.newaxis, :],
                -EPSILON
            ).argmin(axis=1)
        )
    else:
        max_threshold = np.maximum(
            threshold[np.newaxis, :],
            np.min(y_pred_proba_cumsum, axis=1)
        )
        y_pred_index_last = np.argmax(
            np.ma.masked_greater(
                y_pred_proba_cumsum - max_threshold[:, np.newaxis, :],
                EPSILON
            ), axis=1
        )
    return y_pred_index_last[:, np.newaxis, :]
