from typing import Any, Optional, Tuple, Union, cast
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.dummy import check_random_state

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


def get_last_included_proba(
    y_pred_proba: NDArray,
    thresholds: NDArray,
    include_last_label: Union[bool, str, None],
    method: str,
    lambda_: Union[NDArray, float, None],
    k_star: Union[NDArray, Any]
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Function that returns the smallest score
    among those which are included in the prediciton set.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_classes)
        Predictions of the model.

    thresholds: NDArray of shape (n_alphas, )
        Quantiles that have been computed from the conformity scores.

    include_last_label: Union[bool, str, None]
        Whether to include or not the label whose score exceeds the threshold.

    lambda_: Union[NDArray, float, None] of shape (n_alphas)
        Values of lambda for the regularization.

    k_star: Union[NDArray, Any]
        Values of k for the regularization.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, ArrayLike]
        Arrays of shape (n_samples, n_classes, n_alphas),
        (n_samples, 1, n_alphas) and (n_samples, 1, n_alphas).
        They are respectively the cumsumed scores in the original
        order which can be different according to the value of alpha
        with the RAPS method, the index of the last included score
        and the value of the last included score.
    """
    index_sorted = np.flip(
        np.argsort(y_pred_proba, axis=1), axis=1
    )
    # sort probabilities by decreasing order
    y_pred_proba_sorted = np.take_along_axis(
        y_pred_proba, index_sorted, axis=1
    )
    # get sorted cumulated score
    y_pred_proba_sorted_cumsum = np.cumsum(
        y_pred_proba_sorted, axis=1
    )

    if method == "raps":
        y_pred_proba_sorted_cumsum += lambda_ * np.maximum(
            0,
            np.cumsum(
                np.ones(y_pred_proba_sorted_cumsum.shape), axis=1
            ) - k_star
        )
    # get cumulated score at their original position
    y_pred_proba_cumsum = np.take_along_axis(
        y_pred_proba_sorted_cumsum,
        np.argsort(index_sorted, axis=1),
        axis=1
    )
    # get index of the last included label
    y_pred_index_last = get_last_index_included(
        y_pred_proba_cumsum,
        thresholds,
        include_last_label
    )
    # get the probability of the last included label
    y_pred_proba_last = np.take_along_axis(
        y_pred_proba,
        y_pred_index_last,
        axis=1
    )

    zeros_scores_proba_last = (y_pred_proba_last <= EPSILON)

    # If the last included proba is zero, change it to the
    # smallest non-zero value to avoid inluding them in the
    # prediction sets.
    if np.sum(zeros_scores_proba_last) > 0:
        y_pred_proba_last[zeros_scores_proba_last] = np.expand_dims(
            np.min(
                np.ma.masked_less(
                    y_pred_proba,
                    EPSILON
                ).filled(fill_value=np.inf),
                axis=1
            ), axis=1
        )[zeros_scores_proba_last]

    return y_pred_proba_cumsum, y_pred_index_last, y_pred_proba_last


def add_random_tie_breaking(
    prediction_sets: NDArray,
    y_pred_index_last: NDArray,
    y_pred_proba_cumsum: NDArray,
    y_pred_proba_last: NDArray,
    threshold: NDArray,
    method: str,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
    lambda_star: Optional[Union[NDArray, float]] = None,
    k_star: Optional[Union[NDArray, None]] = None
) -> NDArray:
    """
    Randomly remove last label from prediction set based on the
    comparison between a random number and the difference between
    cumulated score of the last included label and the quantile.

    Parameters
    ----------
    prediction_sets: NDArray of shape
        (n_samples, n_classes, n_threshold)
        Prediction set for each observation and each alpha.

    y_pred_index_last: NDArray of shape (n_samples, threshold)
        Index of the last included label.

    y_pred_proba_cumsum: NDArray of shape (n_samples, n_classes)
        Cumsumed probability of the model in the original order.

    y_pred_proba_last: NDArray of shape (n_samples, 1, threshold)
        Last included probability.

    threshold: NDArray of shape (n_alpha,) or shape (n_samples_train,)
        Threshold to compare with y_proba_last_cumsum, can be either:

        - the quantiles associated with alpha values when ``cv`` == "prefit",
            ``cv`` == "split" or ``agg_scores`` is "mean"

        - the conformity score from training samples otherwise
            (i.e., when ``cv`` is CV splitter and ``agg_scores`` is "crossval")

    method: str
        Method that determines how to remove last label in the prediction set.

        - if "cumulated_score" or "aps", compute V parameter from Romano+(2020)

        - else compute V parameter from Angelopoulos+(2020)

    lambda_star: Union[NDArray, float, None] of shape (n_alpha):
        Optimal value of the regulizer lambda.

    k_star: Union[NDArray, None] of shape (n_alpha):
        Optimal value of the regulizer k.

    Returns
    -------
    NDArray of shape (n_samples, n_classes, n_alpha)
        Updated version of prediction_sets with randomly removed labels.
    """
    # get cumsumed probabilities up to last retained label
    y_proba_last_cumsumed = np.squeeze(
        np.take_along_axis(
            y_pred_proba_cumsum,
            y_pred_index_last,
            axis=1
        ), axis=1
    )

    if method in ["cumulated_score", "aps"]:
        # compute V parameter from Romano+(2020)
        vs = (
            (y_proba_last_cumsumed - threshold.reshape(1, -1)) /
            y_pred_proba_last[:, 0, :]
        )
    else:
        # compute V parameter from Angelopoulos+(2020)
        L = np.sum(prediction_sets, axis=1)
        vs = (
            (y_proba_last_cumsumed - threshold.reshape(1, -1)) /
            (
                y_pred_proba_last[:, 0, :] -
                lambda_star * np.maximum(0, L - k_star) +
                lambda_star * (L > k_star)
            )
        )

    # get random numbers for each observation and alpha value
    random_state = check_random_state(random_state)
    random_state = cast(np.random.RandomState, random_state)
    us = random_state.uniform(size=(prediction_sets.shape[0], 1))
    # remove last label from comparison between uniform number and V
    vs_less_than_us = np.less_equal(vs - us, EPSILON)
    np.put_along_axis(
        prediction_sets,
        y_pred_index_last,
        vs_less_than_us[:, np.newaxis, :],
        axis=1
    )
    return prediction_sets
