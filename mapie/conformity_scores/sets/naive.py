from typing import Optional, Tuple, Union, cast

import numpy as np
from sklearn.dummy import check_random_state

from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import (
    check_include_last_label, check_proba_normalized, get_last_index_included
)
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray


class Naive(BaseClassificationScore):
    """
    Naive classification non-conformity score method that is based on the
    cumulative sum of probabilities until the 1-alpha threshold.

    Attributes
    ----------
    classes: Optional[ArrayLike]
        Names of the classes.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``get_sets`` method.
    """

    def __init__(self) -> None:
        super().__init__()

    def set_external_attributes(
        self,
        classes: Optional[ArrayLike] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        **kwargs
    ) -> None:
        """
        Set attributes that are not provided by the user.

        Parameters
        ----------
        classes: Optional[ArrayLike]
            Names of the classes.

            By default ``None``.

        random_state: Optional[Union[int, RandomState]]
            Pseudo random number generator state.
        """
        super().set_external_attributes(**kwargs)
        self.classes = classes
        self.random_state = random_state

    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Get the conformity score.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """
        conformity_scores = np.empty(y_pred.shape, dtype="float")
        return conformity_scores

    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        **kwargs
    ) -> NDArray:
        """
        Get predictions from an EnsembleClassifier.

        Parameters:
        -----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns:
        --------
        NDArray
            Array of predictions.
        """
        y_pred_proba = estimator.predict(X, agg_scores)
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )
        return y_pred_proba

    def get_conformity_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Get the quantiles of the conformity scores for each uncertainty level.

        Parameters:
        -----------
        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        Returns:
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        quantiles_ = 1 - alpha_np
        return quantiles_

    def _add_regualization(
        self,
        y_pred_proba_sorted_cumsum: NDArray,
        **kwargs
    ):
        """
        Add regularization to the sorted cumulative sum of predicted
        probabilities.

        Parameters
        ----------
        y_pred_proba_sorted_cumsum: NDArray of shape (n_samples, n_classes)
            The sorted cumulative sum of predicted probabilities.

        **kwargs: dict, optional
            Additional keyword arguments that might be used.
            The current implementation does not use any.

        Returns
        -------
        NDArray
            The adjusted cumulative sum of predicted probabilities after
            applying the regularization technique.
        """
        return y_pred_proba_sorted_cumsum

    def _get_last_included_proba(
        self,
        y_pred_proba: NDArray,
        thresholds: NDArray,
        include_last_label: Union[bool, str, None],
        **kwargs
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
            Whether to include or not the label whose score exceeds threshold.

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
        y_pred_proba_sorted_cumsum = np.cumsum(y_pred_proba_sorted, axis=1)
        y_pred_proba_sorted_cumsum = self._add_regualization(
            y_pred_proba_sorted_cumsum, **kwargs
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

    def _compute_vs_parameter(
        self,
        y_proba_last_cumsumed: NDArray,
        threshold: NDArray,
        y_pred_proba_last: NDArray,
        prediction_sets: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Compute the V parameters from Romano+(2020).

        Parameters:
        -----------
        y_proba_last_cumsumed: NDArray of shape (n_samples, n_alpha)
            Cumulated score of the last included label.

        threshold: NDArray of shape (n_alpha,) or shape (n_samples_train,)
            Threshold to compare with y_proba_last_cumsum.

        y_pred_proba_last: NDArray of shape (n_samples, 1, n_alpha)
            Last included probability.

        predicition_sets: NDArray of shape (n_samples, n_alpha)
            Prediction sets.

        Returns:
        --------
        NDArray of shape (n_samples, n_alpha)
            Vs parameters.
        """
        # compute V parameter from Romano+(2020)
        vs = (
            (y_proba_last_cumsumed - threshold.reshape(1, -1)) /
            y_pred_proba_last[:, 0, :]
        )
        return vs

    def _add_random_tie_breaking(
        self,
        prediction_sets: NDArray,
        y_pred_index_last: NDArray,
        y_pred_proba_cumsum: NDArray,
        y_pred_proba_last: NDArray,
        threshold: NDArray,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        **kwargs
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

            - the quantiles associated with alpha values when
                ``cv`` == "prefit", ``cv`` == "split"
                or ``agg_scores`` is "mean"

            - the conformity score from training samples otherwise (i.e., when
            ``cv`` is CV splitter and ``agg_scores`` is "crossval")

        method: str
            Method that determines how to remove last label in the prediction
            set.

            - if "cumulated_score" or "aps", compute V parameter
                from Romano+(2020)

            - else compute V parameter from Angelopoulos+(2020)

        lambda_star: Optional[Union[NDArray, float]] of shape (n_alpha):
            Optimal value of the regulizer lambda.

        k_star: Optional[NDArray] of shape (n_alpha):
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

        # get the V parameter from Romano+(2020) or Angelopoulos+(2020)
        vs = self._compute_vs_parameter(
            y_proba_last_cumsumed,
            threshold,
            y_pred_proba_last,
            prediction_sets
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

    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        include_last_label: Optional[Union[bool, str]] = True,
        **kwargs
    ) -> NDArray:
        """
        Generate prediction sets based on the probability predictions,
        the conformity scores and the uncertainty level.

        Parameters:
        -----------
        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Target prediction.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        include_last_label: Optional[Union[bool, str]]
            Whether or not to include last label in prediction sets.
            Choose among ``False``, ``True``  or ``"randomized"``.

            By default, ``True``.

        Returns:
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        include_last_label = check_include_last_label(include_last_label)

        # specify which thresholds will be used
        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            thresholds = self.quantiles_
        else:
            thresholds = conformity_scores.ravel()

        # sort labels by decreasing probability
        y_pred_proba_cumsum, y_pred_index_last, y_pred_proba_last = (
            self._get_last_included_proba(
                y_pred_proba,
                thresholds,
                include_last_label,
                prediction_phase=True,
                **kwargs
            )
        )
        # get the prediction set by taking all probabilities
        # above the last one
        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            y_pred_included = np.greater_equal(
                y_pred_proba - y_pred_proba_last, -EPSILON
            )
        else:
            y_pred_included = np.less_equal(
                y_pred_proba - y_pred_proba_last, EPSILON
            )
        # remove last label randomly
        if include_last_label == "randomized":
            y_pred_included = self._add_random_tie_breaking(
                y_pred_included,
                y_pred_index_last,
                y_pred_proba_cumsum,
                y_pred_proba_last,
                thresholds,
                self.random_state,
                **kwargs
            )
        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            prediction_sets = y_pred_included
        else:
            # compute the number of times the inequality is verified
            prediction_sets_summed = y_pred_included.sum(axis=2)
            prediction_sets = np.less_equal(
                prediction_sets_summed[:, :, np.newaxis]
                - self.quantiles_[np.newaxis, np.newaxis, :],
                EPSILON
            )

        return prediction_sets
