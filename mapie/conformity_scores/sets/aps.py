from typing import Optional, Tuple, Union, cast

import numpy as np
from sklearn.dummy import check_random_state

from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import (
    add_random_tie_breaking, check_include_last_label, check_proba_normalized,
    get_last_included_proba, get_true_label_cumsum_proba
)
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray
from mapie.metrics import classification_mean_width_score
from mapie.utils import check_alpha_and_n_samples, compute_quantiles


class APS(BaseClassificationScore):
    """
    Adaptive Prediction Sets (APS) method-based non-conformity score.
    Three differents method are available in this class:

    - ``"naive"``, sum of the probabilities until the 1-alpha threshold.

    - ``"aps"`` (formerly called "cumulated_score"), Adaptive Prediction
        Sets method. It is based on the sum of the softmax outputs of the
        labels until the true label is reached, on the calibration set.
        See [1] for more details.

    - ``"raps"``, Regularized Adaptive Prediction Sets method. It uses the
        same technique as ``"aps"`` method but with a penalty term
        to reduce the size of prediction sets. See [2] for more
        details. For now, this method only works with ``"prefit"`` and
        ``"split"`` strategies.

    References
    ----------
    [1] Yaniv Romano, Matteo Sesia and Emmanuel J. Candès.
    "Classification with Valid and Adaptive Coverage."
    NeurIPS 202 (spotlight) 2020.

    [2] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan
    and Jitendra Malik.
    "Uncertainty Sets for Image Classifiers using Conformal Prediction."
    International Conference on Learning Representations 2021.
    """

    def __init__(
        self,
        consistency_check: bool = True,
        eps: float = float(EPSILON),
    ):
        super().__init__(
            consistency_check=consistency_check,
            eps=eps
        )

    def set_external_attributes(
        self,
        method: str = 'aps',
        classes: Optional[ArrayLike] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        **kwargs
    ) -> None:
        """
        Set attributes that are not provided by the user.

        Parameters
        ----------
        method: str
            Method to choose for prediction interval estimates.
            Methods available in this class: ``aps``, ``raps`` and ``naive``.

            By default ``aps`` for APS method.

        classes: Optional[ArrayLike]
            Names of the classes.

            By default ``None``.

        random_state: Optional[Union[int, RandomState]]
            Pseudo random number generator state.
        """
        super().set_external_attributes(**kwargs)
        self.method = method
        self.classes = classes
        self.random_state = random_state

    def get_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: ArrayLike,
        y_enc: Optional[ArrayLike] = None,
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
        y = cast(NDArray, y)
        y_pred = cast(NDArray, y_pred)
        y_enc = cast(NDArray, y_enc)
        classes = cast(NDArray, self.classes)

        # Conformity scores
        if self.method == "naive":
            conformity_scores = (
                np.empty(y_pred.shape, dtype="float")
            )
        else:
            conformity_scores, self.cutoff = (
                get_true_label_cumsum_proba(y, y_pred, classes)
            )
            y_proba_true = np.take_along_axis(
                y_pred, y_enc.reshape(-1, 1), axis=1
            )
            random_state = check_random_state(self.random_state)
            random_state = cast(np.random.RandomState, random_state)
            u = random_state.uniform(size=len(y_pred)).reshape(-1, 1)
            conformity_scores -= u * y_proba_true

        return conformity_scores

    def get_estimation_distribution(
        self,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike,
        **kwargs
    ) -> NDArray:
        """
        TODO
        Placeholder for ``get_estimation_distribution``.
        Subclasses should implement this method!

        Compute samples of the estimation distribution given the predicted
        targets and the conformity scores.

        Parameters
        ----------
        y_pred: NDArray of shape (n_samples, ...)
            Predicted target values.

        conformity_scores: NDArray of shape (n_samples, ...)
            Conformity scores.

        Returns
        -------
        NDArray of shape (n_samples, ...)
            Observed values.
        """
        return np.array([])

    @staticmethod
    def _regularize_conformity_score(
        k_star: NDArray,
        lambda_: Union[NDArray, float],
        conf_score: NDArray,
        cutoff: NDArray
    ) -> NDArray:
        """
        Regularize the conformity scores with the ``"raps"``
        method. See algo. 2 in [3].

        Parameters
        ----------
        k_star: NDArray of shape (n_alphas, )
            Optimal value of k (called k_reg in the paper). There
            is one value per alpha.

        lambda_: Union[NDArray, float] of shape (n_alphas, )
            One value of lambda for each alpha.

        conf_score: NDArray of shape (n_samples, 1)
            Conformity scores.

        cutoff: NDArray of shape (n_samples, 1)
            Position of the true label.

        Returns
        -------
        NDArray of shape (n_samples, 1, n_alphas)
            Regularized conformity scores. The regularization
            depends on the value of alpha.
        """
        conf_score = np.repeat(
            conf_score[:, :, np.newaxis], len(k_star), axis=2
        )
        cutoff = np.repeat(
            cutoff[:, np.newaxis], len(k_star), axis=1
        )
        conf_score += np.maximum(
            np.expand_dims(
                lambda_ * (cutoff - k_star),
                axis=1
            ),
            0
        )
        return conf_score

    def _update_size_and_lambda(
        self,
        best_sizes: NDArray,
        alpha_np: NDArray,
        y_ps: NDArray,
        lambda_: Union[NDArray, float],
        lambda_star: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """Update the values of the optimal lambda if the
        average size of the prediction sets decreases with
        this new value of lambda.

        Parameters
        ----------
        best_sizes: NDArray of shape (n_alphas, )
            Smallest average prediciton set size before testing
            for the new value of lambda_

        alpha_np: NDArray of shape (n_alphas)
            Level of confidences.

        y_ps: NDArray of shape (n_samples, n_classes, n_alphas)
            Prediction sets computed with the RAPS method and the
            new value of lambda_

        lambda_: NDArray of shape (n_alphas, )
            New value of lambda_star to test

        lambda_star: NDArray of shape (n_alphas, )
            Actual optimal lambda values for each alpha.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Arrays of shape (n_alphas, ) and (n_alpha, ) which
            respectively represent the updated values of lambda_star
            and the new best sizes.
        """

        sizes = [
            classification_mean_width_score(
                y_ps[:, :, i]
            ) for i in range(len(alpha_np))
        ]

        sizes_improve = (sizes < best_sizes - EPSILON)
        lambda_star = (
            sizes_improve * lambda_ + (1 - sizes_improve) * lambda_star
        )
        best_sizes = sizes_improve * sizes + (1 - sizes_improve) * best_sizes

        return lambda_star, best_sizes

    def _find_lambda_star(
        self,
        y_raps_no_enc: NDArray,
        y_pred_proba_raps: NDArray,
        alpha_np: NDArray,
        include_last_label: Union[bool, str, None],
        k_star: NDArray
    ) -> Union[NDArray, float]:
        """Find the optimal value of lambda for each alpha.

        Parameters
        ----------
        y_pred_proba_raps: NDArray of shape (n_samples, n_labels, n_alphas)
            Predictions of the model repeated on the last axis as many times
            as the number of alphas

        alpha_np: NDArray of shape (n_alphas, )
            Levels of confidences.

        include_last_label: bool
            Whether to include or not last label in
            the prediction sets

        k_star: NDArray of shape (n_alphas, )
            Values of k for the regularization.

        Returns
        -------
        ArrayLike of shape (n_alphas, )
            Optimal values of lambda.
        """
        classes = cast(NDArray, self.classes)

        lambda_star = np.zeros(len(alpha_np))
        best_sizes = np.full(len(alpha_np), np.finfo(np.float64).max)

        for lambda_ in [.001, .01, .1, .2, .5]:  # values given in paper[3]
            true_label_cumsum_proba, cutoff = (
                get_true_label_cumsum_proba(
                    y_raps_no_enc,
                    y_pred_proba_raps[:, :, 0],
                    classes
                )
            )

            true_label_cumsum_proba_reg = self._regularize_conformity_score(
                k_star,
                lambda_,
                true_label_cumsum_proba,
                cutoff
            )

            quantiles_ = compute_quantiles(
                true_label_cumsum_proba_reg,
                alpha_np
            )

            _, _, y_pred_proba_last = get_last_included_proba(
                y_pred_proba_raps,
                quantiles_,
                include_last_label,
                self.method,
                lambda_,
                k_star
            )

            y_ps = np.greater_equal(
                    y_pred_proba_raps - y_pred_proba_last, -EPSILON
            )
            lambda_star, best_sizes = self._update_size_and_lambda(
                best_sizes, alpha_np, y_ps, lambda_, lambda_star
            )
        if len(lambda_star) == 1:
            lambda_star = lambda_star[0]
        return lambda_star

    def get_sets(
        self,
        X: ArrayLike,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        conformity_scores: NDArray,
        include_last_label: Optional[Union[bool, str]] = True,
        agg_scores: Optional[str] = "mean",
        X_raps: Optional[NDArray] = None,
        y_raps_no_enc: Optional[NDArray] = None,
        y_pred_proba_raps: Optional[NDArray] = None,
        position_raps: Optional[NDArray] = None,
        **kwargs
    ):
        """
        Compute classes of the prediction sets from the observed values,
        the estimator of type ``EnsembleClassifier`` and the conformity scores.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores.

        TODO

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
            Prediction sets (Booleans indicate whether classes are included).
        """
        # Checks
        include_last_label = check_include_last_label(include_last_label)

        # if self.method == "raps":
        lambda_star, k_star = None, None
        X_raps = cast(NDArray, X_raps)
        y_raps_no_enc = cast(NDArray, y_raps_no_enc)
        y_pred_proba_raps = cast(NDArray, y_pred_proba_raps)
        position_raps = cast(NDArray, position_raps)

        n = len(conformity_scores)

        y_pred_proba = estimator.predict(X, agg_scores)
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )

        # Choice of the quantileif self.method == "naive":
        if self.method == "naive":
            self.quantiles_ = 1 - alpha_np
        elif (estimator.cv == "prefit") or (agg_scores in ["mean"]):
            if self.method == "raps":
                check_alpha_and_n_samples(alpha_np, X_raps.shape[0])
                k_star = compute_quantiles(
                    position_raps,
                    alpha_np
                ) + 1
                y_pred_proba_raps = np.repeat(
                    y_pred_proba_raps[:, :, np.newaxis],
                    len(alpha_np),
                    axis=2
                )
                lambda_star = self._find_lambda_star(
                    y_raps_no_enc,
                    y_pred_proba_raps,
                    alpha_np,
                    include_last_label,
                    k_star
                )
                conformity_scores_regularized = (
                    self._regularize_conformity_score(
                        k_star,
                        lambda_star,
                        conformity_scores,
                        self.cutoff
                    )
                )
                self.quantiles_ = compute_quantiles(
                    conformity_scores_regularized,
                    alpha_np
                )
            else:
                self.quantiles_ = compute_quantiles(
                    conformity_scores,
                    alpha_np
                )
        else:
            self.quantiles_ = (n + 1) * (1 - alpha_np)

        # Build prediction sets
        # specify which thresholds will be used
        if (estimator.cv == "prefit") or (agg_scores in ["mean"]):
            thresholds = self.quantiles_
        else:
            thresholds = conformity_scores.ravel()
        # sort labels by decreasing probability
        y_pred_proba_cumsum, y_pred_index_last, y_pred_proba_last = (
            get_last_included_proba(
                y_pred_proba,
                thresholds,
                include_last_label,
                self.method,
                lambda_star,
                k_star,
            )
        )
        # get the prediction set by taking all probabilities
        # above the last one
        if (estimator.cv == "prefit") or (agg_scores in ["mean"]):
            y_pred_included = np.greater_equal(
                y_pred_proba - y_pred_proba_last, -EPSILON
            )
        else:
            y_pred_included = np.less_equal(
                y_pred_proba - y_pred_proba_last, EPSILON
            )
        # remove last label randomly
        if include_last_label == "randomized":
            y_pred_included = add_random_tie_breaking(
                y_pred_included,
                y_pred_index_last,
                y_pred_proba_cumsum,
                y_pred_proba_last,
                thresholds,
                self.method,
                self.random_state,
                lambda_star,
                k_star,
            )
        if (estimator.cv == "prefit") or (agg_scores in ["mean"]):
            prediction_sets = y_pred_included
        else:
            # compute the number of times the inequality is verified
            prediction_sets_summed = y_pred_included.sum(axis=2)
            prediction_sets = np.less_equal(
                prediction_sets_summed[:, :, np.newaxis]
                - self.quantiles_[np.newaxis, np.newaxis, :],
                EPSILON
            )

        # Just for coverage: do nothing
        self.get_estimation_distribution(y_pred_proba, conformity_scores)

        return prediction_sets
