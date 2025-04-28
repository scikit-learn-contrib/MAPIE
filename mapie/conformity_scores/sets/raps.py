from typing import Optional, Tuple, Union, cast

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples

from mapie.conformity_scores.sets.aps import APSConformityScore
from mapie.conformity_scores.sets.utils import get_true_label_position
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from numpy.typing import NDArray
from mapie.metrics.classification import classification_mean_width_score
from mapie.utils import _check_alpha_and_n_samples, _compute_quantiles


class RAPSConformityScore(APSConformityScore):
    """
    Regularized Adaptive Prediction Sets (RAPS) method-based non-conformity
    score. It uses the same technique as ``APSConformityScore`` class but with
    a penalty term to reduce the size of prediction sets. See [1] for more
    details. For now, this method only works with ``"prefit"`` and ``"split"``
    strategies.

    References
    ----------
    [1] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan
    and Jitendra Malik.
    "Uncertainty Sets for Image Classifiers using Conformal Prediction."
    International Conference on Learning Representations 2021.

    Parameters
    ----------
    size_raps: Optional[float]
        Percentage of the data to be used for choosing lambda_star and
        k_star for the RAPS method.

    Attributes
    ----------
    classes: ArrayLike
        Names of the classes.

    random_state: Union[int, RandomState]
        Pseudo random number generator state.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``get_sets`` method.

    label_encoder: LabelEncoder
        The label encoder used to encode the labels.

    size_raps: float
        Percentage of the data to be used for choosing lambda_star and
        k_star for the RAPS method.
    """

    def __init__(
        self,
        size_raps: Optional[float] = 0.2
    ) -> None:
        super().__init__()
        self.size_raps = size_raps

    def set_external_attributes(
        self,
        *,
        label_encoder: Optional[LabelEncoder] = None,
        size_raps: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Set attributes that are not provided by the user.

        Parameters
        ----------
        label_encoder: Optional[LabelEncoder]
            The label encoder used to encode the labels.

            By default ``None``.

        size_raps: Optional[float]
            Percentage of the data to be used for choosing lambda_star and
            k_star for the RAPS method.

            By default ``None``.
        """
        super().set_external_attributes(**kwargs)
        self.label_encoder_ = cast(LabelEncoder, label_encoder)
        self.size_raps = size_raps

    def split_data(
        self,
        X: NDArray,
        y: NDArray,
        y_enc: NDArray,
        sample_weight: Optional[NDArray] = None,
        groups: Optional[NDArray] = None,
    ):
        """
        Split data. Keeps part of the data for the calibration estimator
        (separate from the calibration data).

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y: ArrayLike
            Target values.

        y_enc: ArrayLike
            Target values as normalized encodings.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Non-null sample weights.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            By default ``None``.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray, Optional[NDArray],
        Optional[NDArray]]
            - X: NDArray of shape (n_samples, n_features)
            - y: NDArray of shape (n_samples,)
            - y_enc: NDArray of shape (n_samples,)
            - sample_weight: Optional[NDArray] of shape (n_samples,)
            - groups: Optional[NDArray] of shape (n_samples,)
        """
        # Split data for raps method
        raps_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.size_raps, random_state=self.random_state
        )
        train_raps_index, val_raps_index = next(raps_split.split(X, y_enc))
        X, self.X_raps, y_enc, self.y_raps = (
            _safe_indexing(X, train_raps_index),
            _safe_indexing(X, val_raps_index),
            _safe_indexing(y_enc, train_raps_index),
            _safe_indexing(y_enc, val_raps_index),
        )

        # Decode y_raps for use in the RAPS method
        self.y_raps_no_enc = self.label_encoder_.inverse_transform(self.y_raps)
        y = self.label_encoder_.inverse_transform(y_enc)

        # Cast to NDArray for type checking
        y_enc = cast(NDArray, y_enc)
        if sample_weight is not None:
            sample_weight = cast(NDArray, sample_weight)
            sample_weight = sample_weight[train_raps_index]
        if groups is not None:
            groups = cast(NDArray, groups)
            groups = groups[train_raps_index]

        # Keep sample data size for training and calibration
        self.n_samples_ = _num_samples(y_enc)

        return X, y, y_enc, sample_weight, groups

    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        y_enc: Optional[NDArray] = None,
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

        y_enc: NDArray of shape (n_samples,)
            Target values as normalized encodings.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """
        # Compute y_pred and position on the RAPS validation dataset
        self.y_pred_proba_raps = (
            self.predictor.single_estimator_.predict_proba(self.X_raps)
        )
        self.position_raps = get_true_label_position(
            self.y_pred_proba_raps, self.y_raps
        )

        return super().get_conformity_scores(
            y, y_pred, y_enc=y_enc, **kwargs
        )

    @staticmethod
    def _regularize_conformity_score(
        k_star: NDArray,
        lambda_: Union[NDArray, float],
        conf_score: NDArray,
        cutoff: NDArray
    ) -> NDArray:
        """
        Regularize the conformity scores with the ``"raps"``
        method. See algo. 2 in [1].

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
            np.expand_dims(lambda_ * (cutoff - k_star), axis=1), 0
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
        """
        Update the values of the optimal lambda if the average size of the
        prediction sets decreases with this new value of lambda.

        Parameters
        ----------
        best_sizes: NDArray of shape (n_alphas, )
            Smallest average prediction set size before testing
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
        sizes = classification_mean_width_score(y_ps)

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
        """
        Find the optimal value of lambda for each alpha.

        Parameters
        ----------
        y_raps_no_enc: NDArray of shape (n_samples, )
            True labels (after applying `label_encoder_.inverse_transform`).

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

        for lambda_ in [.001, .01, .1, .2, .5]:  # values given in paper[1]
            true_label_cumsum_proba, cutoff = (
                self.get_true_label_cumsum_proba(
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

            quantiles_ = _compute_quantiles(
                true_label_cumsum_proba_reg,
                alpha_np
            )

            _, _, y_pred_proba_last = self._get_last_included_proba(
                y_pred_proba_raps,
                quantiles_,
                include_last_label,
                lambda_=lambda_,
                k_star=k_star
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

    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        include_last_label: Optional[Union[bool, str]] = True,
        **kwargs
    ) -> NDArray:
        """
        Get the quantiles of the conformity scores for each uncertainty level.

        Parameters
        -----------
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

            By default, ``"mean"``.

        include_last_label: Optional[Union[bool, str]]
            Whether or not to include last label in prediction sets.
            Choose among ``False``, ``True``  or ``"randomized"``.

            By default, ``True``.

            See the docstring of
            :meth:`conformity_scores.sets.aps.APSConformityScore.get_prediction_sets`
            for more details.

        X_raps: NDArray of shape (n_samples, n_features)
            Observed feature values for the RAPS method (split data).

            By default, "None" but must be set to work.

        y_raps_no_enc: NDArray of shape (n_samples,)
            Observed labels for the RAPS method (split data).

            By default, "None" but must be set to work.

        y_pred_proba_raps: NDArray of shape (n_samples, n_classes)
            Predicted probabilities for the RAPS method (split data).

            By default, "None" but must be set to work.

        position_raps: NDArray of shape (n_samples,)
            Position of the points in the split set for the RAPS method
            (split data). These positions are returned by the function
            ``get_true_label_position``.

            By default, "None" but must be set to work.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        # Casting to NDArray to avoid mypy errors
        # X_raps = cast(NDArray, X_raps)
        # y_raps_no_enc = cast(NDArray, y_raps_no_enc)
        # y_pred_proba_raps = cast(NDArray, y_pred_proba_raps)
        # position_raps = cast(NDArray, position_raps)

        _check_alpha_and_n_samples(alpha_np, self.X_raps.shape[0])
        self.k_star = _compute_quantiles(
            self.position_raps,
            alpha_np
        ) + 1
        y_pred_proba_raps = np.repeat(
            self.y_pred_proba_raps[:, :, np.newaxis],
            len(alpha_np),
            axis=2
        )
        self.lambda_star = self._find_lambda_star(
            self.y_raps_no_enc,
            y_pred_proba_raps,
            alpha_np,
            include_last_label,
            self.k_star
        )
        conformity_scores_regularized = (
            self._regularize_conformity_score(
                self.k_star,
                self.lambda_star,
                conformity_scores,
                self.cutoff
            )
        )
        quantiles_ = _compute_quantiles(
            conformity_scores_regularized,
            alpha_np
        )

        return quantiles_

    def _add_regularization(
        self,
        y_pred_proba_sorted_cumsum: NDArray,
        lambda_: Optional[float] = None,
        k_star: Optional[int] = None,
        prediction_phase: bool = False,
        **kwargs
    ) -> NDArray:
        """
        Add regularization to the sorted cumulative sum of predicted
        probabilities.

        Parameters
        ----------
        y_pred_proba_sorted_cumsum: NDArray of shape (n_samples, n_classes)
            The sorted cumulative sum of predicted probabilities.

        lambda_: float
            The lambda value used in the paper [1].

            By default, "None" but must be set to work.

        k_star: int
            The optimal value of k (called k_reg in the paper [1]).

            By default, "None" but must be set to work.

        prediction_phase: bool, optional
            Whether the function is called during the prediction phase.
            If ``True``, the function will use the values of ``lambda_star``
            and ``k_star`` of the object.

            By default, ``False``.

        **kwargs: dict, optional
            Additional keyword arguments that might be used.
            The current implementation does not use any.

        Returns
        -------
        NDArray
            The adjusted cumulative sum of predicted probabilities after
            applying the regularization technique.
        """
        if prediction_phase:
            lambda_ = cast(float, self.lambda_star)
            k_star = cast(int, self.k_star)
        else:
            lambda_ = cast(float, lambda_)
            k_star = cast(int, lambda_)

        y_pred_proba_sorted_cumsum += lambda_ * np.maximum(
            0,
            np.cumsum(
                np.ones(y_pred_proba_sorted_cumsum.shape), axis=1
            ) - k_star
        )

        return y_pred_proba_sorted_cumsum

    def _compute_v_parameter(
        self,
        y_proba_last_cumsumed: NDArray,
        threshold: NDArray,
        y_pred_proba_last: NDArray,
        prediction_sets: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Compute the V parameters from Angelopoulos+(2020).

        Parameters
        -----------
        y_proba_last_cumsumed: NDArray of shape (n_samples, n_alpha)
            Cumulated score of the last included label.

        threshold: NDArray of shape (n_alpha,) or shape (n_samples_train,)
            Threshold to compare with y_proba_last_cumsum.

        y_pred_proba_last: NDArray of shape (n_samples, 1, n_alpha)
            Last included probability.

        predicition_sets: NDArray of shape (n_samples, n_alpha)
            Prediction sets.

        Returns
        --------
        NDArray of shape (n_samples, n_alpha)
            Vs parameters.
        """
        # compute V parameter from Angelopoulos+(2020)
        L = np.sum(prediction_sets, axis=1)
        v_param = (
            (y_proba_last_cumsumed - threshold.reshape(1, -1)) /
            (
                y_pred_proba_last[:, 0, :] -
                self.lambda_star * np.maximum(0, L - self.k_star) +
                self.lambda_star * (L > self.k_star)
            )
        )
        return v_param
