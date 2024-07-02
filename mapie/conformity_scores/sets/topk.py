from typing import Optional, Union, cast

import numpy as np

from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import (
    check_proba_normalized, get_true_label_position
)
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray
from mapie.utils import compute_quantiles


class TopK(BaseClassificationScore):
    """
    Top-K method-based non-conformity score.

    It is based on the sorted index of the probability of the true label in the
    softmax outputs, on the calibration set. In case two probabilities are
    equal, both are taken, thus, the size of some prediction sets may be
    different from the others.

    References
    ----------
    [1] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan
    and Jitendra Malik.
    "Uncertainty Sets for Image Classifiers using Conformal Prediction."
    International Conference on Learning Representations 2021.

    Parameters
    ----------
    consistency_check: bool, optional
        Whether to check the consistency between the methods
        ``get_estimation_distribution`` and ``get_conformity_scores``.
        If ``True``, the following equality must be verified:
        ``self.get_estimation_distribution(
            y_pred, self.get_conformity_scores(y, y_pred, **kwargs), **kwargs
        ) == y``

        By default ``True``.

    eps: float, optional
        Threshold to consider when checking the consistency between
        ``get_estimation_distribution`` and ``get_conformity_scores``.
        It should be specified if ``consistency_check==True``.

        By default, it is defined by the default precision.

    Attributes
    ----------
    method: str
        Method to choose for prediction interval estimates.
        This attribute is for compatibility with ``MapieClassifier``
        which previously used a string instead of a score class.

        By default, ``top_k`` for Top-K method.

    classes: Optional[ArrayLike]
        Names of the classes.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``get_sets`` method.
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
        method: str = 'top_k',
        classes: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        **kwargs
    ) -> None:
        """
        Set attributes that are not provided by the user.

        Parameters
        ----------
        method: str
            Method to choose for prediction interval estimates.
            Methods available in this class: ``top_k``.

            By default ``top_k`` for Top-K method.

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
        # Casting
        y_enc = cast(NDArray, y_enc)

        # Conformity scores
        # Here we reorder the labels by decreasing probability and get the
        # position of each label from decreasing probability
        conformity_scores = get_true_label_position(y_pred, y_enc)

        return conformity_scores

    def get_estimation_distribution(
        self,
        y_pred: NDArray,
        conformity_scores: NDArray,
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

    def get_sets(
        self,
        X: ArrayLike,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        conformity_scores: NDArray,
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

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
            Prediction sets (Booleans indicate whether classes are included).
        """
        # Checks
        y_pred_proba = estimator.predict(X, agg_scores="mean")
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        y_pred_proba = np.repeat(
            y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
        )

        # Choice of the quantile
        self.quantiles_ = compute_quantiles(conformity_scores, alpha_np)

        # Build prediction sets
        y_pred_proba = y_pred_proba[:, :, 0]
        index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
        y_pred_index_last = np.stack(
            [
                index_sorted[:, quantile]
                for quantile in self.quantiles_
            ], axis=1
        )
        y_pred_proba_last = np.stack(
            [
                np.take_along_axis(
                    y_pred_proba,
                    y_pred_index_last[:, iq].reshape(-1, 1),
                    axis=1
                )
                for iq, _ in enumerate(self.quantiles_)
            ], axis=2
        )
        prediction_sets = np.greater_equal(
            y_pred_proba[:, :, np.newaxis]
            - y_pred_proba_last,
            -EPSILON
        )

        # Just for coverage: do nothing
        self.get_estimation_distribution(y_pred_proba, conformity_scores)

        return prediction_sets
