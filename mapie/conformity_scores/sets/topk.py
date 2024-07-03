from typing import Optional, Union, cast

import numpy as np

from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import (
    check_proba_normalized, get_true_label_position
)
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import NDArray
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

    def __init__(self) -> None:
        super().__init__()

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

    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        TODO: Compute the predictions.
        """
        y_pred_proba = estimator.predict(X, agg_scores="mean")
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
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
        TODO: Compute the quantiles.
        """
        return compute_quantiles(conformity_scores, alpha_np)

    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ):
        """
        TODO: Compute the prediction sets.
        """
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

        return prediction_sets
