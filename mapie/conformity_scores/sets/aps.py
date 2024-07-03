from typing import Optional, cast

import numpy as np
from sklearn.dummy import check_random_state

from mapie.conformity_scores.sets.naive import Naive
from mapie.conformity_scores.sets.utils import get_true_label_cumsum_proba
from mapie.estimator.classifier import EnsembleClassifier

from mapie._typing import NDArray
from mapie.utils import compute_quantiles


class APS(Naive):
    """TODO:
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

    Attributes
    ----------
    method: str
        Method to choose for prediction interval estimates.
        This attribute is for compatibility with ``MapieClassifier``
        which previously used a string instead of a score class.
        Methods available in this class: ``aps``, ``raps`` and ``naive``.

        By default, ``aps`` for APS method.

    classes: Optional[ArrayLike]
        Names of the classes.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``get_sets`` method.
    """

    def __init__(self) -> None:
        super().__init__()

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
        classes = cast(NDArray, self.classes)

        # Conformity scores
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

    def get_conformity_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
        **kwargs
    ) -> NDArray:
        """
        TODO: Compute the quantiles.
        """
        n = len(conformity_scores)

        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            quantiles_ = compute_quantiles(conformity_scores, alpha_np)
        else:
            quantiles_ = (n + 1) * (1 - alpha_np)

        return quantiles_
