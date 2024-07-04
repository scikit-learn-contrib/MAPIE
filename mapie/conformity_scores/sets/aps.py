from typing import Optional, cast

import numpy as np
from sklearn.dummy import check_random_state

from mapie.conformity_scores.sets.naive import Naive
from mapie.conformity_scores.sets.utils import get_true_label_cumsum_proba
from mapie.estimator.classifier import EnsembleClassifier

from mapie._typing import NDArray
from mapie.utils import compute_quantiles


class APS(Naive):
    """
    Adaptive Prediction Sets (APS) method-based non-conformity score.
    It is based on the sum of the softmax outputs of the labels until the true
    label is reached, on the calibration set. See [1] for more details.

    References
    ----------
    [1] Yaniv Romano, Matteo Sesia and Emmanuel J. Candès.
    "Classification with Valid and Adaptive Coverage."
    NeurIPS 202 (spotlight) 2020.

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

    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        agg_scores: Optional[str] = "mean",
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

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns:
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        n = len(conformity_scores)

        if estimator.cv == "prefit" or agg_scores in ["mean"]:
            quantiles_ = compute_quantiles(conformity_scores, alpha_np)
        else:
            quantiles_ = (n + 1) * (1 - alpha_np)

        return quantiles_
