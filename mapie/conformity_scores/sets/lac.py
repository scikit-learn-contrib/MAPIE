from typing import Optional, cast, Union

import numpy as np

from mapie.conformity_scores.classification import BaseClassificationScore
from sklearn.model_selection import BaseCrossValidator

from mapie._machine_precision import EPSILON
from numpy.typing import NDArray
from mapie.utils import _compute_quantiles


class LACConformityScore(BaseClassificationScore):
    """
    Least Ambiguous set-valued Classifier (LAC) method-based
    non conformity score (also formerly called ``"score"``).

    It is based on the scores (i.e. 1 minus the softmax score of the true
    label) on the conformalization set.

    References
    ----------
    [1] Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.",
    Journal of the American Statistical Association, 114, 2019.

    Attributes
    ----------
    classes: Optional[ArrayLike]
        Names of the classes.

    random_state: Optional[Union[int, np.random.RandomState]]
        Pseudo random number generator state.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``get_sets`` method.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_conformity_scores(
        self, y: NDArray, y_pred: NDArray, y_enc: Optional[NDArray] = None, **kwargs
    ) -> NDArray:
        """
        Get the conformity score.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values (not used here).

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
        conformity_scores = np.take_along_axis(1 - y_pred, y_enc.reshape(-1, 1), axis=1)

        return conformity_scores

    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        y_pred_proba: NDArray,
        cv: Optional[Union[int, str, BaseCrossValidator]],
        agg_scores: Optional[str] = "mean",
        **kwargs,
    ) -> NDArray:
        """
        Just processes the passed y_pred_proba.

        Parameters
        -----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values (not used since predictions are passed).

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence interval.

        y_pred_proba: NDArray
            Predicted probabilities from the estimator.

        cv: Optional[Union[int, str, BaseCrossValidator]]
            Cross-validation strategy used by the estimator (not used here).

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of predictions.
        """
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )

        return y_pred_proba

    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        cv: Optional[Union[int, str, BaseCrossValidator]],
        agg_scores: Optional[str] = "mean",
        **kwargs,
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

        cv: Optional[Union[int, str, BaseCrossValidator]]
            Cross-validation strategy used by the estimator.

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        n = len(conformity_scores)

        if cv == "prefit" or agg_scores in ["mean"]:
            quantiles_ = _compute_quantiles(conformity_scores, alpha_np)
        else:
            quantiles_ = (n + 1) * (1 - alpha_np)

        return quantiles_

    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        cv: Optional[Union[int, str, BaseCrossValidator]],
        agg_scores: Optional[str] = "mean",
        **kwargs,
    ) -> NDArray:
        """
        Generate prediction sets based on the probability predictions,
        the conformity scores and the uncertainty level.

        Parameters
        -----------
        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Target prediction.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence interval.

        cv: Optional[Union[int, str, BaseCrossValidator]]
            Cross-validation strategy used by the estimator.

        agg_scores: Optional[str]
            Method to aggregate the scores from the base estimators.
            If "mean", the scores are averaged. If "crossval", the scores are
            obtained from cross-validation.

            By default ``"mean"``.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """
        n = len(conformity_scores)

        if (cv == "prefit") or (agg_scores == "mean"):
            prediction_sets = np.less_equal(
                (1 - y_pred_proba) - self.quantiles_, EPSILON
            )
        else:
            y_pred_included = np.less_equal(
                (1 - y_pred_proba) - conformity_scores.ravel(), EPSILON
            ).sum(axis=2)
            prediction_sets = np.stack(
                [
                    np.greater_equal(y_pred_included - _alpha * (n - 1), -EPSILON)
                    for _alpha in alpha_np
                ],
                axis=2,
            )

        return cast(NDArray, prediction_sets)
