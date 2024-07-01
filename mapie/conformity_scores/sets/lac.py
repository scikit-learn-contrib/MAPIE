from typing import Optional, Union, cast

import numpy as np

from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.sets.utils import check_proba_normalized
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray
from mapie.utils import compute_quantiles


class LAC(BaseClassificationScore):
    """
    Least Ambiguous set-valued Classifier (LAC) method-based
    non conformity score (also formerly called ``"score"``).

    It is based on the the scores (i.e. 1 minus the softmax score of the true
    label) on the calibration set.

    References
    ----------
    [1] Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.",
    Journal of the American Statistical Association, 114, 2019.
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
        method: str = 'lac',
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
            Methods available in this class: ``lac``.

            By default ``lac`` for LAC method.

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
        y_pred = cast(NDArray, y_pred)
        y_enc = cast(NDArray, y_enc)

        # Conformity scores
        conformity_scores = np.take_along_axis(
            1 - y_pred, y_enc.reshape(-1, 1), axis=1
        )
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

    def get_sets(
        self,
        X: ArrayLike,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        conformity_scores: NDArray,
        agg_scores: Optional[str] = "mean",
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
        n = len(conformity_scores)

        y_pred_proba = estimator.predict(X, agg_scores)
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        if agg_scores != "crossval":
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )

        # Choice of the quantile
        if (estimator.cv == "prefit") or (agg_scores in ["mean"]):
            self.quantiles_ = compute_quantiles(
                conformity_scores,
                alpha_np
            )
        else:
            self.quantiles_ = (n + 1) * (1 - alpha_np)

        # Build prediction sets
        if (estimator.cv == "prefit") or (agg_scores == "mean"):
            prediction_sets = np.greater_equal(
                y_pred_proba - (1 - self.quantiles_), -EPSILON
            )
        else:
            y_pred_included = np.less_equal(
                (1 - y_pred_proba) - conformity_scores.ravel(),
                EPSILON
            ).sum(axis=2)
            prediction_sets = np.stack(
                [
                    np.greater_equal(
                        y_pred_included - _alpha * (n - 1), -EPSILON
                    )
                    for _alpha in alpha_np
                ], axis=2
            )

        # Just for coverage: do nothing
        self.get_estimation_distribution(y_pred_proba, conformity_scores)

        return prediction_sets
