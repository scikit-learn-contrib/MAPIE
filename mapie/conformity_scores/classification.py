from abc import ABCMeta, abstractmethod

from mapie.conformity_scores.interface import BaseConformityScore
from mapie.estimator.classifier import EnsembleClassifier

from mapie._typing import NDArray


class BaseClassificationScore(BaseConformityScore, metaclass=ABCMeta):
    """
    Base conformity score class for classification task.

    This class should not be used directly. Use derived classes instead.

    Attributes
    ----------
    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``get_sets`` method.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    def get_sets(
        self,
        X: NDArray,
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
        ()

        # Predict probabilities
        y_pred_proba = self.get_predictions(
            X, alpha_np, estimator, **kwargs
        )

        # Choice of the quantile
        self.quantiles_ = self.get_conformity_quantiles(
            conformity_scores, alpha_np, estimator, **kwargs
        )

        # Build prediction sets
        prediction_sets = self.get_prediction_sets(
            y_pred_proba, conformity_scores, alpha_np, estimator, **kwargs
        )

        return prediction_sets

    def predict_set(
        self,
        X: NDArray,
        alpha_np: NDArray,
        **kwargs
    ):
        """
        Compute the prediction sets on new samples based on the uncertainty of
        the target confidence interval.

        Parameters:
        -----------
        X: NDArray of shape (n_samples, ...)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence interval to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns:
        --------
        The output strcture depend on the ``get_sets`` method.
            The prediction sets for each sample and each alpha level.
        """
        return self.get_sets(X=X, alpha_np=alpha_np, **kwargs)
