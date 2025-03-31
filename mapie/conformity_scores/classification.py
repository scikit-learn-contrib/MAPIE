from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np

from mapie.conformity_scores.interface import BaseConformityScore
from mapie.estimator.classifier import EnsembleClassifier

from numpy.typing import ArrayLike, NDArray


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

    def set_external_attributes(
        self,
        *,
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

    @abstractmethod
    def get_predictions(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Abstract method to get predictions from an EnsembleClassifier.

        This method should be implemented by any subclass of the current class.

        Parameters
        -----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between ``0`` and ``1``, represents the
            uncertainty of the confidence set.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        Returns
        --------
        NDArray
            Array of predictions.
        """

    @abstractmethod
    def get_conformity_score_quantiles(
        self,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Abstract method to get quantiles of the conformity scores.

        This method should be implemented by any subclass of the current class.

        Parameters
        -----------
        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence set.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """

    @abstractmethod
    def get_prediction_sets(
        self,
        y_pred_proba: NDArray,
        conformity_scores: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        **kwargs
    ) -> NDArray:
        """
        Abstract method to generate prediction sets based on the probability
        predictions, the conformity scores and the uncertainty level.

        This method should be implemented by any subclass of the current class.

        Parameters
        -----------
        y_pred_proba: NDArray of shape (n_samples, n_classes)
            Target prediction.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores for each sample.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence set.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        Returns
        --------
        NDArray
            Array of quantiles with respect to alpha_np.
        """

    def get_sets(
        self,
        X: NDArray,
        alpha_np: NDArray,
        estimator: EnsembleClassifier,
        conformity_scores: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Compute classes of the prediction sets from the observed values,
        the estimator of type ``EnsembleClassifier`` and the conformity scores.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        alpha_np: NDArray of shape (n_alpha,)
            NDArray of floats between 0 and 1, representing the uncertainty
            of the confidence set.

        estimator: EnsembleClassifier
            Estimator that is fitted to predict y from X.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores.

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
            Prediction sets (Booleans indicate whether classes are included).
        """
        # Predict probabilities
        y_pred_proba = self.get_predictions(
            X, alpha_np, estimator, **kwargs
        )

        # Choice of the quantile
        self.quantiles_ = self.get_conformity_score_quantiles(
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
        the target confidence set.

        Parameters
        -----------
        X: NDArray of shape (n_samples,)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence set to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns
        --------
        The output structure depend on the ``get_sets`` method.
            The prediction sets for each sample and each alpha level.
        """
        return self.get_sets(X=X, alpha_np=alpha_np, **kwargs)
