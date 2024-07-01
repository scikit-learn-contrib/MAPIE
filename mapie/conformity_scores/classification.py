from abc import ABCMeta, abstractmethod

from mapie.conformity_scores.interface import BaseConformityScore
from mapie.estimator.classifier import EnsembleClassifier

from mapie._machine_precision import EPSILON
from mapie._typing import NDArray


class BaseClassificationScore(BaseConformityScore, metaclass=ABCMeta):
    """
    Base conformity score class for classification task.

    This class should not be used directly. Use derived classes instead.

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
    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``conformity_scores_`` and alpha values.
    """

    def __init__(
        self,
        consistency_check: bool = True,
        eps: float = float(EPSILON),
    ):
        super().__init__(consistency_check=consistency_check, eps=eps)

    @abstractmethod
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
