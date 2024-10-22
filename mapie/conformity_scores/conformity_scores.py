from abc import ABCMeta, abstractmethod

from sklearn.utils import deprecated

from mapie.conformity_scores.regression import BaseConformityScore
from mapie._machine_precision import EPSILON
from mapie._typing import NDArray


@deprecated(
    "WARNING: Deprecated path to import ConformityScore. "
    "Please prefer the new path: "
    "[from mapie.conformity_scores import BaseRegressionScore]."
)
class ConformityScore(BaseConformityScore, metaclass=ABCMeta):
    """
    Base conformity score class for regression task.

    This class should not be used directly. Use derived classes instead.

    Parameters
    ----------
    sym: bool
        Whether to consider the conformity score as symmetrical or not.

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
    """

    def __init__(
        self,
        sym: bool,
        consistency_check: bool = True,
        eps: float = float(EPSILON),
    ):
        super().__init__()
        self.sym = sym
        self.consistency_check = consistency_check
        self.eps = eps

    @abstractmethod
    def get_signed_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Placeholder for ``get_conformity_scores``.
        Subclasses should implement this method!

        Compute the sample conformity scores given the predicted and
        observed targets.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Signed conformity scores.
        """

    @abstractmethod
    def get_conformity_scores(
        self,
        y: NDArray,
        y_pred: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Placeholder for ``get_conformity_scores``.
        Subclasses should implement this method!

        Compute the sample conformity scores given the predicted and
        observed targets.

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

    @abstractmethod
    def get_estimation_distribution(
        self,
        y_pred: NDArray,
        conformity_scores: NDArray,
        **kwargs
    ) -> NDArray:
        """
        Placeholder for ``get_estimation_distribution``.
        Subclasses should implement this method!

        Compute samples of the estimation distribution given the predicted
        targets and the conformity scores.

        Parameters
        ----------
        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        conformity_scores: NDArray of shape (n_samples,)
            Conformity scores.

        Returns
        -------
        NDArray of shape (n_samples,)
            Observed values.
        """

    @abstractmethod
    def predict_set(
        self,
        X: NDArray,
        alpha_np: NDArray,
        **kwargs
    ):
        """
        Compute the prediction sets on new samples based on the uncertainty of
        the target confidence set.

        Parameters:
        -----------
        X: NDArray of shape (n_samples,)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence set to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns:
        --------
        The output structure depend on the subclass.
            The prediction sets for each sample and each alpha level.
        """
