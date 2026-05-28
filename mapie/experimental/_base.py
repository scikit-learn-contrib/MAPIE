from abc import abstractmethod

from numpy.typing import NDArray

from mapie._machine_precision import EPSILON
from mapie.conformity_scores.regression import BaseRegressionScore


class BaseFitRegressionScore(BaseRegressionScore):
    """
    Base conformity score class for regression task, for scores requiring fitting.

    This class should not be used directly. Use derived classes instead.

    Parameters
    ----------
    sym: bool
        Whether to consider the conformity score as symmetrical or not.

    consistency_check: bool, optional
        Whether to check the consistency between the methods
        ``get_estimation_distribution`` and ``get_conformity_scores``.
        If ``True``, the following equality must be verified::

            y == self.get_estimation_distribution(
                y_pred,
                self.get_conformity_scores(y, y_pred, **kwargs),
                **kwargs)

        By default ``True``.

    eps: float, optional
        Threshold to consider when checking the consistency
        between ``get_estimation_distribution`` and ``get_conformity_scores``.
        It should be specified if ``consistency_check==True``.

        By default, it is defined by the default precision.
    """

    def __init__(
        self,
        sym: bool,
        consistency_check: bool = True,
        eps: float = float(EPSILON),
    ) -> None:
        super().__init__(sym, consistency_check, eps)
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray, **kwargs) -> "BaseFitRegressionScore":
        """
        Placeholder for ``fit``.
        Subclasses should implement this method and set `self.is_fitted = True`.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_features)
            Observed feature values.

        y: NDArray of shape (n_samples,)
            Observed target values.

        Returns
        -------
        self: BaseFitRegressionScore
            Fitted conformity score.
        """
        self.is_fitted = True
        return self
