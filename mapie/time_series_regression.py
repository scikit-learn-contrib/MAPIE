from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from .regression import MapieRegressor
from ._typing import ArrayLike


class MapieTimeSeriesRegressor(MapieRegressor):
    """
    Prediction interval with out-of-fold residuals for time series.

    This class implements the EnbPI strategy and some variations
    for estimating prediction intervals on single-output time series.
    It is ``MapieRegressor`` with one more method ``partial_fit``.
    Actually, EnbPI only corresponds to MapieRegressor if the ``cv`` argument
    if of type ``Subsample`` (Jackknife+-after-Bootstrap method). Moreover, for
    the moment we consider the absolute values of the residuals of the model,
    and consequently the prediction intervals are symmetryc. Moreover we did
    not implement the PI's optimization to the oracle interval yet. It is still
    a first step before implementing the actual EnbPI.

    References
    ----------
    Chen Xu, and Yao Xie.
    "Conformal prediction for dynamic time-series."
    """

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "plus",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
    ) -> None:
        super().__init__(estimator, method, cv, n_jobs, agg_function, verbose)

    def partial_fit(
        self, X: ArrayLike, y: ArrayLike, ensemble: bool = True
    ) -> MapieTimeSeriesRegressor:
        """
        Update the ``residuals_`` attribute when data with known labels are
        available.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
            Input labels.

        ensemble : bool
            Boolean corresponding to the ``ensemble`` argument of ``predict``
            method, determining whether the predictions computed to determine
            the new ``residuals_``  are ensembled or not.
            If False, predictions are those of the model trained on the whole
            training set.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.
        """
        y_pred, _ = self.predict(X, alpha=0.5, ensemble=ensemble)
        new_residuals = np.abs(y - y_pred)

        cut_index = min(len(new_residuals), len(self.residuals_))
        self.residuals_ = np.concatenate(
            [self.residuals_[cut_index:], new_residuals], axis=0
        )
        return self
