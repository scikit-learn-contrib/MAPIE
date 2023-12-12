from __future__ import annotations

from typing import Optional, Union, cast

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted

from mapie._typing import ArrayLike, NDArray
from .regression import MapieRegressor


class MapieTimeSeriesRegressor(MapieRegressor):
    """
    Prediction intervals with out-of-fold residuals for time series.

    This class implements the EnbPI strategy for estimating
    prediction intervals on single-output time series. The only valid
    ``method`` is ``"enbpi"``.

    Actually, EnbPI only corresponds to ``MapieTimeSeriesRegressor`` if the
    ``cv`` argument is of type ``BlockBootstrap``.

    References
    ----------
    Chen Xu, and Yao Xie.
    "Conformal prediction for dynamic time-series."
    https://arxiv.org/abs/2010.09107
    """

    cv_need_agg_function_ = MapieRegressor.cv_need_agg_function_ \
        + ["BlockBootstrap"]
    valid_methods_ = ["enbpi"]

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "enbpi",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
            random_state=random_state
        )

    def _relative_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> NDArray:
        """
        Compute the conformity scores on a data set.

        Parameters
        ----------
            X : ArrayLike of shape (n_samples, n_features)
            Input data.

            y : ArrayLike of shape (n_samples,)
                Input labels.

        Returns
        -------
            The conformity scores corresponding to the input data set.
        """
        y_pred, _ = super().predict(X, alpha=0.5, ensemble=True)
        return np.asarray(y) - np.asarray(y_pred)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieTimeSeriesRegressor:
        """
        Compared to the method ``fit`` of ``MapieRegressor``, the ``fit``
        method of ``MapieTimeSeriesRegressor`` computes the
        ``conformity_scores_`` with relative values.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no conformity scores.
            If weights are non-uniform,
            conformity scores are still uniformly weighted.

            By default ``None``.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.
        """
        self = super().fit(X=X, y=y, sample_weight=sample_weight)
        self.conformity_scores_ = self._relative_conformity_scores(X, y)
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> MapieTimeSeriesRegressor:
        """
        Update the ``conformity_scores_`` attribute when new data with known
        labels are available.
        Note: Don't use ``partial_fit`` with samples of the training set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data.

        y: ArrayLike of shape (n_samples_test,)
            Input labels.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.

        Raises
        ------
        ValueError
            If the length of ``y`` is greater than
            the length of the training set.
        """
        check_is_fitted(self, self.fit_attributes)
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        n = len(self.conformity_scores_)
        if len(X) > n:
            raise ValueError(
                "The number of observations to update is higher than the"
                "number of training instances."
            )
        new_conformity_scores_ = self._relative_conformity_scores(X, y)
        self.conformity_scores_ = np.roll(
            self.conformity_scores_, -len(new_conformity_scores_)
        )
        self.conformity_scores_[
            -len(new_conformity_scores_):
        ] = new_conformity_scores_
        return self

    def _more_tags(self):
        return {
            "_xfail_checks":
            {
                "check_estimators_partial_fit_n_features":
                "partial_fit can only be called on fitted models"
            }
        }
