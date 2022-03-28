from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np
import numpy.ma as ma
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .aggregation_functions import aggregate_all
from .regression import MapieRegressor
from ._typing import ArrayLike, NDArray
from .utils import (
    check_alpha,
    check_alpha_and_n_samples,
)


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
        self.cv_need_agg_function.append("BlockBootstrap")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieTimeSeriesRegressor:
        """
        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.
        """
        self = super().fit(X=X, y=y, sample_weight=sample_weight)
        y_pred, _ = super().predict(X, alpha=0.5, ensemble=True)
        self.conformity_scores_ = np.asarray(y) - y_pred
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> MapieTimeSeriesRegressor:
        """
        Update the ``conformity_scores_`` attribute when data with known labels
        are available.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
            Input labels.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.
        """
        y_pred, _ = self.predict(X, alpha=0.5, ensemble=True)
        new_conformity_scores_ = np.asarray(y) - np.asarray(y_pred)
        new_conformity_scores_ = new_conformity_scores_[
            ~np.isnan(new_conformity_scores_)
        ]

        cut_index = min(
            len(new_conformity_scores_), len(self.conformity_scores_)
        )
        self.conformity_scores_ = np.concatenate(
            [
                self.conformity_scores_[cut_index:],
                new_conformity_scores_,
            ],
            axis=0,
        )
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:

        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha = cast(Optional[NDArray], check_alpha(alpha))
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)
        n = len(self.conformity_scores_)

        if alpha is None:
            return np.array(y_pred)
        else:
            alpha_np = cast(NDArray, alpha)
            check_alpha_and_n_samples(alpha_np, n)
            betas_0 = np.full_like(alpha_np, np.nan, dtype=float)
            for ind, _alpha in enumerate(alpha_np):
                betas = np.linspace(0.0, _alpha, num=n + 1)

                one_alpha_beta = np.quantile(
                    ma.masked_invalid(self.conformity_scores_),
                    1 - _alpha + betas,
                    axis=0,
                    interpolation="higher",
                )  # type: ignore

                beta = np.quantile(
                    ma.masked_invalid(self.conformity_scores_),
                    betas,
                    axis=0,
                    interpolation="lower",
                )  # type: ignore
                betas_0[ind] = betas[np.argmin(one_alpha_beta - beta, axis=0)]

            lower_quantiles = np.quantile(
                ma.masked_invalid(self.conformity_scores_),
                betas_0,
                axis=0,
                interpolation="lower",
            )  # type: ignore
            higher_quantiles = np.quantile(
                ma.masked_invalid(self.conformity_scores_),
                1 - alpha_np + betas_0,
                axis=0,
                interpolation="higher",
            )  # type: ignore

            if self.method in ["naive", "base"] or self.cv == "prefit":
                y_pred_low = np.column_stack(
                    [
                        y_pred[:, np.newaxis] + lower_quantiles[k]
                        for k in range(len(alpha_np))
                    ]
                )
                y_pred_up = np.column_stack(
                    [
                        y_pred[:, np.newaxis] + higher_quantiles[k]
                        for k in range(len(alpha_np))
                    ]
                )
            else:
                y_pred_multi = np.column_stack(
                    [e.predict(X) for e in self.estimators_]
                )

                # At this point, y_pred_multi is of shape
                # (n_samples_test, n_estimators_). The method
                # ``aggregate_with_mask`` fits it to the right size thanks to
                # the shape of k_.

                y_pred_multi = self.aggregate_with_mask(y_pred_multi, self.k_)

                if self.method == "plus":
                    pred = aggregate_all(self.agg_function, y_pred_multi)
                    y_pred_low = np.column_stack(
                        [
                            pred + lower_quantiles[k]
                            for k in range(len(alpha_np))
                        ]
                    )
                    y_pred_up = np.column_stack(
                        [
                            pred + higher_quantiles[k]
                            for k in range(len(alpha_np))
                        ]
                    )

                if self.method == "minmax":
                    lower_bounds = np.min(y_pred_multi, axis=1, keepdims=True)
                    upper_bounds = np.max(y_pred_multi, axis=1, keepdims=True)
                    y_pred_low = np.column_stack(
                        [
                            lower_bounds + lower_quantiles[k]
                            for k in range(len(alpha_np))
                        ]
                    )
                    y_pred_up = np.column_stack(
                        [
                            upper_bounds + higher_quantiles[k]
                            for k in range(len(alpha_np))
                        ]
                    )
                if ensemble:
                    y_pred = aggregate_all(self.agg_function, y_pred_multi)
            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
