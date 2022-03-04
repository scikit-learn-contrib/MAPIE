from __future__ import annotations
from argparse import ArgumentDefaultsHelpFormatter

from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np
import numpy.ma as ma
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .aggregation_functions import aggregate_all
from .regression import MapieRegressor
from .subsample import Subsample, BlockBootstrap
from ._typing import ArrayLike
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

    cv_need_agg_function = [BlockBootstrap, Subsample]
    valid_methods_ = ["plus"]
    valid_agg_functions_ = [None, "median", "mean"]
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "residuals_",
        "n_features_in_",
        "n_samples_val_",
    ]

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
        y_pred = super().predict(X=X)
        self.residuals_ = y.values - y_pred
        return self

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
        new_residuals = y - y_pred

        cut_index = min(
            len(new_residuals[~np.isnan(new_residuals)]), len(self.residuals_)
        )
        self.residuals_ = np.concatenate(
            [
                self.residuals_[cut_index:],
                new_residuals[~np.isnan(new_residuals)],
            ],
            axis=0,
        )
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:

        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha_ = check_alpha(alpha)
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)
        else:
            alpha_ = cast(ArrayLike, alpha_)
            check_alpha_and_n_samples(alpha_, self.residuals_.shape[0])

            betas_0 = np.full_like(alpha_, np.nan, dtype=float)

            for ind, _alpha in enumerate(alpha_):
                betas = np.linspace(0.0, _alpha, num=len(self.residuals_) + 2)

                one_alpha_beta = np.quantile(
                    self.residuals_,
                    1 - _alpha + betas,
                    axis=0,
                    interpolation="higher",
                )

                beta = np.quantile(
                    self.residuals_,
                    betas,
                    axis=0,
                    interpolation="lower",
                )
                betas_0[ind] = betas[np.argmin(one_alpha_beta - beta, axis=0)]

            lower_quantiles = np.quantile(
                self.residuals_,
                betas_0,
                axis=0,
                interpolation="lower",
            )
            higher_quantiles = np.quantile(
                self.residuals_,
                1 - _alpha + betas_0,
                axis=0,
                interpolation="higher",
            )

            if self.method in ["naive", "base"] or self.cv == "prefit":
                y_pred_low = np.column_stack(
                    [
                        y_pred[:, np.newaxis] + lower_quantiles[k]
                        for k in range(len(alpha_))
                    ]
                )
                y_pred_up = np.column_stack(
                    [
                        y_pred[:, np.newaxis] + higher_quantiles[k]
                        for k in range(len(alpha_))
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
                        [pred + lower_quantiles[k] for k in range(len(alpha_))]
                    )
                    y_pred_up = np.column_stack(
                        [
                            pred + higher_quantiles[k]
                            for k in range(len(alpha_))
                        ]
                    )

                if self.method == "minmax":
                    lower_bounds = np.min(y_pred_multi, axis=1, keepdims=True)
                    upper_bounds = np.max(y_pred_multi, axis=1, keepdims=True)
                    y_pred_low = np.column_stack(
                        [
                            lower_bounds + lower_quantiles[k]
                            for k in range(len(alpha_))
                        ]
                    )
                    y_pred_up = np.column_stack(
                        [
                            upper_bounds + higher_quantiles[k]
                            for k in range(len(alpha_))
                        ]
                    )
                if ensemble:
                    y_pred = aggregate_all(self.agg_function, y_pred_multi)
            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)

    def predict2(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha_ = check_alpha(alpha)
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)
        else:
            alpha_ = cast(ArrayLike, alpha_)
            check_alpha_and_n_samples(alpha_, self.residuals_.shape[0])
            if self.method in ["naive", "base"] or self.cv == "prefit":
                quantile = np.quantile(
                    self.residuals_, 1 - alpha_, interpolation="higher"
                )
                y_pred_low = y_pred[:, np.newaxis] - quantile
                y_pred_up = y_pred[:, np.newaxis] + quantile
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
                    lower_bounds = y_pred_multi + self.residuals_
                    upper_bounds = y_pred_multi + self.residuals_
                if self.method == "minmax":
                    lower_bounds = np.min(y_pred_multi, axis=1, keepdims=True)
                    upper_bounds = np.max(y_pred_multi, axis=1, keepdims=True)
                    lower_bounds = lower_bounds + self.residuals_
                    upper_bounds = upper_bounds + self.residuals_

                y_pred_low = np.column_stack(
                    [
                        np.quantile(
                            ma.masked_invalid(lower_bounds),
                            _alpha,
                            axis=1,
                            interpolation="lower",
                        )
                        for _alpha in alpha_
                    ]
                )

                y_pred_up = np.column_stack(
                    [
                        np.quantile(
                            ma.masked_invalid(upper_bounds),
                            1 - _alpha,
                            axis=1,
                            interpolation="higher",
                        )
                        for _alpha in alpha_
                    ]
                )
                if ensemble:
                    y_pred = aggregate_all(self.agg_function, y_pred_multi)
            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)

    def predict3(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha_ = check_alpha(alpha)
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)
        else:
            alpha_ = cast(ArrayLike, alpha_)
            check_alpha_and_n_samples(alpha_, self.residuals_.shape[0])

            y_pred_low = []
            y_pred_up = []

            for _alpha in alpha_:
                betas = np.linspace(0.0, _alpha, num=len(self.residuals_) + 2)

                if self.method in ["naive", "base"] or self.cv == "prefit":
                    one_alpha_beta = np.quantile(
                        self.residuals_,
                        1 - _alpha + betas,
                        axis=0,
                        interpolation="higher",
                    )
                    beta = np.quantile(
                        self.residuals_,
                        betas,
                        axis=0,
                        interpolation="lower",
                    )

                    beta_0 = betas[np.argmin(one_alpha_beta - beta, axis=0)]

                    lower_quantiles = np.quantile(
                        self.residuals_,
                        beta_0,
                        axis=0,
                        interpolation="lower",
                    )
                    higher_quantiles = np.quantile(
                        self.residuals_,
                        1 - _alpha + beta_0,
                        axis=0,
                        interpolation="higher",
                    )
                    y_pred_low.append(y_pred[:, np.newaxis] + lower_quantiles)
                    y_pred_up.append(y_pred[:, np.newaxis] + higher_quantiles)
                else:
                    y_pred_multi = np.column_stack(
                        [e.predict(X) for e in self.estimators_]
                    )

                    # At this point, y_pred_multi is of shape
                    # (n_samples_test, n_estimators_). The method
                    # ``aggregate_with_mask`` fits it to the right size thanks to
                    # the shape of k_.

                    y_pred_multi = self.aggregate_with_mask(
                        y_pred_multi, self.k_
                    )

                    if self.method == "plus":
                        lower_bounds = y_pred_multi + self.residuals_
                        upper_bounds = y_pred_multi + self.residuals_

                    if self.method == "minmax":
                        lower_bounds = np.min(
                            y_pred_multi, axis=1, keepdims=True
                        )
                        upper_bounds = np.max(
                            y_pred_multi, axis=1, keepdims=True
                        )
                        lower_bounds = lower_bounds + self.residuals_
                        upper_bounds = upper_bounds + self.residuals_

                    one_alpha_beta = np.quantile(
                        upper_bounds,
                        1 - _alpha + betas,
                        axis=1,
                        interpolation="higher",
                    )

                    beta = np.quantile(
                        lower_bounds,
                        betas,
                        axis=1,
                        interpolation="lower",
                    )

                    betas_0 = betas[np.argmin(one_alpha_beta - beta, axis=0)]

                    lower_quantiles = np.empty((len(betas_0),))
                    upper_quantiles = np.empty((len(betas_0),))

                    for ind, beta_0 in enumerate(betas_0):
                        lower_quantiles[ind] = np.quantile(
                            lower_bounds[ind, :],
                            beta_0,
                            axis=0,
                            interpolation="lower",
                        )
                        upper_quantiles[ind] = np.quantile(
                            upper_bounds[ind, :],
                            1 - _alpha + beta_0,
                            axis=0,
                            interpolation="higher",
                        )
                    y_pred_low.append(lower_quantiles)
                    y_pred_up.append(upper_quantiles)

                y_pred_low = np.column_stack(y_pred_low)
                y_pred_up = np.column_stack(y_pred_up)

                print(y_pred_low.shape)
                print(y_pred_up.shape)
                if ensemble:
                    y_pred = aggregate_all(self.agg_function, y_pred_multi)
            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
