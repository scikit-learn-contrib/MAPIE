from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._compatibility import np_nanquantile
from ._typing import ArrayLike, NDArray
from .aggregation_functions import aggregate_all
from .regression import MapieRegressor
from .utils import (
    check_alpha,
    check_alpha_and_n_samples,
)


class MapieTimeSeriesRegressor(MapieRegressor):
    """
    Prediction intervals with out-of-fold residuals for time series.

    This class implements the EnbPI strategy and some variants for estimating
    prediction intervals on single-output time series.

    Actually, EnbPI only corresponds to ``MapieTimeSeriesRegressor`` if the
    ``cv`` argument is of type ``BlockBootstrap`` and ``method`` is "enbpi".

    References
    ----------
    Chen Xu, and Yao Xie.
    "Conformal prediction for dynamic time-series."
    https://arxiv.org/abs/2010.09107
    """

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "enbpi",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
    ) -> None:
        super().__init__(estimator, method, cv, n_jobs, agg_function, verbose)
        self.cv_need_agg_function.append("BlockBootstrap")
        self.valid_methods_.append("enbpi")
        self.plus_like_method.append("enbpi")

    def _relative_conformity_scores(
        self,
        X: NDArray,
        y: NDArray,
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
        y_pred, _ = self.root_predict(X, alpha=0.5, ensemble=True)
        return np.asarray(y) - np.asarray(y_pred)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieTimeSeriesRegressor:
        """
        Compare to the method ``fit`` of ``MapieRegressor``, the ``fit`` method
        of ``MapieTimeSeriesRegressor`` computes the ``conformity_scores_``
        with relative values.

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
        Update the ``conformity_scores_`` and ``k_`` attributes when new data
        with known labels are available.
        Note: Don't use ``partial_fit`` with samples of the training set.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples_test, n_features)
            Input data.

        y : ArrayLike of shape (n_samples_test,)
            Input labels.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.
        """
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        if len(X) > len(self.conformity_scores_):
            raise ValueError("You try to update more residuals than tere are!")
        new_conformity_scores_ = self._relative_conformity_scores(X, y)
        self.conformity_scores_ = np.concatenate(
            [
                self.conformity_scores_[-len(new_conformity_scores_):],
                new_conformity_scores_,
            ]
        )
        self.k_[:, -len(new_conformity_scores_)] = 1.0
        return self

    def _beta_optimize(
        self,
        alpha: Union[float, NDArray],
        upper_bounds: NDArray,
        lower_bounds: NDArray,
        beta_optimize: bool = False,
    ) -> NDArray:
        """
        ``_beta_optimize`` offers to minimize the width of the PIs, for a given
        difference of quantiles.

        Parameters
        ----------
            alpha: Optional[NDArray]
                The quantiles to compute.
            upper_bounds: NDArray
                The array of upper values.
            lower_bounds: NDArray
                The array of lower values.
            optimize: bool
                Whether to optimize or not. If ``False``, betas are the half of
                alphas.

        Returns
        -------
        NDArray
            Array of betas minimizing the differences
            ``(1-alpa+beta)-quantile - beta-quantile``.
        """
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError(
                "Lower and upper bounds arrays should have the same shape."
            )
        alpha = cast(NDArray, alpha)
        betas_0 = np.full(
            shape=(len(alpha), len(lower_bounds)),
            fill_value=np.nan,
            dtype=float,
        )
        if not beta_optimize:
            for ind_alpha, _alpha in enumerate(alpha):
                betas_0[ind_alpha, :] = _alpha / 2.0
            return betas_0

        for ind_alpha, _alpha in enumerate(alpha):
            betas = np.linspace(
                _alpha / (len(lower_bounds) + 1),
                _alpha,
                num=len(lower_bounds),
                endpoint=True,
            )
            one_alpha_beta = np_nanquantile(
                upper_bounds,
                1 - _alpha + betas,
                axis=1,
                method="higher",
            )  # type: ignore
            beta = np_nanquantile(
                lower_bounds,
                betas,
                axis=1,
                method="lower",
            )  # type: ignore
            betas_0[ind_alpha, :] = betas[
                np.argmin(one_alpha_beta - beta, axis=0)
            ]

        return betas_0

    def _pred_multi(self, X: ArrayLike) -> NDArray:
        """
        Return a prediction per train sample for each test sample, by
        aggregation with matrix  ``k_``.

        Parameters
        ----------
            X: NDArray of shape (n_samples_test, n_features)
                Input data

        Returns
        -------
            NDArray of shape (n_samples_test, n_samples_train)
        """
        y_pred_multi = np.column_stack(
            [e.predict(X) for e in self.estimators_]
        )
        # At this point, y_pred_multi is of shape
        # (n_samples_test, n_estimators_). The method
        # ``aggregate_with_mask`` fits it to the right size
        # thanks to the shape of k_.

        y_pred_multi = self.aggregate_with_mask(y_pred_multi, self.k_)
        return y_pred_multi

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        beta_optimize: bool = True,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Correspond to the ``MapieRegressor``'s one with the
        method ``'plus'``. In case  ``method`` is ``'enbpi'``, predictions
        correspond to 'Conformal prediction for dynamic time-series'. The
        method ``'plus'`` is slower because of PI-wise optimization. However,
        you can choose not to optimize the width of the PIs by setting
        ``beta_optimize`` to ``False``.

        Parameters
        ----------
        beta_optimize: bool
            Whether to optimize the PIs' width or not.

        """

        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha = cast(Optional[NDArray], check_alpha(alpha))
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)
        else:
            alpha_np = cast(NDArray, alpha)
            check_alpha_and_n_samples(alpha_np, len(self.conformity_scores_))

            if (self.method in ["base", "enbpi", "minmax", "naive"]) or (
                self.cv == "prefit"
            ):
                betas_0 = self._beta_optimize(
                    alpha=alpha_np,
                    lower_bounds=self.conformity_scores_.reshape(1, -1),
                    upper_bounds=self.conformity_scores_.reshape(1, -1),
                    beta_optimize=beta_optimize,
                )
                lower_quantiles = np_nanquantile(
                    self.conformity_scores_,
                    betas_0[:, 0],
                    axis=0,
                    method="lower",
                ).T  # type: ignore
                higher_quantiles = np_nanquantile(
                    self.conformity_scores_,
                    1 - alpha_np + betas_0[:, 0],
                    axis=0,
                    method="higher",
                ).T  # type: ignore

                if (self.method in ["naive", "base"]) or (self.cv == "prefit"):
                    y_pred_low = y_pred[:, np.newaxis] + lower_quantiles
                    y_pred_up = y_pred[:, np.newaxis] + higher_quantiles
                else:
                    y_pred_multi = self._pred_multi(X)

                    if self.method == "enbpi":
                        # Correspond to "Conformal prediction for dynamic time
                        # series". Its PIs are closed to the oracle's ones if
                        # beta_optimized is True.
                        pred = aggregate_all(self.agg_function, y_pred_multi)
                        lower_bounds, upper_bounds = pred, pred
                    else:  # self.method == "minmax":
                        lower_bounds = np.min(
                            y_pred_multi, axis=1, keepdims=True
                        )
                        upper_bounds = np.max(
                            y_pred_multi, axis=1, keepdims=True
                        )

                    y_pred_low = np.column_stack(
                        [
                            lower_bounds + lower_quantiles[k]
                            for k, _ in enumerate(alpha_np)
                        ]
                    )
                    y_pred_up = np.column_stack(
                        [
                            upper_bounds + higher_quantiles[k]
                            for k, _ in enumerate(alpha_np)
                        ]
                    )

                    if ensemble:
                        y_pred = aggregate_all(self.agg_function, y_pred_multi)

            else:  # self.method == "plus":
                # This version of predict corresponds to "Predictive
                # Inference is Free with the Jackknife+-after-Bootstrap.".
                # Its PIs are wider. It does not coorespond to "Conformal
                # prediction for dynamic time series". It is a try. It is
                # slower because the betas (width optimization parameters
                # of the PIs) are optimized for every point
                y_pred_multi = self._pred_multi(X)
                y_pred_low = np.empty((len(y_pred), len(alpha)), dtype=float)
                y_pred_up = np.empty((len(y_pred), len(alpha)), dtype=float)

                lower_bounds = y_pred_multi + self.conformity_scores_
                upper_bounds = y_pred_multi + self.conformity_scores_

                betas_0 = self._beta_optimize(
                    alpha=alpha_np,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    beta_optimize=beta_optimize,
                )

                for ind_alpha, _alpha in enumerate(alpha_np):
                    lower_quantiles = np.empty((betas_0.shape[1],))
                    upper_quantiles = np.empty((betas_0.shape[1],))

                    for ind_beta_0, beta_0 in enumerate(betas_0[ind_alpha, :]):
                        lower_quantiles[ind_beta_0] = np_nanquantile(
                            lower_bounds[ind_beta_0, :],
                            beta_0,
                            axis=0,
                            method="lower",
                        )  # type: ignore

                        upper_quantiles[ind_beta_0] = np_nanquantile(
                            upper_bounds[ind_beta_0, :],
                            1 - _alpha + beta_0,
                            axis=0,
                            method="higher",
                        )  # type: ignore
                    y_pred_low[:, ind_alpha] = lower_quantiles
                    y_pred_up[:, ind_alpha] = upper_quantiles

                if ensemble:
                    y_pred = aggregate_all(self.agg_function, y_pred_multi)

            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)

    def root_predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        ``root_predict`` method correspond to the one of ``MapieRegressor``'s.
        """
        conformity_scores_save = self.conformity_scores_.copy()
        self.conformity_scores_ = np.abs(self.conformity_scores_)
        if alpha is None:
            y_pred = super().predict(X=X, ensemble=ensemble, alpha=alpha)
            self.conformity_scores_ = conformity_scores_save
            return y_pred
        y_pred, y_pis = super().predict(X=X, ensemble=ensemble, alpha=alpha)
        self.conformity_scores_ = conformity_scores_save
        return y_pred, y_pis
