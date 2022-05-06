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

    This class implements the EnbPI strategy for estimating
    prediction intervals on single-output time series. The only valid
    ``method`` is 'enbpi'.

    Actually, EnbPI only corresponds to ``MapieTimeSeriesRegressor`` if the
    ``cv`` argument is of type ``BlockBootstrap``.

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
        self.valid_methods_ = ["enbpi"]
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
        y_pred, _ = super().predict(X, alpha=0.5, ensemble=True)
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

        Raises
        ------
        ValueError
            If the lenght of y is greater than the lenght of the training set.
        """
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        if len(X) > len(self.conformity_scores_):
            raise ValueError(
                "You try to update more residuals than there are!"
            )
        new_conformity_scores_ = self._relative_conformity_scores(X, y)
        self.conformity_scores_ = np.roll(
            self.conformity_scores_, -len(new_conformity_scores_)
        )
        self.conformity_scores_[-len(new_conformity_scores_):] = (
            new_conformity_scores_
        )
        return self

    def _beta_optimize(
        self,
        alpha: Union[float, NDArray],
        upper_bounds: NDArray,
        lower_bounds: NDArray,
    ) -> NDArray:
        """
        ``_beta_optimize`` offers to minimize the width of the PIs, for a given
        difference of quantiles.

        Parameters
        ----------
        alpha: Union[float, NDArray]
            The quantiles to compute.
        upper_bounds: NDArray
            The array of upper values.
        lower_bounds: NDArray
            The array of lower values.

        Returns
        -------
        NDArray
            Array of betas minimizing the differences
            ``(1-alpa+beta)-quantile - beta-quantile``.

        Raises
        ------
        ValueError
            If lower and upper bounds arrays don't have the same shape.
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
        Correspond to 'Conformal prediction for dynamic time-series'.

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

            if beta_optimize:
                betas_0 = self._beta_optimize(
                    alpha=alpha_np,
                    lower_bounds=self.conformity_scores_.reshape(1, -1),
                    upper_bounds=self.conformity_scores_.reshape(1, -1),
                )
            else:
                betas_0 = np.full(
                    shape=(len(alpha), len(self.conformity_scores_)),
                    fill_value=np.nan,
                    dtype=float,
                )
                for ind_alpha, _alpha in enumerate(alpha):
                    betas_0[ind_alpha, :] = _alpha / 2.0

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

            if self.cv == "prefit":
                y_pred_low = y_pred[:, np.newaxis] + lower_quantiles
                y_pred_up = y_pred[:, np.newaxis] + higher_quantiles
            else:
                y_pred_multi = self._pred_multi(X)
                pred = aggregate_all(self.agg_function, y_pred_multi)
                lower_bounds, upper_bounds = pred, pred

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

            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
