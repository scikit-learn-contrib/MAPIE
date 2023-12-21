from __future__ import annotations

from typing import Optional, Union, cast

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import ConformityScore
from mapie.regression import MapieRegressor
from mapie.estimator.estimator import EnsembleRegressor
from mapie.utils import (check_gamma, convert_to_numpy)


class MapieTimeSeriesRegressor(MapieRegressor):
    """
    Prediction intervals with out-of-fold residuals for time series.
    This class only has two valid ``method`` : ``"enbpi"`` or ``"aci"``

    The prediction intervals are calibrated on a split of the trained data.
    Both strategies are estimating prediction intervals
    on single-output time series.

    EnbPI allows you to update conformal scores using the ``partial_fit``
    function. It will replace the oldest one with the newest scores.
    It will keep the same amount of total scores

    Actually, EnbPI only corresponds to ``MapieTimeSeriesRegressor`` if the
    ``cv`` argument is of type ``BlockBootstrap``.

    The ACI strategy allows you to adapt the conformal inference
    (i.e the quantile). If the real values are not in the coverage,
    the size of the intervals will grow.
    Conversely, if the real values are in the coverage,
    the size of the intervals will decrease.
    You can use a gamma coefficient to adjust the strength of the correction.

    References
    ----------
    Chen Xu, and Yao Xie.
    "Conformal prediction for dynamic time-series."
    https://arxiv.org/abs/2010.09107

    Isaac Gibbs, Emmanuel Candes
    "Adaptive conformal inference under distribution shift"
    https://proceedings.neurips.cc/paper/2021/file/\
    0d441de75945e5acbc865406fc9a2559-Paper.pdf

    Margaux Zaffran et al.
    "Adaptive Conformal Predictions for Time Series"
    https://arxiv.org/pdf/2202.07282.pdf
    """

    cv_need_agg_function_ = MapieRegressor.cv_need_agg_function_ + \
        ["BlockBootstrap"]
    valid_methods_ = ["enbpi", "aci"]
    default_sym_ = False

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        method: str = "enbpi",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        conformity_score: Optional[ConformityScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
            conformity_score=conformity_score,
            random_state=random_state,
        )

    def _relative_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
    ) -> NDArray:
        """
        Compute the conformity scores on a data set.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
                Input labels.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        Returns
        -------
            The conformity scores corresponding to the input data set.
        """
        y_pred = super().predict(X, ensemble=ensemble)
        scores = np.array(
            self.conformity_score_function_.get_conformity_scores(X, y, y_pred)
        )
        return scores

    def init_alpha(self) -> None:
        if self.method == "aci":
            self.current_alpha: dict[float, float] = {}
        return None

    def pre_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieTimeSeriesRegressor:
        """
        Prefit to only fit the model
        """
        (
            estimator,
            self.conformity_score_function_,
            agg_function,
            cv,
            X,
            y,
            sample_weight,
        ) = self._check_fit_parameters(X, y, sample_weight)

        self.estimator_ = EnsembleRegressor(
            estimator,
            self.method,
            cv,
            agg_function,
            self.n_jobs,
            self.random_state,
            self.test_size,
            self.verbose,
        )
        # Fit the prediction function
        self.estimator_ = self.estimator_.fit(X, y, sample_weight)

        return self

    def update(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieTimeSeriesRegressor:
        self = super().fit(X=X, y=y, sample_weight=sample_weight)
        self.conformity_scores_ = self._relative_conformity_scores(X, y)
        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        ensemble: bool = False,
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

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        Returns
        -------
        MapieTimeSeriesRegressor
            The model itself.
        """
        self = super().fit(X=X, y=y, sample_weight=sample_weight)
        self.conformity_scores_ = self._relative_conformity_scores(
            X, y, ensemble=ensemble
        )
        if self.method == "aci":
            self.current_alpha: dict[float, float] = {}
        return self

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        ensemble: bool = False,
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

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

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
        new_conformity_scores_ = self._relative_conformity_scores(
            X, y, ensemble=ensemble
        )
        self.conformity_scores_ = np.roll(
            self.conformity_scores_, -len(new_conformity_scores_)
        )
        self.conformity_scores_[-len(new_conformity_scores_):] = \
            new_conformity_scores_
        return self

    def adapt_conformal_inference(
        self,
        X: ArrayLike,
        y_true: ArrayLike,
        gamma: float,
    ) -> MapieTimeSeriesRegressor:
        """
        Adapt the ``alpha_t`` attribute when new data with known
        labels are available.
        Note: Don't use ``adapt_conformal_inference``
        with samples of the training set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        y_true: ArrayLike of shape (n_samples_test,)
            Input labels.

        gamma: float
            Coefficient that decides the correction of the conformal inference.
            If it equals 0, there are no corrections.

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
        check_gamma(gamma)

        if self.method != "aci":
            raise AttributeError(
                "This method can be called "
                f"only with method='aci', "
                f"not with '{self.method}'."
            )

        X = cast(NDArray, X)
        y_true = cast(NDArray, y_true)

        X, y_true = convert_to_numpy(X, y_true)

        for x_row, y_row in zip(X, y_true):
            x = np.expand_dims(x_row, axis=0)
            _, y_pred_bounds = self.predict(
                x, alpha=list(self.current_alpha.keys())
            )

            for alpha_ix, alpha_0 in enumerate(self.current_alpha):
                alpha_t = self.current_alpha[alpha_0]
                is_true_in_quantile = 1 - float(
                    y_pred_bounds[:, 0, alpha_ix]
                    < y_row
                    < y_pred_bounds[:, 1, alpha_ix]
                )

                new_alpha_t = np.clip(
                    alpha_t + gamma * (alpha_0 - is_true_in_quantile), 0, 1
                )
                self.current_alpha[alpha_0] = new_alpha_t

        return self

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_estimators_partial_fit_n_features":
                "partial_fit can only be called on fitted models"
            }
        }
