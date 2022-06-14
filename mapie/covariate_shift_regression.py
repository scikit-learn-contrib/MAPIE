from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import (
    check_is_fitted,
)

from ._typing import ArrayLike
from .dre import DensityRatioEstimator, ProbClassificationDRE
from .utils import (
    check_alpha,
    check_alpha_and_n_samples,
    empirical_quantile
)
from .regression import MapieRegressor


class MapieCovShiftRegressor(MapieRegressor):  # type: ignore
    """
    Prediction interval with out-of-fold residuals.

    This class implements the jackknife+ strategy and its variations
    for estimating prediction intervals on single-output data. The
    idea is to evaluate out-of-fold residuals on hold-out validation
    sets and to deduce valid confidence intervals with strong theoretical
    guarantees.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with fit and predict methods), by default ``None``.
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

    dr_estimator : Optional[DensityRatioEstimator]
        Any density ratio estimator with scikit-learn API
        (i.e. with fit and predict methods), by default ``None``.
        If ``None``, dr_estimator defaults to a ``ProbClassificationDRE``
        instance with ``LogisticRegression`` model.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "naive", based on training set residuals,
        - "base", based on validation sets residuals,
        - "plus", based on validation residuals and testing predictions,
        - "minmax", based on validation residuals and testing predictions
          (min/max among cross-validation clones).

        By default "plus".

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing residuals.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
          - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
          - ``sklearn.model_selection.KFold`` (cross-validation),
          - ``subsample.Subsample`` object (bootstrap).
        - ``"prefit"``, assumes that ``estimator`` has been fitted already,
          and the ``method`` parameter is ignored.
          All data provided in the ``fit`` method is then used
          for computing residuals only.
          At prediction time, quantiles of these residuals are used to provide
          a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          residual estimate are disjoint.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 - n_jobs)`` are used.
        None is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    agg_function : str
        Determines how to aggregate predictions from perturbed models, both at
        training and prediction time.

        If ``None``, it is ignored except if cv class is ``Subsample``,
        in which case an error is raised.
        If "mean" or "median", returns the mean or median of the predictions
        computed from the out-of-folds models.
        Note: if you plan to set the ``ensemble`` argument to ``True`` in the
        ``predict`` method, you have to specify an aggregation function.
        Otherwise an error would be raised.

        The Jackknife+ interval can be interpreted as an interval around the
        median prediction, and is guaranteed to lie inside the interval,
        unlike the single estimator predictions.

        When the cross-validation strategy is Subsample (i.e. for the
        Jackknife+-after-Bootstrap method), this function is also used to
        aggregate the training set in-sample predictions.

        If cv is ``"prefit"``, ``agg_function`` is ignored.

        By default "mean".

    verbose : int, optional
        The verbosity level, used with joblib for multiprocessing.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    single_estimator_ : sklearn.RegressorMixin
        Estimator fitted on the whole training set.

    estimators_ : list
        List of out-of-folds estimators.

    residuals_ : ArrayLike of shape (n_samples_train,)
        Residuals between ``y_train`` and ``y_pred``.

    k_ : ArrayLike
        - Array of nans, of shape (len(y), 1) if cv is ``"prefit"``
          (defined but not used)
        - Dummy array of folds containing each training sample, otherwise.
          Of shape (n_samples_train, cv.get_n_splits(X_train, y_train)).

    n_features_in_: int
        Number of features passed to the fit method.

    n_samples_: List[int]
        Number of samples passed to the fit method.

    References
    ----------

    Examples
    --------

    """
    valid_methods_ = ["naive", "base"]
    valid_agg_functions_ = [None, "median", "mean"]
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "residuals_",
        "residuals_dre_",
        "n_features_in_",
        "n_samples_",
    ]

    def __init__(
        self,
        estimator: Optional[RegressorMixin] = None,
        dr_estimator: Optional[DensityRatioEstimator] = None,
        method: str = "base",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
    ) -> None:
        self.dr_estimator = dr_estimator
        if cv != "prefit":
            raise NotImplementedError
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
        )

    def _check_dr_estimator(
        self,
        dr_estimator: Optional[DensityRatioEstimator] = None
    ) -> DensityRatioEstimator:
        """
        Check if estimator is ``None``, and returns a ``ProbClassificationDRE``
        instance with ``LogisticRegression`` model if necessary.
        If the ``cv`` attribute is ``"prefit"``, check if estimator is indeed
        already fitted.

        Parameters
        ----------
        dr_estimator : Optional[DensityRatioEstimator], optional
            Estimator to check, by default ``None``.

        Returns
        -------
        DensityRatioEstimator
            The estimator itself or a default ``ProbClassificationDRE``
            instance with ``LogisticRegression`` model.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit nor predict methods.

        NotFittedError
            If the estimator is not fitted and ``cv`` attribute is "prefit".
        """
        if dr_estimator is None:
            return ProbClassificationDRE(clip_min=0.01, clip_max=0.99)
        if not (hasattr(dr_estimator, "fit") and
                hasattr(dr_estimator, "predict")):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a density ratio estimator with fit"
                "and predict methods."
            )
        if self.cv == "prefit":
            dr_estimator.check_is_fitted()

        return dr_estimator

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieRegressor:
        """
        Fit estimator and compute residuals used for prediction intervals.
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold residuals are stored under the ``residuals_`` attribute.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.

            By default ``None``.

        Returns
        -------
        MapieRegressor
            The model itself.
        """
        super().fit(X=X, y=y, sample_weight=sample_weight)
        self.residuals_dre_ = self.dr_estimator.predict(X)

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of residuals (naive and base methods),
        - quantiles of (predictions +/- residuals) (plus method),
        - quantiles of (max/min(predictions) +/- residuals) (minmax method).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If False, predictions are those of the model trained on the whole
            training set.
            If True, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If cv is ``"prefit"``, ``ensemble`` is ignored.

            By default ``False``.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represents the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        Returns
        -------
        Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]

        - ArrayLike of shape (n_samples,) if alpha is None.

        - Tuple[ArrayLike, ArrayLike] of shapes
        (n_samples,) and (n_samples, 2, n_alpha) if alpha is not None.

            - [:, 0, :]: Lower bound of the prediction interval.
            - [:, 1, :]: Upper bound of the prediction interval.
        """
        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha_ = check_alpha(alpha)

        y_pred = self.single_estimator_.predict(X)
        dre_pred = self.dr_estimator.predict(X)
        dre_calib = self.residuals_dre_

        if alpha is None:
            return np.array(y_pred)
        else:
            alpha_ = cast(ArrayLike, alpha_)
            check_alpha_and_n_samples(alpha_, self.residuals_.shape[0])
            if self.method in ["naive", "base"] or self.cv == "prefit":

                # Denominator in weight calculation (array; differs based
                # on each test point)
                denom = dre_calib.sum() + dre_pred

                y_pred_low = np.empty(
                    (y_pred.shape[0], len(alpha_)), dtype=y_pred.dtype)
                y_pred_up = np.empty_like(y_pred_low, dtype=y_pred.dtype)
                for i in range(dre_pred.shape[0]):

                    # Numerator in weight calculation
                    # Calibration (array)
                    cal_weights = dre_calib / denom[i]
                    # Test (float)
                    test_weight = dre_pred[i] / denom[i]

                    # Calculate the quantile for constructing interval
                    quantile = empirical_quantile(
                        np.hstack([self.residuals_, np.array([np.inf])]),
                        alphas=1-alpha_,
                        weights=np.hstack(
                            [cal_weights, np.array([test_weight])]),
                    )

                    y_pred_low[i, :] = y_pred[i] - quantile
                    y_pred_up[i, :] = y_pred[i] + quantile

            else:
                raise NotImplementedError

            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
