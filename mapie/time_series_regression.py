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
    It is ``MapieReegressor`` with one more method ``partial_fit``.
    Actually, EnbPI only corresponds to MapieRegressor if the ``cv`` argument
    if of type ``Subsample`` (Jackknife+-after-Bootstrap method). Moreover, for
    the moment we consider the absolute values of the residuals of the model,
    and consequently the prediction intervals are symetryc.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with fit and predict methods), by default ``None``.
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

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

    n_samples_val_: List[int]
        Number of samples passed to the fit method.

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

    def partial_update(
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
            Boolean corresponing to the ``ensemble`` argument of ``predict``
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
