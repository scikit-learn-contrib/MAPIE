from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import numpy.ma as ma
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import (
    _check_y,
    _num_samples,
    check_is_fitted,
    indexable,
)

from ._typing import ArrayLike
from .aggregation_functions import aggregate_all, phi2D
from .conformity_scores import AbsoluteConformityScore, ConformityScore
from .subsample import Subsample
from .utils import (
    check_alpha,
    check_alpha_and_n_samples,
    check_cv,
    check_n_features_in,
    check_n_jobs,
    check_nan_in_aposteriori_prediction,
    check_null_weight,
    check_verbose,
    fit_estimator,
)


class MapieRegressor(BaseEstimator, RegressorMixin):  # type: ignore
    """
    Prediction interval with out-of-fold conformity scores.

    This class implements the jackknife+ strategy and its variations
    for estimating prediction intervals on single-output data. The
    idea is to evaluate out-of-fold conformity scores on hold-out validation
    sets and to deduce valid confidence intervals with strong theoretical
    guarantees.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with fit and predict methods), by default ``None``.
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "naive", based on training set conformity scores,
        - "base", based on validation sets conformity scores,
        - "plus", based on validation conformity scores and
          testing predictions,
        - "minmax", based on validation conformity scores and
          testing predictions (min/max among cross-validation clones).

        By default "plus".

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing conformity scores.
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
          for computing conformity scores only.
          At prediction time, quantiles of these conformity scores are used
          to provide a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          conformity scores estimate are disjoint.

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

    conformity_scores_ : ArrayLike of shape (n_samples_train,)
        Conformity scores between ``y_train`` and ``y_pred``.

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
    Rina Foygel Barber, Emmanuel J. Candès,
    Aaditya Ramdas, and Ryan J. Tibshirani.
    "Predictive inference with the jackknife+."
    Ann. Statist., 49(1):486–507, February 2021.

    Byol Kim, Chen Xu, and Rina Foygel Barber.
    "Predictive Inference Is Free with the Jackknife+-after-Bootstrap."
    34th Conference on Neural Information Processing Systems (NeurIPS 2020).

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import MapieRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from mapie.conformity_scores import AbsoluteConformityScore
    >>> X_toy = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y_toy = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> mapie_reg = MapieRegressor(LinearRegression()).fit(X_toy, y_toy)
    >>> y_pred, y_pis = mapie_reg.predict(X_toy, alpha=0.5)
    >>> print(y_pis[:, :, 0])
    [[ 4.7972973   5.8       ]
     [ 6.69767442  7.65540541]
     [ 8.59883721  9.58108108]
     [10.5        11.40116279]
     [12.4        13.30232558]
     [14.25       15.20348837]]
    >>> print(y_pred)
    [ 5.28571429  7.17142857  9.05714286 10.94285714 12.82857143 14.71428571]
    """

    valid_methods_ = ["naive", "base", "plus", "minmax"]
    valid_agg_functions_ = [None, "median", "mean"]
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "conformity_score_",
        "conformity_scores_",
        "n_features_in_",
        "n_samples_",
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
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.agg_function = agg_function
        self.verbose = verbose

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        if self.method not in self.valid_methods_:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'naive', 'base', 'plus' and 'minmax'."
            )

        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)

    def _check_agg_function(
        self, agg_function: Optional[str] = None
    ) -> Optional[str]:
        """
        Check if ``agg_function`` is correct, and consistent with other
        arguments.

        Parameters
        ----------
        agg_function : Optional[str], optional
            Aggregation function's name to check, by default ``None``.

        Returns
        -------
        str
            ``agg_function`` itself or ``"mean"``.

        Raises
        ------
        ValueError
            If ``agg_function`` is not in [``None``, ``"mean"``, ``"median"``],
            or is ``None`` while cv class is ``Subsample``.
        """
        if agg_function not in self.valid_agg_functions_:
            raise ValueError(
                "Invalid aggregation function "
                "Allowed values are None, 'mean', 'median'."
            )

        if isinstance(self.cv, Subsample) and (agg_function is None):
            raise ValueError(
                "You need to specify an aggregation function when "
                "cv is a Subsample. "
            )
        if (agg_function is not None) or (self.cv == "prefit"):
            return agg_function
        return "mean"

    def _check_estimator(
        self, estimator: Optional[RegressorMixin] = None
    ) -> RegressorMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LinearRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator : Optional[RegressorMixin], optional
            Estimator to check, by default ``None``.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``LinearRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit nor predict methods.

        NotFittedError
            If the estimator is not fitted and ``cv`` attribute is "prefit".
        """
        if estimator is None:
            return LinearRegression()
        if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a regressor with fit and predict methods."
            )
        if self.cv == "prefit":
            if isinstance(self.estimator, Pipeline):
                check_is_fitted(self.estimator[-1])
            else:
                check_is_fitted(self.estimator)
        return estimator

    def _check_ensemble(
        self,
        ensemble: bool,
    ) -> None:
        """
        Check if ``ensemble`` is False if ``self.agg_function`` is ``None``.
        Else raise error.

        Parameters
        ----------
        ensemble : bool
            ``ensemble`` argument to check the coherennce with
            ``self.agg_function``.

        Raises
        ------
        ValueError
            If ``ensemble`` is True and ``self.agg_function`` is None.
        """
        if ensemble and (self.agg_function is None):
            raise ValueError(
                "If ensemble is True, the aggregation function has to be "
                "'mean' or 'median'."
            )

    def _fit_and_predict_oof_model(
        self,
        estimator: RegressorMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        val_index: ArrayLike,
        k: int,
        sample_weight: Optional[ArrayLike] = None,
    ) -> Tuple[RegressorMixin, ArrayLike, ArrayLike]:
        """
        Fit a single out-of-fold model on a given training set and
        perform predictions on a test set.

        Parameters
        ----------
        estimator : RegressorMixin
            Estimator to train.

        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
            Input labels.

        train_index : ArrayLike of shape (n_samples_train)
            Training data indices.

        val_index : ArrayLike of shape (n_samples_val)
            Validation data indices.

        k : int
            Split identification number.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default ``None``.

        Returns
        -------
        Tuple[RegressorMixin, ArrayLike, ArrayLike]

        - [0]: Fitted estimator
        - [1]: Estimator predictions on the validation fold,
          of shape (n_samples_val,)
        - [3]: Validation data indices,
          of shape (n_samples_val,).

        """
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        X_val = _safe_indexing(X, val_index)
        if sample_weight is None:
            estimator = fit_estimator(estimator, X_train, y_train)
        else:
            sample_weight_train = _safe_indexing(sample_weight, train_index)
            estimator = fit_estimator(
                estimator, X_train, y_train, sample_weight_train
            )
        if _num_samples(X_val) > 0:
            y_pred = estimator.predict(X_val)
        else:
            y_pred = np.array([])
        return estimator, y_pred, val_index

    def aggregate_with_mask(self, x: ArrayLike, k: ArrayLike) -> ArrayLike:
        """
        Take the array of predictions, made by the refitted estimators,
        on the testing set, and the 1-nan array indicating for each training
        sample which one to integrate, and aggregate to produce phi-{t}(x_t)
        for each training sample x_t.


        Parameters:
        -----------
        x : ArrayLike of shape (n_samples_test, n_estimators)
            Array of predictions, made by the refitted estimators,
            for each sample of the testing set.

        k : ArrayLike of shape (n_samples_training, n_estimators)
            1-or-nan array: indicates whether to integrate the prediction
            of a given estimator into the aggregation, for each training
            sample.

        Returns:
        --------
        ArrayLike of shape (n_samples_test,)
            Array of aggregated predictions for each testing  sample.
        """
        if self.agg_function == "median":
            return phi2D(A=x, B=k, fun=lambda x: np.nanmedian(x, axis=1))
        if self.cv == "prefit":
            raise ValueError(
                "There should not be aggregation of predictions if cv is "
                "'prefit'"
            )
        # To aggregate with mean() the aggregation coud be done
        # with phi2D(A=x, B=k, fun=lambda x: np.nanmean(x, axis=1).
        # However, phi2D contains a np.apply_along_axis loop which
        # is much slower than the matrices multiplication that can
        # be used to compute the means.
        if self.agg_function in ["mean", None]:
            K = np.nan_to_num(k, nan=0.0)
            return np.matmul(x, (K / (K.sum(axis=1, keepdims=True))).T)
        raise ValueError("The value of self.agg_function is not correct")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        conformity_score: ConformityScore = AbsoluteConformityScore(),
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieRegressor:
        """
        Fit estimator and compute conformity scores used for
        prediction intervals.
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold conformity scores are stored under
        the ``conformity_scores_`` attribute.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        conformity_score : Optional[ConformityScore]
            ConformityScore instance.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no conformity scores.
            If weights are non-uniform,
            conformity scores are still uniformly weighted.

            By default ``None``.

        Returns
        -------
        MapieRegressor
            The model itself.
        """
        # Checks
        self._check_parameters()
        cv = check_cv(self.cv)
        estimator = self._check_estimator(self.estimator)
        agg_function = self._check_agg_function(self.agg_function)
        X, y = indexable(X, y)
        y = _check_y(y)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        # Initialization
        self.estimators_: List[RegressorMixin] = []

        # Work
        if cv == "prefit":
            self.single_estimator_ = estimator
            y_pred = self.single_estimator_.predict(X)
            self.n_samples_ = [_num_samples(X)]
            self.k_ = np.full(
                shape=(len(y), 1), fill_value=np.nan, dtype=float
            )
        else:
            self.k_ = np.full(
                shape=(len(y), cv.get_n_splits(X, y)),  # type: ignore
                fill_value=np.nan,
                dtype=float,
            )

            pred_matrix = np.full(
                shape=(len(y), cv.get_n_splits(X, y)),  # type: ignore
                fill_value=np.nan,
                dtype=float,
            )

            self.single_estimator_ = fit_estimator(
                clone(estimator), X, y, sample_weight
            )
            if self.method == "naive":
                y_pred = self.single_estimator_.predict(X)
                self.n_samples_ = [_num_samples(X)]
            else:
                outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._fit_and_predict_oof_model)(
                        clone(estimator),
                        X,
                        y,
                        train_index,
                        val_index,
                        k,
                        sample_weight,
                    )
                    for k, (train_index, val_index) in enumerate(cv.split(X))
                )
                self.estimators_, predictions, val_indices = map(
                    list, zip(*outputs)
                )

                self.n_samples_ = [
                    np.array(pred).shape[0] for pred in predictions
                ]

                for i, val_ind in enumerate(val_indices):
                    pred_matrix[val_ind, i] = np.array(predictions[i]).ravel()
                    self.k_[val_ind, i] = 1
                check_nan_in_aposteriori_prediction(pred_matrix)

                y_pred = aggregate_all(agg_function, pred_matrix)

        conformity_score.check_consistency(y, y_pred)
        self.conformity_scores_ = conformity_score.get_conformity_scores(
            y, y_pred
        )
        self.conformity_score_ = conformity_score

        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
        """
        Predict target on new samples with confidence intervals.
        Conformity scores from the training set and predictions
        from the model clones are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of conformity scores (naive and base methods),
        - quantiles of (predictions +/- conformity scores) (plus method),
        - quantiles of (max/min(predictions) +/- conformity scores)
        (minmax method).

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

        if alpha is None:
            return np.array(y_pred)
        else:
            alpha_ = cast(ArrayLike, alpha_)
            check_alpha_and_n_samples(alpha_, self.conformity_scores_.shape[0])
            if self.method in ["naive", "base"] or self.cv == "prefit":
                if self.conformity_score_.sym:
                    conformity_scores_q_low_bound = -np.quantile(
                        self.conformity_scores_,
                        1 - alpha_,
                        interpolation="higher",
                    )
                    conformity_scores_q_up_bound = (
                        -conformity_scores_q_low_bound
                    )
                else:
                    alpha_lower_bound = alpha_ / 2
                    alpha_upper_bound = 1 - alpha_ / 2
                    conformity_scores_q_low_bound = np.quantile(
                        self.conformity_scores_,
                        alpha_lower_bound,
                        interpolation="higher",
                    )
                    conformity_scores_q_up_bound = np.quantile(
                        self.conformity_scores_,
                        alpha_upper_bound,
                        interpolation="higher",
                    )
                y_pred_low = self.conformity_score_.get_observed_value(
                    y_pred[:, np.newaxis], conformity_scores_q_low_bound
                )
                y_pred_up = self.conformity_score_.get_observed_value(
                    y_pred[:, np.newaxis], conformity_scores_q_up_bound
                )
            else:
                y_pred_multi = np.column_stack(
                    [e.predict(X) for e in self.estimators_]
                )

                # At this point, y_pred_multi is of shape
                # (n_samples_test, n_estimators_).
                # If ``method`` is "plus":
                #   - if ``cv`` is not a ``Subsample``,
                #       we enforce y_pred_multi to be of shape
                #       (n_samples_test, n_samples_train),
                #       thanks to the folds identifier.
                #   - if ``cv``is a ``Subsample``, the methode
                #       ``aggregate_with_mask`` fits it to the right size
                #       thanks to the shape of k_.

                y_pred_multi = self.aggregate_with_mask(y_pred_multi, self.k_)

                if self.method == "plus":
                    if self.conformity_score_.sym:
                        y_pred_multi_with_conformity_scores_lower_bound = (
                            self.conformity_score_.get_observed_value(
                                y_pred_multi, -self.conformity_scores_
                            )
                        )
                        y_pred_multi_with_conformity_scores_upper_bound = (
                            self.conformity_score_.get_observed_value(
                                y_pred_multi, self.conformity_scores_
                            )
                        )
                    else:
                        y_pred_multi_with_conformity_scores_lower_bound = (
                            self.conformity_score_.get_observed_value(
                                y_pred_multi, self.conformity_scores_
                            )
                        )
                        y_pred_multi_with_conformity_scores_upper_bound = (
                            y_pred_multi_with_conformity_scores_lower_bound
                        )

                if self.method == "minmax":
                    lower_bounds = np.min(y_pred_multi, axis=1, keepdims=True)
                    upper_bounds = np.max(y_pred_multi, axis=1, keepdims=True)
                    if self.conformity_score_.sym:
                        y_pred_multi_with_conformity_scores_lower_bound = (
                            self.conformity_score_.get_observed_value(
                                lower_bounds, -self.conformity_scores_
                            )
                        )
                        y_pred_multi_with_conformity_scores_upper_bound = (
                            self.conformity_score_.get_observed_value(
                                upper_bounds, self.conformity_scores_
                            )
                        )
                    else:
                        y_pred_multi_with_conformity_scores_lower_bound = (
                            self.conformity_score_.get_observed_value(
                                lower_bounds, self.conformity_scores_
                            )
                        )
                        y_pred_multi_with_conformity_scores_upper_bound = (
                            self.conformity_score_.get_observed_value(
                                upper_bounds, self.conformity_scores_
                            )
                        )

                y_pred_low = np.column_stack(
                    [
                        np.quantile(
                            ma.masked_invalid(
                                y_pred_multi_with_conformity_scores_lower_bound
                            ),
                            _alpha,
                            axis=1,
                            interpolation="lower",
                        )
                        for _alpha in alpha_
                    ]
                ).data
                y_pred_up = np.column_stack(
                    [
                        np.quantile(
                            ma.masked_invalid(
                                y_pred_multi_with_conformity_scores_upper_bound
                            ),
                            1 - _alpha,
                            axis=1,
                            interpolation="higher",
                        )
                        for _alpha in alpha_
                    ]
                ).data
                if ensemble:
                    y_pred = aggregate_all(self.agg_function, y_pred_multi)
            return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
