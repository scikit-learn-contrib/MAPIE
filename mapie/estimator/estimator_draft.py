from __future__ import annotations

from typing import List, Optional, Tuple, Union, cast

import numpy as np
import inspect
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples, check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.aggregation_functions import aggregate_all, phi2D
from mapie.estimator.interface import EnsembleEstimator
from mapie.utils import (check_nan_in_aposteriori_prediction, check_no_agg_cv,
                         fit_estimator, fix_number_of_classes)


class EnsembleClassifier(EnsembleEstimator):
    """
    This class implements methods to handle the training and usage of the
    estimator. This estimator can be unique or composed by cross validated
    estimators.

    Parameters
    ----------
    estimator: Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

        By default ``None``.

    cv: Optional[str]
        The cross-validation strategy for computing scores.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
          - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
          - ``sklearn.model_selection.KFold`` (cross-validation)
        - ``"split"``, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          to calibrate the predictions through the score computation.
          At prediction time, quantiles of these scores are used to estimate
          prediction sets.

        By default ``None``.

    test_size: Optional[Union[int, float]]
        If ``float``, should be between ``0.0`` and ``1.0`` and represent the
        proportion of the dataset to include in the test split. If ``int``,
        represents the absolute number of test samples. If ``None``,
        it will be set to ``0.1``.

        If cv is not ``"split"``, ``test_size`` is ignored.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For ``n_jobs`` below ``-1``, ``(n_cpus + 1 - n_jobs)`` are used.
        ``None`` is a marker for `unset` that will be interpreted as
        ``n_jobs=1`` (sequential execution).

        By default ``None``.

   random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        for evaluation quantiles and prediction sets.
        Pass an int for reproducible output across multiple function calls.

        By default ``None``.

    verbose: int, optional
        The verbosity level, used with joblib for multiprocessing.
        At this moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    single_estimator_: sklearn.RegressorMixin
        Estimator fitted on the whole training set.

    estimators_: list
        List of out-of-folds estimators.

    k_: ArrayLike
        - Array of nans, of shape (len(y), 1) if ``cv`` is ``"prefit"``
          (defined but not used)
        - Dummy array of folds containing each training sample, otherwise.
          Of shape (n_samples_train, cv.get_n_splits(X_train, y_train)).
    """
    no_agg_cv_ = ["prefit", "split"]
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "use_split_method_",
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin],
        n_classes: int,
        cv: Optional[Union[int, str, BaseCrossValidator]],
        n_jobs: Optional[int],
        random_state: Optional[Union[int, np.random.RandomState]],
        test_size: Optional[Union[int, float]],
        verbose: int
    ):
        print()
        print("EC : USE OF INIT")
        self.estimator = estimator
        print()
        print("estimator", estimator)
        self.n_classes = n_classes
        print()
        print("n_classes", n_classes)
        self.cv = cv
        print()
        print("cv", cv)
        self.n_jobs = n_jobs
        print()
        print("n_jobs", n_jobs)
        self.random_state = random_state
        print()
        print("random_state", random_state)
        self.test_size = test_size
        print()
        print("test_size", test_size)
        self.verbose = verbose
        print()
        print("verbose", verbose)

    @staticmethod
    def _fit_oof_estimator(
        estimator: ClassifierMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params,
    ) -> ClassifierMixin:
        """
        Fit a single out-of-fold model on a given training set.

        Parameters
        ----------
        estimator: RegressorMixin
            Estimator to train.

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples,)
            Input labels.

        train_index: ArrayLike of shape (n_samples_train)
            Training data indices.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default ``None``.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        RegressorMixin
            Fitted estimator.
        """
        print()
        print("EC : use of _fit_oof_estimator :")
        X_train = _safe_indexing(X, train_index)
        print()
        print("X_train", X_train, "shape_X_train", X_train.shape)
        y_train = _safe_indexing(y, train_index)
        print()
        print("y_train", y_train,"shape_y_train", y_train.shape)
        if not (sample_weight is None):
            sample_weight = _safe_indexing(sample_weight, train_index)
            sample_weight = cast(NDArray, sample_weight)
            print()
            print("sample_weight", sample_weight)

        estimator = fit_estimator(
            estimator,
            X_train,
            y_train,
            sample_weight=sample_weight,
            **fit_params
        )
        print()
        print("estimator:", estimator)
        return estimator
    
    def _predict_proba_oof_estimator(self, estimator, X):
        print()
        print("EC : use of _predict_proba_oof_estimator")
        y_pred_proba = estimator.predict_proba(X)
        if len(estimator.classes_) != self.n_classes:
            y_pred_proba = fix_number_of_classes(
                self.n_classes,
                estimator.classes_,
                y_pred_proba
            )
        return y_pred_proba
        

    def _predict_proba_calib_oof_estimator(
        self,
        estimator: ClassifierMixin,
        X: ArrayLike,
        val_index: ArrayLike,
        k: int
    ) -> Tuple[NDArray, ArrayLike]:
        """
        Perform predictions on a single out-of-fold model on a validation set.

        Parameters
        ----------
        estimator: RegressorMixin
            Estimator to train.

        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        val_index: ArrayLike of shape (n_samples_val)
            Validation data indices.

        Returns
        -------
        Tuple[NDArray, ArrayLike]
            Predictions of estimator from val_index of X.
        """
        print()
        print("EC : use of _predict_proba_calib_oof_estimator")
        X_val = _safe_indexing(X, val_index)
        print()
        print("X_val:", X_val, "shape", X_val.shape)
        if _num_samples(X_val) > 0:
            y_pred_proba = self._predict_proba_oof_estimator(
                estimator, X_val
            )
            print()
            print("y_pred_proba", y_pred_proba, "shape:", y_pred_proba.shape)
        else:
            y_pred_proba = np.array([])
        val_id = np.full(len(X_val), k, dtype=int)
        print()
        print("val_id :", val_id, "val_index: ", val_index)
        return y_pred_proba, val_id, val_index

    def _aggregate_with_mask(
        self,
        x: NDArray,
        k: NDArray
    ) -> NDArray:
        """
        Take the array of predictions, made by the refitted estimators,
        on the testing set, and the 1-or-nan array indicating for each training
        sample which one to integrate, and aggregate to produce phi-{t}(x_t)
        for each training sample x_t.

        Parameters
        ----------
        x: ArrayLike of shape (n_samples_test, n_estimators)
            Array of predictions, made by the refitted estimators,
            for each sample of the testing set.

        k: ArrayLike of shape (n_samples_training, n_estimators)
            1-or-nan array: indicates whether to integrate the prediction
            of a given estimator into the aggregation, for each training
            sample.

        Returns
        -------
        ArrayLike of shape (n_samples_test,)
            Array of aggregated predictions for each testing sample.
        """
        print()
        print("EC : use of _aggregate_with_mask")
        if self.method in self.no_agg_methods_ or self.use_split_method_:
            raise ValueError(
                "There should not be aggregation of predictions "
                f"if cv is in '{self.no_agg_cv_}', if cv >=2 "
                f"or if method is in '{self.no_agg_methods_}'."
            )
        elif self.agg_function == "median":
            return phi2D(A=x, B=k, fun=lambda x: np.nanmedian(x, axis=1))
        # To aggregate with mean() the aggregation coud be done
        # with phi2D(A=x, B=k, fun=lambda x: np.nanmean(x, axis=1).
        # However, phi2D contains a np.apply_along_axis loop which
        # is much slower than the matrices multiplication that can
        # be used to compute the means.
        elif self.agg_function in ["mean", None]:
            K = np.nan_to_num(k, nan=0.0)
            return np.matmul(x, (K / (K.sum(axis=1, keepdims=True))).T)
        else:
            raise ValueError("The value of self.agg_function is not correct")

    def _pred_multi(self, X: ArrayLike) -> NDArray:
        """
        Return a prediction per train sample for each test sample, by
        aggregation with matrix ``k_``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data

        Returns
        -------
        NDArray of shape (n_samples_test, n_samples_train)
        """
        print()
        print("EC : use of _pred_multi")
        y_pred_multi = np.column_stack(
            [e.predict(X) for e in self.estimators_]
        )
        # At this point, y_pred_multi is of shape
        # (n_samples_test, n_estimators_). The method
        # ``_aggregate_with_mask`` fits it to the right size
        # thanks to the shape of k_.
        y_pred_multi = self._aggregate_with_mask(y_pred_multi, self.k_)
        return y_pred_multi

    def predict_proba_calib(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        y_enc=None,
        groups: Optional[ArrayLike] = None
    ) -> NDArray:
        """
        Perform predictions on X : the calibration set.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples_test, n_features)
            Input data

        y: Optional[ArrayLike] of shape (n_samples_test,)
            Input labels.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples_test,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        Returns
        -------
        NDArray of shape (n_samples_test, 1)
            The predictions.
        """
        print()
        print("EC : USE OF PREDICT_PROBA_CALIB")
        check_is_fitted(self, self.fit_attributes)

        if self.cv == "prefit":
            y_pred_proba = self.single_estimator_.predict_proba(X)
            print()
            print("dans le cas prefit: ", y_pred_proba)
        else:
            y_pred_proba = np.empty(
                (len(X), self.n_classes),
                dtype=float
            )
            print()
            print("y_pred_proba", y_pred_proba,"y_pred_proba_shape", y_pred_proba.shape, "y_pred_proba_max :", np.max(y_pred_proba), "y_pred_proba_min :", np.min(y_pred_proba))
            cv = cast(BaseCrossValidator, self.cv)
            print()
            print("cv", cv)
            outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed( 
                    self._predict_proba_calib_oof_estimator)(
                    estimator, X, calib_index, k
                )

                for k, ((_, calib_index), estimator) in enumerate(zip(
                    cv.split(X, y, groups),
                    self.estimators_
                ))
            )

            (
                predictions_list,
                val_ids_list,
                val_indices_list
            ) = map(list, zip(*outputs))

            predictions = np.concatenate(
                cast(List[NDArray], predictions_list)
            )
            print()
            print("predictions",predictions, "shape", predictions.shape)
            val_ids = np.concatenate(cast(List[NDArray], val_ids_list))
            print()
            print("val_ids", val_ids)
            val_indices = np.concatenate(
                cast(List[NDArray], val_indices_list)
            )
            print()
            print("val_indices", val_indices)
            self.k_[val_indices] = val_ids
            print()
            print("self.k_[val_indices]", self.k_[val_indices])
            y_pred_proba[val_indices] = predictions
            print()
            print("y_pred_proba[val_indices]: ", y_pred_proba[val_indices])

            if isinstance(cv, ShuffleSplit):
                # Should delete values indices that
                # are not used during calibration
                print()
                print("on est dans le cas cv = shuffle split")
                self.k_ = self.k_[val_indices]
                print()
                print("self.k_ :", self.k_)
                y_pred_proba = y_pred_proba[val_indices]
                print()
                print("y_pred_proba", y_pred_proba)
                y_enc = y_enc[val_indices]
                print("y_enc", y_enc)
                y = cast(NDArray, y)[val_indices]
                print("y", y)

        return y_pred_proba, y, y_enc

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_enc: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> EnsembleClassifier:
        """
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold conformity scores are stored under
        the ``conformity_scores_`` attribute.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples,)
            Input labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        EnsembleRegressor
            The estimator fitted.
        """
        print()
        print("EC : USE OF FIT")
        # Initialization
        single_estimator_: ClassifierMixin
        estimators_: List[ClassifierMixin] = []
        full_indexes = np.arange(_num_samples(X))
        print()
        print("full_indexes", full_indexes)
        cv = self.cv
        print()
        print("cv", cv)
        self.use_split_method_ = check_no_agg_cv(X, self.cv, self.no_agg_cv_)
        print()
        print("self.use_split_method_",self.use_split_method_)
        estimator = self.estimator
        print()
        print("estimator: ", estimator)
        n_samples = _num_samples(y)
        print()
        print("n_samples", n_samples)

        # Computation
        if cv == "prefit":
            single_estimator_ = estimator
            self.k_ = np.full(
                shape=(n_samples, 1), fill_value=np.nan, dtype=float
            )
        else:
            single_estimator_ = self._fit_oof_estimator(
                clone(estimator),
                X,
                y,
                full_indexes,
                sample_weight,
                **fit_params
            )
            print()
            print("single_estimator_ :",single_estimator_)
            cv = cast(BaseCrossValidator, cv)
            print()
            print("cv: ", cv)
            self.k_ = np.empty_like(y, dtype=int)
            print("self.k_", self.k_, "shape_of k_ :", self.k_.shape, "unique_values :", np.unique(self.k_.shape))
            estimators_ = Parallel(self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_oof_estimator)(
                    clone(estimator),
                    X,
                    y_enc,
                    train_index,
                    sample_weight,
                    **fit_params
                )
                for train_index, _ in cv.split(X, y, groups)
            )
            # In split-CP, we keep only the model fitted on train dataset
            if self.use_split_method_:
                single_estimator_ = estimators_[0]
        self.single_estimator_ = single_estimator_
        print()
        print("self.single_estimator_", self.single_estimator_)
        self.estimators_ = estimators_
        print()
        print("self.estimators_", self.estimators_)

        return self

    def predict(
        self,
        X: ArrayLike,
        agg_scores
    ) -> Union[NDArray, Tuple[NDArray, NDArray, NDArray]]:
        """
        Predict target from X. It also computes the prediction per train sample
        for each test sample according to ``self.method``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If ``False``, predictions are those of the model trained on the
            whole training set.
            If ``True``, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If ``cv`` is ``"prefit"`` or ``"split"``, ``ensemble`` is ignored.

            By default ``False``.

        return_multi_pred: bool
            If ``True`` the method returns the predictions and the multiple
            predictions (3 arrays). If ``False`` the method return the
            simple predictions only.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray]
            - Predictions
            - The multiple predictions for the lower bound of the intervals.
            - The multiple predictions for the upper bound of the intervals.
        """
        print()
        print("EC : use of predict")
        check_is_fitted(self, self.fit_attributes)

        if self.cv == "prefit":
            y_pred_proba = self.single_estimator_.predict_proba(X)
        else:
            y_pred_proba_k = np.asarray(
                Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose
                )(
                    delayed(self._predict_proba_oof_estimator)(estimator, X)
                    for estimator in self.estimators_
                )
            )

            if agg_scores == "crossval":
                y_pred_proba = np.moveaxis(y_pred_proba_k[self.k_], 0, 2)
            elif agg_scores == "mean":
                y_pred_proba = np.mean(y_pred_proba_k, axis=0)
            else:
                raise ValueError("Invalid 'agg_scores' argument.")
        # y_pred_proba = self._check_proba_normalized(y_pred_proba, axis=1)

        return y_pred_proba
