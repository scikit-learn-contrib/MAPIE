###TEST####

from __future__ import annotations
import warnings

from typing import Any, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils.multiclass import (check_classification_targets,
                                      type_of_target)
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)
from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray
from mapie.aggregation_functions import aggregate_all, phi2D
from mapie.metrics import classification_mean_width_score
from mapie.estimator.interface import EnsembleEstimator
from mapie.utils import (check_nan_in_aposteriori_prediction, check_no_agg_cv,
                         fit_estimator)
from mapie.utils import (check_alpha, check_alpha_and_n_samples, check_cv,
                    check_estimator_classification, check_n_features_in,
                    check_n_jobs, check_null_weight, check_verbose,
                    compute_quantiles, fit_estimator, fix_number_of_classes)


class EnsembleClassifier(EnsembleEstimator):
    
    raps_valid_cv_ = ["prefit", "split"]
    valid_methods_ = [
        "naive", "score", "lac", "cumulated_score", "aps", "top_k", "raps"
    ]    
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "k_",
        "n_features_in_",
        "conformity_scores_",
        "classes_",
        "label_encoder_"
    ]

    #TODO : dans le paragraphe init, pas sûr de garder les "None" par défaut présent dans MapieClassifier
    def __init__(
        self,
        estimator: Optional[ClassifierMixin]= None,
        method: str = "lac",
        cv: Optional[Union[int, str, BaseCrossValidator]]= None,
        agg_function: Optional[str] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        test_size: Optional[Union[int, float]] = None,
        verbose: int =0
    )--> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.test_size = test_size
        self.verbose = verbose

    @staticmethod
    def _fit_and_predict_oof_estimator(
        self,
        estimator: ClassifierMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        val_index: ArrayLike,
        k:int,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params,
    ) -> Tuple[ClassifierMixin, NDArray, NDArray, ArrayLike]:
        
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        X_val = _safe_indexing(X, val_index)
        y_val = _safe_indexing(y, val_index)

        if sample_weight is None:
            estimator = fit_estimator(
                estimator, X_train, y_train, **fit_params
            )
        else:
            sample_weight_train = _safe_indexing(sample_weight, train_index)
            estimator = fit_estimator(
                estimator, X_train, y_train, sample_weight_train, **fit_params
            )
        if _num_samples(X_val) > 0:
            y_pred_proba = self._predict_oof_model(estimator, X_val)
        else:
            y_pred_proba = np.array([])
        val_id = np.full_like(y_val, k, dtype=int)
        return estimator, y_pred_proba, val_id, val_index

    @staticmethod
    def _predict_oof_model(
        self,
        estimator: ClassifierMixin,
        X: ArrayLike,
    ) -> NDArray:
        """
        Predict probabilities of a test set from a fitted estimator.

        Parameters
        ----------
        estimator: ClassifierMixin
            Fitted estimator.

        X: ArrayLike
            Test set.

        Returns
        -------
        ArrayLike
            Predicted probabilities.
        """
        y_pred_proba = estimator.predict_proba(X)
        # we enforce y_pred_proba to contain all labels included in y
        if len(estimator.classes_) != self.n_classes_:
            y_pred_proba = fix_number_of_classes(
                self.n_classes_,
                estimator.classes_,
                y_pred_proba
            )
        y_pred_proba = self._check_proba_normalized(y_pred_proba)
        return y_pred_proba

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
        y_pred_multi = np.column_stack(
            [e.predict(X) for e in self.estimators_]
        )
        # At this point, y_pred_multi is of shape
        # (n_samples_test, n_estimators_). The method
        # ``_aggregate_with_mask`` fits it to the right size
        # thanks to the shape of k_.
        y_pred_multi = self._aggregate_with_mask(y_pred_multi, self.k_)
        return y_pred_multi

    def predict_calib(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
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
        check_is_fitted(self, self.fit_attributes)

        if self.cv == "prefit":
            y_pred = self.single_estimator_.predict(X)
        else:
            if self.method == "naive":
                y_pred = self.single_estimator_.predict(X)
            else:
                cv = cast(BaseCrossValidator, self.cv)
                outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._predict_oof_estimator)(
                        estimator, X, calib_index,
                    )
                    for (_, calib_index), estimator in zip(
                        cv.split(X, y, groups),
                        self.estimators_
                    )
                )
                predictions, indices = map(
                    list, zip(*outputs)
                )
                n_samples = _num_samples(X)
                pred_matrix = np.full(
                    shape=(n_samples, cv.get_n_splits(X, y, groups)),
                    fill_value=np.nan,
                    dtype=float,
                )
                for i, ind in enumerate(indices):
                    pred_matrix[ind, i] = np.array(
                        predictions[i], dtype=float
                    )
                    self.k_[ind, i] = 1
                check_nan_in_aposteriori_prediction(pred_matrix)

                y_pred = aggregate_all(self.agg_function, pred_matrix)

        return y_pred

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        size_raps: Optional[float] = .2,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> EnsembleClassifier:
        # Checks

        self._check_parameters()
        cv = check_cv(
            self.cv, test_size=self.test_size, random_state=self.random_state
        )
        X, y = indexable(X, y)
        y = _check_y(y)

        sample_weight = cast(Optional[NDArray], sample_weight)
        groups = cast(Optional[NDArray], groups)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        y = cast(NDArray, y)

        estimator = check_estimator_classification(
            X,
            y,
            cv,
            self.estimator
        )
        self.n_features_in_ = check_n_features_in(X, cv, estimator)

        n_samples = _num_samples(y)

        self.n_classes_, self.classes_ = self._get_classes_info(
            estimator, y
        )
        enc = LabelEncoder()
        enc.fit(self.classes_)
        y_enc = enc.transform(y)

        self.label_encoder_ = enc
        self._check_target(y)

        # Initialization
        self.estimators_: List[ClassifierMixin] = []
        self.k_ = np.empty_like(y, dtype=int)
        self.n_samples_ = _num_samples(X)

        if self.method == "raps":
            raps_split = ShuffleSplit(
                1, test_size=size_raps, random_state=self.random_state
            )
            train_raps_index, val_raps_index = next(raps_split.split(X))
            X, self.X_raps, y_enc, self.y_raps = \
                _safe_indexing(X, train_raps_index), \
                _safe_indexing(X, val_raps_index), \
                _safe_indexing(y_enc, train_raps_index), \
                _safe_indexing(y_enc, val_raps_index)
            self.y_raps_no_enc = self.label_encoder_.inverse_transform(
                self.y_raps
            )
            y = self.label_encoder_.inverse_transform(y_enc)
            y_enc = cast(NDArray, y_enc)
            n_samples = _num_samples(y_enc)
            if sample_weight is not None:
                sample_weight = sample_weight[train_raps_index]
                sample_weight = cast(NDArray, sample_weight)
            if groups is not None:
                groups = groups[train_raps_index]
                groups = cast(NDArray, groups)

        # Work
        if cv == "prefit":
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba = self._check_proba_normalized(y_pred_proba)

        else:
            cv = cast(BaseCrossValidator, cv)
            self.single_estimator_ = fit_estimator(
                clone(estimator), X, y, sample_weight, **fit_params
            )
            y_pred_proba = np.empty(
                (n_samples, self.n_classes_),
                dtype=float
            )
            outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_and_predict_oof_model)(
                    clone(estimator),
                    X,
                    y,
                    train_index,
                    val_index,
                    k,
                    sample_weight,
                    **fit_params,
                )
                for k, (train_index, val_index) in enumerate(
                    cv.split(X, y_enc, groups)
                )
            )
            (
                self.estimators_,
                predictions_list,
                val_ids_list,
                val_indices_list
            ) = map(list, zip(*outputs))
            predictions = np.concatenate(
                cast(List[NDArray], predictions_list)
            )
            val_ids = np.concatenate(cast(List[NDArray], val_ids_list))
            val_indices = np.concatenate(
                cast(List[NDArray], val_indices_list)
            )
            self.k_[val_indices] = val_ids
            y_pred_proba[val_indices] = predictions

            if isinstance(cv, ShuffleSplit):
                # Should delete values indices that
                # are not used during calibration
                self.k_ = self.k_[val_indices]
                y_pred_proba = y_pred_proba[val_indices]
                y_enc = y_enc[val_indices]
                y = cast(NDArray, y)[val_indices]

        # RAPS: compute y_pred and position on the RAPS validation dataset
        if self.method == "raps":
            self.y_pred_proba_raps = self.single_estimator_.predict_proba(
                self.X_raps
            )
            self.position_raps = self._get_true_label_position(
                self.y_pred_proba_raps,
                self.y_raps
            )

        # Conformity scores
        if self.method == "naive":
            self.conformity_scores_ = np.empty(
                y_pred_proba.shape,
                dtype="float"
            )
        elif self.method in ["score", "lac"]:
            self.conformity_scores_ = np.take_along_axis(
                1 - y_pred_proba, y_enc.reshape(-1, 1), axis=1
            )
        elif self.method in ["cumulated_score", "aps", "raps"]:
            self.conformity_scores_, self.cutoff = (
                self._get_true_label_cumsum_proba(
                    y,
                    y_pred_proba
                )
            )
            y_proba_true = np.take_along_axis(
                y_pred_proba, y_enc.reshape(-1, 1), axis=1
            )
            random_state = check_random_state(self.random_state)
            u = random_state.uniform(size=len(y_pred_proba)).reshape(-1, 1)
            self.conformity_scores_ -= u * y_proba_true
        elif self.method == "top_k":
            # Here we reorder the labels by decreasing probability
            # and get the position of each label from decreasing
            # probability
            self.conformity_scores_ = self._get_true_label_position(
                y_pred_proba,
                y_enc
            )
        else:
            raise ValueError(
                "Invalid method. "
                f"Allowed values are {self.valid_methods_}."
            )

        if isinstance(cv, ShuffleSplit):
            self.single_estimator_ = self.estimators_[0]

        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        include_last_label: Optional[Union[bool, str]] = True,
        agg_scores: Optional[str] = "mean"
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:

        if self.method == "top_k":
            agg_scores = "mean"
        # Checks
        cv = check_cv(
            self.cv, test_size=self.test_size, random_state=self.random_state
        )
        include_last_label = self._check_include_last_label(include_last_label)
        alpha = cast(Optional[NDArray], check_alpha(alpha))
        check_is_fitted(self, self.fit_attributes)
        lambda_star, k_star = None, None
        # Estimate prediction sets
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return y_pred

        n = len(self.conformity_scores_)

        # Estimate of probabilities from estimator(s)
        # In all cases: len(y_pred_proba.shape) == 3
        # with  (n_test, n_classes, n_alpha or n_train_samples)
        alpha_np = cast(NDArray, alpha)
        check_alpha_and_n_samples(alpha_np, n)
        if cv == "prefit":
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba = np.repeat(
                y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
            )
        else:
            y_pred_proba_k = np.asarray(
                Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose
                )(
                    delayed(self._predict_oof_model)(estimator, X)
                    for estimator in self.estimators_
                )
            )
            if agg_scores == "crossval":
                y_pred_proba = np.moveaxis(y_pred_proba_k[self.k_], 0, 2)
            elif agg_scores == "mean":
                y_pred_proba = np.mean(y_pred_proba_k, axis=0)
                y_pred_proba = np.repeat(
                    y_pred_proba[:, :, np.newaxis], len(alpha_np), axis=2
                )
            else:
                raise ValueError("Invalid 'agg_scores' argument.")
        # Check that sum of probas is equal to 1
        y_pred_proba = self._check_proba_normalized(y_pred_proba, axis=1)

        # Choice of the quantile
        check_alpha_and_n_samples(alpha_np, n)

        if self.method == "naive":
            self.quantiles_ = 1 - alpha_np
        else:
            if (cv == "prefit") or (agg_scores in ["mean"]):
                if self.method == "raps":
                    check_alpha_and_n_samples(alpha_np, len(self.X_raps))
                    k_star = compute_quantiles(
                        self.position_raps,
                        alpha_np
                    ) + 1
                    y_pred_proba_raps = np.repeat(
                        self.y_pred_proba_raps[:, :, np.newaxis],
                        len(alpha_np),
                        axis=2
                    )
                    lambda_star = self._find_lambda_star(
                        y_pred_proba_raps,
                        alpha_np,
                        include_last_label,
                        k_star
                    )
                    self.conformity_scores_regularized = (
                        self._regularize_conformity_score(
                                    k_star,
                                    lambda_star,
                                    self.conformity_scores_,
                                    self.cutoff
                        )
                    )
                    self.quantiles_ = compute_quantiles(
                        self.conformity_scores_regularized,
                        alpha_np
                    )
                else:
                    self.quantiles_ = compute_quantiles(
                        self.conformity_scores_,
                        alpha_np
                    )
            else:
                self.quantiles_ = (n + 1) * (1 - alpha_np)

        # Build prediction sets
        if self.method in ["score", "lac"]:
            if (cv == "prefit") or (agg_scores == "mean"):
                prediction_sets = np.greater_equal(
                    y_pred_proba - (1 - self.quantiles_), -EPSILON
                )
            else:
                y_pred_included = np.less_equal(
                    (1 - y_pred_proba) - self.conformity_scores_.ravel(),
                    EPSILON
                ).sum(axis=2)
                prediction_sets = np.stack(
                    [
                        np.greater_equal(
                            y_pred_included - _alpha * (n - 1), -EPSILON
                        )
                        for _alpha in alpha_np
                    ], axis=2
                )

        elif self.method in ["naive", "cumulated_score", "aps", "raps"]:
            # specify which thresholds will be used
            if (cv == "prefit") or (agg_scores in ["mean"]):
                thresholds = self.quantiles_
            else:
                thresholds = self.conformity_scores_.ravel()
            # sort labels by decreasing probability
            y_pred_proba_cumsum, y_pred_index_last, y_pred_proba_last = (
                self._get_last_included_proba(
                    y_pred_proba,
                    thresholds,
                    include_last_label,
                    lambda_star,
                    k_star,
                )
            )
            # get the prediction set by taking all probabilities
            # above the last one
            if (cv == "prefit") or (agg_scores in ["mean"]):
                y_pred_included = np.greater_equal(
                    y_pred_proba - y_pred_proba_last, -EPSILON
                )
            else:
                y_pred_included = np.less_equal(
                    y_pred_proba - y_pred_proba_last, EPSILON
                )
            # remove last label randomly
            if include_last_label == "randomized":
                y_pred_included = self._add_random_tie_breaking(
                    y_pred_included,
                    y_pred_index_last,
                    y_pred_proba_cumsum,
                    y_pred_proba_last,
                    thresholds,
                    lambda_star,
                    k_star
                )
            if (cv == "prefit") or (agg_scores in ["mean"]):
                prediction_sets = y_pred_included
            else:
                # compute the number of times the inequality is verified
                prediction_sets_summed = y_pred_included.sum(axis=2)
                prediction_sets = np.less_equal(
                    prediction_sets_summed[:, :, np.newaxis]
                    - self.quantiles_[np.newaxis, np.newaxis, :],
                    EPSILON
                )
        elif self.method == "top_k":
            y_pred_proba = y_pred_proba[:, :, 0]
            index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
            y_pred_index_last = np.stack(
                [
                    index_sorted[:, quantile]
                    for quantile in self.quantiles_
                ], axis=1
            )
            y_pred_proba_last = np.stack(
                [
                    np.take_along_axis(
                        y_pred_proba,
                        y_pred_index_last[:, iq].reshape(-1, 1),
                        axis=1
                    )
                    for iq, _ in enumerate(self.quantiles_)
                ], axis=2
            )
            prediction_sets = np.greater_equal(
                y_pred_proba[:, :, np.newaxis]
                - y_pred_proba_last,
                -EPSILON
            )
        else:
            raise ValueError(
                "Invalid method. "
                f"Allowed values are {self.valid_methods_}."
            )
        return y_pred, prediction_sets

