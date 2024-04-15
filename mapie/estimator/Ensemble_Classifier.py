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
from mapie._typing import ArrayLike, NDArray
from mapie.aggregation_functions import aggregate_all, phi2D
from mapie.estimator.interface import EnsembleEstimator
from mapie.utils import (check_nan_in_aposteriori_prediction, check_no_agg_cv,
                         fit_estimator)


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

    #TODO : dans le paragraphe init, pas sûr de garder les "None" par défaut
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
        self.agg_function = agg_function # TODO : à voir si je garde l'argument (pas présent dans MapieClassifier)
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
        #TODO : reprendre ici
        if not (sample_weight is None):
            sample_weight = _safe_indexing(sample_weight, train_index)
            sample_weight = cast(NDArray, sample_weight)

        estimator = fit_estimator(
            estimator,
            X_train,
            y_train,
            sample_weight=sample_weight,
            **fit_params
        )
        return estimator

    @staticmethod
    def _predict_oof_estimator(
        estimator: RegressorMixin,
        X: ArrayLike,
        val_index: ArrayLike,
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
        X_val = _safe_indexing(X, val_index)
        if _num_samples(X_val) > 0:
            y_pred = estimator.predict(X_val)
        else:
            y_pred = np.array([])
        return y_pred, val_index

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
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> EnsembleRegressor:
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
        # Initialization
        single_estimator_: RegressorMixin
        estimators_: List[RegressorMixin] = []
        full_indexes = np.arange(_num_samples(X))
        cv = self.cv
        self.use_split_method_ = check_no_agg_cv(X, self.cv, self.no_agg_cv_)
        estimator = self.estimator
        n_samples = _num_samples(y)

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
            cv = cast(BaseCrossValidator, cv)
            self.k_ = np.full(
                shape=(n_samples, cv.get_n_splits(X, y, groups)),
                fill_value=np.nan,
                dtype=float,
            )
            if self.method == "naive":
                estimators_ = [single_estimator_]
            else:
                estimators_ = Parallel(self.n_jobs, verbose=self.verbose)(
                    delayed(self._fit_oof_estimator)(
                        clone(estimator),
                        X,
                        y,
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
        self.estimators_ = estimators_

        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        return_multi_pred: bool = True
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
        check_is_fitted(self, self.fit_attributes)

        y_pred = self.single_estimator_.predict(X)
        if not return_multi_pred and not ensemble:
            return y_pred

        if self.method in self.no_agg_methods_ or self.use_split_method_:
            y_pred_multi_low = y_pred[:, np.newaxis]
            y_pred_multi_up = y_pred[:, np.newaxis]
        else:
            y_pred_multi = self._pred_multi(X)

            if self.method == "minmax":
                y_pred_multi_low = np.min(y_pred_multi, axis=1, keepdims=True)
                y_pred_multi_up = np.max(y_pred_multi, axis=1, keepdims=True)
            elif self.method == "plus":
                y_pred_multi_low = y_pred_multi
                y_pred_multi_up = y_pred_multi
            else:
                y_pred_multi_low = y_pred[:, np.newaxis]
                y_pred_multi_up = y_pred[:, np.newaxis]

            if ensemble:
                y_pred = aggregate_all(self.agg_function, y_pred_multi)

        if return_multi_pred:
            return y_pred, y_pred_multi_low, y_pred_multi_up
        else:
            return y_pred
