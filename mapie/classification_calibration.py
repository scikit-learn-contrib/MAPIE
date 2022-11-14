from __future__ import annotations
from typing import Any, Optional, Union, Tuple, Iterable, List, cast

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils.multiclass import type_of_target, check_classification_targets
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _compute_predictions, _SigmoidCalibration

from sklearn.utils.validation import (
    indexable,
    _check_sample_weight,
    check_consistent_length,
    check_is_fitted,
    _num_samples,
    _check_y,
)

from ._machine_precision import EPSILON
from .metrics import classification_mean_width_score
from ._typing import ArrayLike, NDArray
from .utils import (
    check_cv,
    check_null_weight,
    check_n_features_in,
    check_alpha,
    check_alpha_and_n_samples,
    check_n_jobs,
    check_verbose,
    compute_quantiles,
    fit_estimator,
    check_calib_set,
)


class MapieCalibrator(BaseEstimator, ClassifierMixin):


    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        class_method: str = "top_label",
        calibration_method: str = "sigmoid",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
    ):
        estimator = self.estimator,
        class_method = self.class_method,
        calibration_method = self.calibration_method,
        cv = self.cv,


    def fit(        
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ):
        cv = check_cv(self.cv)
        estimator = self._check_estimator(X, y, self.estimator)
        X, y = indexable(X, y)
        y = _check_y(y)

        assert type_of_target(y) == "multiclass"

        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)
        n_samples = _num_samples(y)
        self.n_classes_ = len(np.unique(y))
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        random_state = check_random_state(random_state)

        if cv == "prefit":
            pass
        elif cv == "split":
            results = check_calib_set(
                X,
                y,
                sample_weight,
                calib_size,
                random_state,
                shuffle,
                stratify,
            )
            X_train, y_train, X_calib, y_calib, sample_weight_train = results
            X_train, y_train = indexable(X_train, y_train)
            X_calib, y_calib = indexable(X_calib, y_calib)
            y_train, y_calib = _check_y(y_train), _check_y(y_calib)
            self.n_calib_samples = _num_samples(y_calib)
            sample_weight_train, X_train, y_train = check_null_weight(
                sample_weight_train,
                X_train,
                y_train
            )
            y_train = cast(NDArray, y_train)

            estimator_ = fit_estimator(
                clone(estimator), X_train, y_train, sample_weight
            )

            calibrator = _get_calibrator(self.calibrator_method)

            y_pred_calib_ = estimator_.predict_proba(X_calib)
            top_class_prob = np.max(y_pred_calib_, axis=1)
            top_class_prob_arg = np.argmax(y_pred_calib_, axis=1)+1

            calibrators = {}
            for item in np.unique(top_class_prob_arg):
                correct_label = np.where(top_class_prob_arg == item)[0]
                n_l = len(correct_label)
                bins_l = np.floor(n_l/self.points_per_bin).astype('int')
                if(bins_l == 0):
                    print("Predictions for class {:d} not recalibrated since fewer than {:d} calibration points were predicted as class {:d}.".format(item, points_per_bin, item))
                else:
                    hb_clone = calibrator
                    y_calib_ = (y_calib[correct_label] == item)
                    top_class_prob_ = top_class_prob[correct_label]
                    hb_clone.fit(top_class_prob_, y_calib_)
                    calibrators[item] = hb_clone

            self.calibrators = calibrators

    

def _get_calibrator(calibrator_method: str):
    if calibrator_method == "sigmoid":
        calibrator = _SigmoidCalibration()
    elif calibrator_method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
    return calibrator