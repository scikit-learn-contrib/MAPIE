from __future__ import annotations
from typing import Optional, Union, cast, Tuple
import warnings

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator, ClassifierMixin, clone, RegressorMixin
from sklearn.utils import check_random_state, check_consistent_length
from sklearn.utils.multiclass import type_of_target
from sklearn.calibration import _compute_predictions, _get_prediction_method
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration

from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    _check_y,
    _num_samples,
)

from ._typing import ArrayLike, NDArray
from .utils import (
    check_cv,
    check_null_weight,
    check_n_features_in,
    check_calib_set,
    fit_estimator,
    check_estimator_classification,
    check_estimator_fit_predict,
)


class MapieCalibrator(BaseEstimator, ClassifierMixin):

    fit_attributes = [
        "estimator",
        "calibrators",
        "max_prob",
        "max_prob_arg",
    ]

    valid_calibrators = {
        "sigmoid": _SigmoidCalibration(),
        "isotonic": IsotonicRegression(out_of_bounds="clip")
    }

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "top_label",
        calibration_method: Optional[Union[str, RegressorMixin]] = None,
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
    ) -> None:
        self.estimator = estimator,
        self.method = method,
        self.calibration_method = calibration_method,
        self.cv = cv

    def _fit_calibrator(
        self,
        item: int,
        calibrator: RegressorMixin,
        y_calib: NDArray,
        top_class_prob_: NDArray,
        top_class_prob_arg_: NDArray,
    ) -> RegressorMixin:
        calibrator_ = clone(calibrator)
        correct_label = np.array(
            np.where(np.array(top_class_prob_arg_) == item)
        )
        y_calib_ = np.equal(y_calib[correct_label], item).astype(int)
        label_top_class_prob_ = top_class_prob_[correct_label]
        calibrator_.fit_estimator(label_top_class_prob_, y_calib_)
        return calibrator

    def _check_calibration_method(
        self,
        calibration_method: Optional[Union[str, RegressorMixin]]
    ):
        if calibration_method is None:
            calibration_method = "sigmoid"
        elif (
            isinstance(calibration_method, str) and
            calibration_method not in self.valid_calibrator.keys()
        ):
            ValueError(
                "Please provide a valid calibrator."
            )
        elif isinstance(calibration_method, RegressorMixin):
            check_estimator_fit_predict(calibration_method)
        return calibration_method

    def _get_labels(
        self, method: str, y: NDArray
    ) -> Tuple[NDArray, NDArray]:
        if method == "top_label":
            max_class_prob = np.max(y, axis=1).reshape(-1, 1)
            max_class_prob_arg = (np.argmax(y, axis=1)+1).reshape(-1, 1)
        else:
            ValueError("No other methods have been implemented yet.")
        check_consistent_length(max_class_prob, max_class_prob_arg)
        return max_class_prob, max_class_prob_arg

    def _get_calibrator(
        self,
        calibration_method: Optional[Union[str, RegressorMixin]]
    ):
        if calibration_method in self.valid_calibrator.keys():
            calibrator = self.valid_calibrator[calibration_method]
        elif isinstance(calibration_method, RegressorMixin):
            calibrator = calibration_method
        else:
            ValueError(
                "Please provide a valid calibrator or input a valid"
                + " regressor to be used as a calibrator."
            )
        return calibrator

    def _pred_proba_calib(
        self, item, new_values, max_prob, max_prob_arg, calibrators,
    ) -> NDArray:
        correct_label = np.array(
            np.where(np.array(max_prob_arg) == item)
        )
        if item not in calibrators:
            new_values[correct_label] = max_prob[correct_label]
            warnings.warn(
                "WARNING: This calibration was not previously seen"
                + " and therefore scores will remain unchanged"
            )
        else:
            calibrator_ = calibrators[item]
            new_values[correct_label] = calibrator_.predict_proba(
                max_prob[correct_label]
            )
        return new_values

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
        self.method = self._check_method(cast(str, self.method))
        self.calibration_method = self._check_calibration_method(
            self.calibration_method
            )
        estimator = check_estimator_classification(X, y, self.estimator)
        X, y = indexable(X, y)
        y = _check_y(y)
        assert type_of_target(y) == "multiclass"
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)
        self.n_classes_ = len(np.unique(y))
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        random_state = check_random_state(random_state)

        if cv == "prefit":
            pass
        elif cv == "split":
            results = check_calib_set(
                X,
                y,
                sample_weight=sample_weight,
                calib_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            )
            X_train, y_train, X_calib, y_calib, sample_weight_train = results
            X_train, y_train = indexable(X_train, y_train)
            X_calib, y_calib = indexable(X_calib, y_calib)
            y_train, y_calib = _check_y(y_train), _check_y(y_calib)
            sample_weight_train, X_train, y_train = check_null_weight(
                sample_weight_train,
                X_train,
                y_train
            )
            y_train = cast(NDArray, y_train)

            estimator_ = fit_estimator(
                clone(estimator), X_train, y_train, sample_weight,
            )
            self.estimator = estimator_

            self.n_classes_ = len(np.unique(y))
            pred_method, method_name = _get_prediction_method(estimator_)
            y_pred_calib = _compute_predictions(
                pred_method,
                method_name,
                X_calib,
                self.n_classes_
            )

            calibrator = self._get_calibrator(self.calibration_method)
            max_prob, max_prob_arg = self._get_labels(
                cast(str, self.method),
                y_pred_calib
            )
            self.max_prob = max_prob
            self.max_prob_arg = max_prob_arg

            calibrators = {}
            for item in np.unique(max_prob_arg):
                calibrator_ = self._fit_calibrator(
                    item,
                    calibrator,
                    cast(NDArray, y_calib),
                    max_prob,
                    max_prob_arg
                )
                calibrators[item] = calibrator_
            self.calibrators = calibrators

    def predict_proba(
        self,
        X: ArrayLike,
    ):
        check_is_fitted(self, self.fit_attributes)
        pred_method, method_name = _get_prediction_method(self.estimator_)
        y_pred_calib = _compute_predictions(
            pred_method,
            method_name,
            X,
            self.n_classes_
        )

        max_prob, max_prob_arg = self._get_labels(
            cast(str, self.method),
            y_pred_calib
        )

        n = _num_samples(max_prob)
        calibrated_test_values = np.zeros((n))

        for item in np.unique(max_prob_arg):
            calibrated_test_values = self._pred_proba_calib(
                item,
                calibrated_test_values,
                max_prob,
                max_prob_arg,
                self.calibrators,
            )
        return calibrated_test_values
