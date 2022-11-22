from typing import Optional, Union, cast, Tuple
import warnings
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone, RegressorMixin
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state, check_consistent_length
from sklearn.utils.multiclass import type_of_target


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
        cv: Optional[Union[int, str, BaseCrossValidator]] = "split",
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.calibration_method = calibration_method
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
        correct_label = np.where(top_class_prob_arg_.ravel() == item)[0]
        y_calib_ = np.equal(y_calib[correct_label], item).astype(int)
        label_top_class_prob_ = top_class_prob_[correct_label]
        calibrator_ = fit_estimator(
            calibrator_, label_top_class_prob_, y_calib_
        )
        return calibrator_

    def _check_calibration_method(
        self,
        calibration_method: Union[str, RegressorMixin]
    ) -> Union[str, RegressorMixin]:
        if calibration_method is None:
            calibration_method = "sigmoid"
        if (
            isinstance(calibration_method, str) and
            calibration_method not in self.valid_calibrators.keys()
        ):
            raise ValueError(
                "Please provide a valid string from the valid calibrators."
            )
        return calibration_method

    def _get_labels(
        self, method: str,
        pred: NDArray
    ) -> Tuple[NDArray, NDArray]:
        if method == "top_label":
            max_class_prob = np.max(pred, axis=1).reshape(-1, 1)
            max_class_prob_arg = (np.argmax(pred, axis=1)+1).reshape(-1, 1)
        else:
            raise ValueError("No other methods have been implemented yet.")
        check_consistent_length(max_class_prob, max_class_prob_arg)
        return max_class_prob, max_class_prob_arg

    def _get_calibrator(
        self,
        calibration_method: Union[str, RegressorMixin]
    ) -> RegressorMixin:
        if calibration_method in self.valid_calibrators.keys():
            calibrator = self.valid_calibrators[calibration_method]
        else:
            calibrator = calibration_method
        check_estimator_fit_predict(calibrator)
        return calibrator

    def _pred_proba_calib(
        self,
        item: int,
        calibrated_values: NDArray,
        max_prob: NDArray,
        max_prob_arg: NDArray,
        calibrators: RegressorMixin,
    ) -> NDArray:
        correct_label = np.where(max_prob_arg.ravel() == item)[0].ravel()
        if item not in calibrators:
            calibrated_values[correct_label, item-1] = max_prob[correct_label].ravel()
            warnings.warn(
                "WARNING: This calibration was not previously seen"
                + " and therefore scores will remain unchanged."
            )
        else:
            calibrator_ = calibrators[item]
            calibrated_values[correct_label, item-1] = calibrator_.predict(
                max_prob[correct_label]
            )
        return calibrated_values

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        X_calib: Optional[ArrayLike] = None,
        y_calib: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ):
        cv = check_cv(self.cv)
        self.calibration_method = self._check_calibration_method(
            self.calibration_method
            )
        estimator = check_estimator_classification(X, y, self.estimator)
        X, y = indexable(X, y)
        y = _check_y(y)
        assert type_of_target(y) in ["multiclass", "binary"]
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)
        self.n_classes_ = len(np.unique(y))
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        random_state = check_random_state(random_state)

        assert cv == "split"
        results = check_calib_set(
            X,
            y,
            sample_weight=sample_weight,
            X_calib=X_calib,
            y_calib=y_calib,
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

        self.estimator = fit_estimator(
            clone(estimator), X_train, y_train, sample_weight,
        )

        self.n_classes_ = len(np.unique(y))
        y_pred_calib = self.estimator.predict_proba(X=X_calib)
        calibrator = self._get_calibrator(self.calibration_method)
        max_prob, max_prob_arg = self._get_labels(
            cast(str, self.method),
            y_pred_calib
        )

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
        return self

    def predict_proba(
        self,
        X: ArrayLike,
    ):
        check_is_fitted(self, self.fit_attributes)
        y_pred_calib = self.estimator.predict_proba(X=X)
        self.uncalib_pred = y_pred_calib

        max_prob, max_prob_arg = self._get_labels(
            cast(str, self.method),
            y_pred_calib
        )

        n = _num_samples(max_prob)
        calibrated_test_values = np.zeros((n, self.n_classes_))

        for item in np.unique(max_prob_arg):
            calibrated_test_values = self._pred_proba_calib(
                item,
                calibrated_test_values,
                max_prob,
                max_prob_arg,
                self.calibrators,
            )
        return calibrated_test_values
