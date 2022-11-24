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
    check_binary_zero_one,
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
        calibrator: Optional[Union[str, RegressorMixin]] = None,
        cv: Optional[Union[int, str, BaseCrossValidator]] = "split",
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.calibrator = calibrator
        self.cv = cv

    def _fit_calibrator(
        self,
        item: int,
        calibrator: RegressorMixin,
        y_calib: NDArray,
        top_class_prob: NDArray,
        top_class_prob_arg: NDArray,
        sample_weight: Optional[NDArray],
    ) -> RegressorMixin:
        calibrator_ = clone(calibrator)
        correct_label = np.where(top_class_prob_arg.ravel() == item)[0]
        y_calib_ = check_binary_zero_one(
            np.equal(y_calib[correct_label], item).astype(int)
        )
        top_class_prob_ = top_class_prob[correct_label]

        if sample_weight is not None:
            sample_weight_ = sample_weight[correct_label]
            (
                sample_weight_, top_class_prob_, y_calib_
            ) = check_null_weight(  # type: ignore
                sample_weight_,
                top_class_prob_,
                y_calib_
            )
        else:
            sample_weight_ = sample_weight
        calibrator_ = fit_estimator(
            calibrator_, top_class_prob_, y_calib_, sample_weight_
        )
        return calibrator_

    def _check_calibrator(
        self,
        calibrator: Union[str, RegressorMixin]
    ) -> Union[str, RegressorMixin]:
        if calibrator is None:
            calibrator = "sigmoid"
        if (
            isinstance(calibrator, str) and
            calibrator not in self.valid_calibrators.keys()
        ):
            raise ValueError(
                "Please provide a valid string from the valid calibrators."
            )
        return calibrator

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
        calibrator: Union[str, RegressorMixin]
    ) -> RegressorMixin:
        if calibrator in self.valid_calibrators.keys():
            calibrator = self.valid_calibrators[calibrator]
        else:
            calibrator = calibrator
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
            calibrated_values[
                correct_label, item-1
                ] = max_prob[correct_label].ravel()
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
        sample_weight: Optional[NDArray] = None,
        X_calib: Optional[ArrayLike] = None,
        y_calib: Optional[ArrayLike] = None,
        sample_weight_calib: Optional[NDArray] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ):
        cv = check_cv(self.cv)
        self.calibrator = self._check_calibrator(
            self.calibrator
            )
        estimator = check_estimator_classification(X, y, self.estimator)
        X, y = indexable(X, y)
        y = _check_y(y)
        assert type_of_target(y) in ["multiclass", "binary"]
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)
        sample_weight = cast(Optional[NDArray], sample_weight)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        random_state = check_random_state(random_state)

        assert cv == "split"
        results = check_calib_set(
            X,
            y,
            sample_weight=sample_weight,
            X_calib=X_calib,
            y_calib=y_calib,
            sample_weight_calib=cast(Optional[NDArray], sample_weight_calib),
            calib_size=calib_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )
        X_train, y_train, X_calib, y_calib, sw_train, sw_calib = results
        X_train, y_train = indexable(X_train, y_train)
        X_calib, y_calib = indexable(X_calib, y_calib)
        y_train, y_calib = _check_y(y_train), _check_y(y_calib)
        sw_train, X_train, y_train = check_null_weight(
            sw_train,
            X_train,
            y_train
        )
        y_train = cast(NDArray, y_train)
        self.n_classes_ = len(np.unique(y_train))

        self.estimator = fit_estimator(
            clone(estimator), X_train, y_train, sw_train,
        )
        y_pred_calib = self.estimator.predict_proba(X=X_calib)
        calibrator = self._get_calibrator(self.calibrator)
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
                max_prob_arg,
                sw_calib,
            )
            calibrators[item] = calibrator_
        self.calibrators = calibrators
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(
        self,
        X: ArrayLike,
    ):
        check_is_fitted(self, self.fit_attributes)
        y_pred_probs = self.estimator.predict_proba(X=X)  # type: ignore
        self.uncalib_pred = y_pred_probs

        max_prob, max_prob_arg = self._get_labels(
            cast(str, self.method),
            y_pred_probs
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

    def predict(
        self,
        X: ArrayLike,
    ):
        check_is_fitted(self, self.fit_attributes)
        return self.classes_[np.argmax(self.predict_proba(X=X), axis=1)]
