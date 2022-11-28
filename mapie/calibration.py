from typing import Optional, Union, cast, Tuple, Dict

import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone, RegressorMixin
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
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
        cv: Optional[Union[str, BaseCrossValidator]] = "split",
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.calibrator = calibrator
        self.cv = cv

    def _fit_calibrator(
        self,
        item: int,
        calibrator: RegressorMixin,
        y_calib: ArrayLike,
        top_class_prob: NDArray,
        top_class_prob_arg: NDArray,
        sample_weight: Optional[ArrayLike],
    ) -> RegressorMixin:
        """
        Fitting the calibrator requires that for each class we
        get the correct values to fit a calibrator for this specific
        class.

        Parameters
        ----------
        item : int
            The class for which we will fit a calibrator.
        calibrator : RegressorMixin
            Calibrator to train.
        y_calib : NDArray of shape (n_samples,)
            The dependent values of the calibrator.
        top_class_prob : NDArray of shape (n_samples,)
            The independent values of the calibrator.
        top_class_prob_arg : NDArray of shape (n_samples,)
            The array to determine which class the max prediction belongs to.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        RegressorMixin
            Calibrated estimator.
        """
        calibrator_ = clone(calibrator)
        y_calib = cast(NDArray, y_calib)
        sample_weight = cast(NDArray, sample_weight)
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

    def _get_calibrator(
        self,
        cv: Union[str, BaseCrossValidator],
        calibrator: Optional[Union[str, RegressorMixin]],
    ) -> RegressorMixin:
        """
        Check the input that has been provided for
        calibrator and checks that the calibrator is a correct
        estimator to calibrate.

        Parameters
        ----------
        cv : Union[str, BaseCrossValidator]
            Cross validation parameter.
        calibrator : Union[str, RegressorMixin]
            Calibrator as string to then be linked to one of the
            valid methods otherwise calibrator as estimator.

        Returns
        -------
        RegressorMixin
            RegressorMixin to be used as calibrator.
        Raises
        ------
        ValueError
            If str is not one of the valid estimators.
        """
        if calibrator is None:
            calibrator = "sigmoid"

        if isinstance(calibrator, str):
            if calibrator in self.valid_calibrators.keys():
                calibrator = self.valid_calibrators[calibrator]
            else:
                raise ValueError(
                    "Please provide a valid string from the valid calibrators."
                )
        if isinstance(calibrator, Pipeline):
            est = calibrator[-1]
        else:
            est = calibrator
        check_estimator_fit_predict(est)
        if cv == "prefit":
            check_is_fitted(est)
        return calibrator

    def _get_labels(
        self, method: str,
        pred: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        The "labels" is the way to create the different arrays needed
        for the type of calibration method you want to make.

        - Top-Label method means that you take the maximum probability
        and calibrated each class when it is the maximum separately.

        Parameters
        ----------
        method : str
            The method that will be taken into account.
        pred : NDArray of shape (n_samples, n_classes)
            The output from the predict_proba of the classifier.

        Returns
        -------
        Tuple[NDArray, NDArray] of shapes (n_samples,) and (n_samples,)
            The first element corresponds to the values that have to be
            calibrated and the second element corresponds to class to be
            associated.

            In the Top-Label setting, the latter refers to the different
            classes that have to be individually calibrated.

        Raises
        ------
        ValueError
            If the method provided has not been implemented.
        """
        if method == "top_label":
            max_class_prob = np.max(pred, axis=1).reshape(-1, 1)
            max_class_prob_arg = (np.argmax(pred, axis=1)+1).reshape(-1, 1)
        else:
            raise ValueError("No other methods have been implemented yet.")
        check_consistent_length(max_class_prob, max_class_prob_arg)
        return max_class_prob, max_class_prob_arg

    def _fit_calibrators(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike],
        estimator: Union[ClassifierMixin, Pipeline],
        calibrator: RegressorMixin,
    ) -> Dict[int, RegressorMixin]:
        """_summary_

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
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.
            By default ``None``.
        estimator : ClassifierMixin
            Estimator fitted.
        calibrator : RegressorMixin
            Calibrator to train.

        Returns
        -------
        Dict[int, RegressorMixin]
            Dictionnary of fitted calibrators.
        """
        # X, y = indexable(X, y)
        # y = _check_y(y)
        y_pred_calib = estimator.predict_proba(X=X)
        max_prob, max_prob_arg = self._get_labels(
            cast(str, self.method),
            y_pred_calib
        )
        calibrators = {}
        for item in np.unique(max_prob_arg):
            calibrator_ = self._fit_calibrator(
                item,
                calibrator,
                y,
                max_prob,
                max_prob_arg,
                sample_weight,
            )
            calibrators[item] = calibrator_
        return calibrators

    def _pred_proba_calib(
        self,
        item: int,
        calibrated_values: NDArray,
        max_prob: NDArray,
        max_prob_arg: NDArray,
        calibrators: Dict[int, RegressorMixin],
    ) -> NDArray:
        """
        Using the predicted probabilities, we calibrate new values with
        the fitted calibrators. Note that if there is no calibrator for a
        the specific class, then we simply output the not calibrated values.

        Parameters
        ----------
        item : int
            The class value to be calibrated.
        calibrated_values : NDArray
            Array of calibrated values.
        max_prob : NDArray of shape (n_samples,)
            Values to be calibrated.
        max_prob_arg : NDArray of shape (n_samples,)
            Indicator of the values to be calibrated.
        calibrators : Dict[RegressorMixin] of len n_classes in calibration set
            Dictionary of all the previously fitted calibrators.

        Returns
        -------
        NDArray
            Updated calibrated values from the predictions of the calibrators.

        Warnings
            If there has not been a fitted calibrator for the specific class.
        """
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
    ):  # It's not recognizing MapieCalibrator here?
        """
        Fit estimator will calibrate the predicted proabilities from the output
        of a classifier.

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
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.
            By default ``None``.
        X_calib : Optional[ArrayLike] of shape (n_calib_samples, n_features)
            Calibration data.
        y_calib : Optional[ArrayLike] of shape (n_calib_samples,)
            Calibration labels.
        sample_weight_calib : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for calib for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.
            By default ``None``.
        calib_size : Optional[float]
            If X_calib and y_calib are not defined, then the calibration
            dataset is created with the split defined by calib_size.
        random_state : int, RandomState instance or None, default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Controls the shuffling applied to the data before applying the
            split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        shuffle : bool, default=True
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Whether or not to shuffle the data before splitting.
            If shuffle=False
            then stratify must be None.
        stratify : array-like, default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            If not None, data is split in a stratified fashion, using this as
            the class labels.
            Read more in the :ref:`User Guide <stratification>`.

        Returns
        -------
        MapieCalibrator
            The model itself.
        """
        cv = check_cv(self.cv)
        estimator = check_estimator_classification(X, y, self.estimator)
        calibrator = self._get_calibrator(cv, self.calibrator)
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
        y_train = _check_y(y_train)
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
        self.calibrators = self._fit_calibrators(
            X_calib, y_calib, sw_calib, estimator, calibrator
        )
        self.classes_ = estimator.classes_
        return self

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Prediction of the calibrated probability score of the class after
        fitting of the classifer and calibrator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            The calibrated score for each max score and zeros at every
            other position in that line.
        """
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
    ) -> NDArray:
        """
        Predict the class of the estimator after calibration.
        Note that in the top-label setting, this class does not change.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples,)
            The class from the predictions.
        """
        check_is_fitted(self, self.fit_attributes)
        return self.classes_[np.argmax(self.predict_proba(X=X), axis=1)]
