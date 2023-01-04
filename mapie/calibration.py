from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from ._typing import ArrayLike, NDArray
from .utils import (check_binary_zero_one, check_cv,
                    check_estimator_classification,
                    check_estimator_fit_predict, check_n_features_in,
                    check_null_weight, fit_estimator, get_calib_set)


class MapieCalibrator(BaseEstimator, ClassifierMixin):
    """
    Calibration for multi-class problems.

    This class performs calibration for various methods, currently only
    top-level [1].

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default
        ``None``.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    method: Optional[str]
        Method to choose for prediction interval estimates.
        Choose among:

        - "top_label", performs a calibration procedure on the class with
           highest score, see section 2 of [1].

        By default "top_label".

    calibrator : Optional[Union[str, RegressorMixin]]
        Any calibrator with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default
        ``None``.
        If ``None``, calibrator defaults to a string ``sigmoid`` instance.

        By default ``None``.

    cv: Optional[str]
        The cross-validation strategy for computing scores :

        - ``split``, performs a standard splitting procedure into a
          calibration and test set.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          to calibrate the predictions through the score computation.
          At prediction time, quantiles of these scores are used to estimate
          prediction sets.

        By default ``split``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    classes_: NDArray
        Array with the name of each class.

    n_classes_: int
        Number of classes that are in the training dataset.

    uncalib_pred: NDArray
        Array of the uncalibrated predictions.

    calibrators: Dict[str, RegressorMixin]
        Dictionnary of all the fitted calibrators.

    References
    ----------
    [1] Gupta, Chirag, and Aaditya K. Ramdas. "Top-label calibration
    and multiclass-to-binary reductions." arXiv preprint
    arXiv:2107.08353 (2021).


    Examples
    --------
    >>> import numpy as np
    >>> from mapie.calibration import MapieCalibrator
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> mapie = MapieCalibrator().fit(X_toy, y_toy, random_state=20)
    >>> y_calib = mapie.predict_proba(X_toy)
    >>> print(y_calib)
    [[0.84900723 0.         0.        ]
     [0.75432411 0.         0.        ]
     [0.62285341 0.         0.        ]
     [0.         0.66666667 0.        ]
     [0.         0.66666667 0.        ]
     [0.         0.66666667 0.        ]
     [0.         0.         0.33333002]
     [0.         0.         0.54326683]
     [0.         0.         0.66666124]]
    """

    fit_attributes = [
        "estimator",
        "calibrators",
    ]

    valid_calibrators = {
        "sigmoid": _SigmoidCalibration(),
        "isotonic": IsotonicRegression(out_of_bounds="clip")
    }

    valid_methods = ["top_label"]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "top_label",
        calibrator: Optional[Union[str, RegressorMixin]] = None,
        cv: Optional[str] = "split",
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
        y_pred: NDArray,
        sample_weight: Optional[ArrayLike],
    ) -> RegressorMixin:
        """
        Fitting the calibrator requires that we have a binary setting in place,
        hence, we find the subset of values on which we want to apply this
        calibration and apply a binary calibration on these.

        Parameters
        ----------
        item : int
            The class for which we fit a calibrator.
        calibrator : RegressorMixin
            Calibrator to fit.
        y_calib : NDArray of shape (n_samples,)
            Training labels.
        top_class_prob : NDArray of shape (n_samples,)
            The independent values of the calibrator, it represents the
            maximum score of the predictions.
        y_pred : NDArray of shape (n_samples,)
            The array to determine which class the maximum prediction belongs
            to.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If ``None``, then samples are equally weighted.

        Returns
        -------
        RegressorMixin
            Calibrated estimator.
        """
        calibrator_ = clone(calibrator)
        y_calib = cast(NDArray, y_calib)
        sample_weight = cast(NDArray, sample_weight)
        given_label_indices = np.where(y_pred.ravel() == item)[0]
        y_calib_ = check_binary_zero_one(
            np.equal(y_calib[given_label_indices], item).astype(int)
        )
        top_class_prob_ = top_class_prob[given_label_indices]

        if sample_weight is not None:
            sample_weight_ = sample_weight[given_label_indices]
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
        calibrator: Optional[Union[str, RegressorMixin]],
    ) -> RegressorMixin:
        """
        Check the input that has been provided for
        calibrator and check that the calibrator is a valid
        estimator to calibrate.

        Parameters
        ----------
        cv : Union[str, BaseCrossValidator]
            Cross validation parameter.
        calibrator : Union[str, RegressorMixin]
            If calibrator is a string then it is linked to one of the
            valid methods else calibrator as estimator.

            By defaults ``None``. And if ``None``, we set to ``sigmoid``.

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
                    """
                    Please provide a valid string from the valid calibrators
                    such as {self.valid_calibrators.keys().join(", ")}.
                    """
                )
        check_estimator_fit_predict(calibrator)
        return calibrator

    def _get_labels(
        self,
        X: ArrayLike
    ) -> Tuple[NDArray, NDArray]:
        """
        This method depends on the {self.method} and applies the separation
        that will be needed from a multi-class problem to a binary calibration.

        - Top-Label method means that you take the maximum score and
        calibrate each maximum class separately in a one-versus all setting.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        Tuple[NDArray, NDArray] of shapes (n_samples,) and (n_samples,)
            The first element corresponds to the output scores that have
            to be calibrated and the second element corresponds to the
            subset that needs to be taken into account for calibration.

            In the Top-Label setting, this refers to the maximum scores
            and the class associated to the maximum score.

        Raises
        ------
        ValueError
            If the method provided has not been implemented.
        """
        pred = self.estimator.predict_proba(X=X)  # type: ignore
        max_class_prob = np.max(pred, axis=1).reshape(-1, 1)
        y_pred = self.classes_[np.argmax(pred, axis=1)]
        return max_class_prob, y_pred

    def _check_method(self) -> None:
        """
        Check that the method is valid.

        Raises
        ------
        ValueError
            If the method does not belong to the valid methods.
        """
        if self.method not in self.valid_methods:
            raise ValueError(
                "Invalid method, allowed method are ",
                self.valid_methods.join(", ")
            )

    def _fit_calibrators(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike],
        calibrator: RegressorMixin,
    ) -> Dict[int, RegressorMixin]:
        """
        This method sequentially fits the calibrators for each labels
        defined by ``_get_labels`` method.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,)
            Training labels.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.
            By default ``None``.
        calibrator : RegressorMixin
            Calibrator to fit.

        Returns
        -------
        Dict[int, RegressorMixin]
            Dictionnary of fitted calibrators.
        """
        X, y = indexable(X, y)
        y = _check_y(y)
        max_prob, y_pred = self._get_labels(X)
        calibrators = {}
        for item in np.unique(y_pred):
            calibrator_ = self._fit_calibrator(
                item,
                calibrator,
                y,
                max_prob,
                y_pred,
                sample_weight,
            )
            calibrators[item] = calibrator_
        return calibrators

    def _pred_proba_calib(
        self,
        idx: int,
        item: int,
        calibrated_values: NDArray,
        max_prob: NDArray,
        y_pred: NDArray,
        calibrators: Dict[int, RegressorMixin],
    ) -> NDArray:
        """
        Using the predicted scores, we calibrate scores with the fitted
        calibrators. Note that if there is no calibrator for a the specific
        class, then we simply output the not calibrated values.

        Note that if the calibrated score prediction is 0, there would be an
        issue when finding the class it belongs to. We set it equal to
        {np.finfo(np.float64).eps} as this would likely not have large impact
        on the calibration scores, yet, would set the maximum to the correct
        label.


        Parameters
        ----------
        item : int
            The defined subset to be calibrated.
        calibrated_values : NDArray
            Array of calibrated values.
        max_prob : NDArray of shape (n_samples,)
            Values to be calibrated.
        y_pred : NDArray of shape (n_samples,)
            Indicator of the values to be calibrated.
        calibrators : Dict[RegressorMixin] of length n_classes
            Dictionary of all the previously fitted calibrators.

        Returns
        -------
        NDArray
            Updated calibrated values from the predictions of the calibrators.

        Raises
        ------
        Warnings
            If there has not been a fitted calibrator for the specific class.
        """
        correct_label = np.where(y_pred.ravel() == item)[0].ravel()
        if item not in calibrators:
            calibrated_values[
                correct_label, idx
                ] = max_prob[correct_label].ravel()
            warnings.warn(
                "WARNING: This predicted label {item} has not been seen during"
                + " the calibration and therefore scores will remain"
                + " unchanged."
            )
        else:
            EPSILON = np.finfo(np.float64).eps
            calibrator_ = calibrators[item]
            preds_ = calibrator_.predict(max_prob[correct_label])
            idx_zero_pred = np.where(preds_ < EPSILON)[0]
            preds_[idx_zero_pred] = EPSILON
            calibrated_values[correct_label, idx] = preds_
        return calibrated_values

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        calib_size: Optional[float] = 0.33,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> MapieCalibrator:
        """
        Fit estimator will calibrate the predicted probabilities from the output
        of a classifier.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,)
            Training labels.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
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
        random_state : int, RandomState instance or ``None``, default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Controls the shuffling applied to the data before applying the
            split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.
        shuffle : bool, default=True
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Whether or not to shuffle the data before splitting.
            If shuffle=False
            then stratify must be ``None``.
        stratify : array-like, default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            If not ``None``, data is split in a stratified fashion, using this
            as the class labels.
            Read more in the :ref:`User Guide <stratification>`.

        Returns
        -------
        MapieCalibrator
            The model itself.
        """
        self._check_method()
        cv = check_cv(self.cv)
        estimator = check_estimator_classification(X, y, cv, self.estimator)
        calibrator = self._get_calibrator(self.calibrator)
        X, y = indexable(X, y)
        y = _check_y(y)
        assert type_of_target(y) in ["multiclass", "binary"]
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        random_state = check_random_state(random_state)

        if cv == "prefit":
            self.classes_ = self.estimator.classes_  # type: ignore
            self.n_classes_ = len(self.classes_)
            self.calibrators = self._fit_calibrators(
                X, y, sample_weight, calibrator
            )
        if cv == "split":
            results = get_calib_set(
                X,
                y,
                sample_weight=sample_weight,
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
            self.n_classes_ = len(np.unique(y_train))
            self.estimator = fit_estimator(
                clone(estimator), X_train, y_train, sw_train,
            )
            self.classes_ = self.estimator.classes_  # type: ignore
            self.calibrators = self._fit_calibrators(
                X_calib, y_calib, sw_calib, calibrator
            )
        return self

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Prediction of the calibrated score of the class after fitting of
        the classifer and calibrator.

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
        self.uncalib_pred = self.estimator.predict_proba(X=X)  # type: ignore

        max_prob, y_pred = self._get_labels(X)

        n = _num_samples(max_prob)
        calibrated_test_values = np.zeros((n, self.n_classes_))

        for idx, item in enumerate(np.unique(y_pred)):
            calibrated_test_values = self._pred_proba_calib(
                idx,
                item,
                calibrated_test_values,
                max_prob,
                y_pred,
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
        return self.estimator.predict(X)  # type: ignore
