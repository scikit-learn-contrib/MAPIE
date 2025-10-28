from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, Union, cast
from inspect import signature
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.calibration import _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _check_y, _num_samples, check_is_fitted, indexable

from numpy.typing import ArrayLike, NDArray
from .utils import (
    _check_estimator_classification,
    _check_estimator_fit_predict,
    _check_n_features_in,
    _check_null_weight,
    _fit_estimator,
    _get_calib_set,
)

from ._venn_abers import predict_proba_prefitted_va, VennAbers, VennAbersMultiClass


class TopLabelCalibrator(BaseEstimator, ClassifierMixin):
    """
    Top-label calibration for multi-class problems.
    Performs a calibration on the class with the highest score
    given both score and class, see section 2 of [1].

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default
        ``None``.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    calibrator : Optional[Union[str, RegressorMixin]]
        Any calibrator with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default
        ``None``.
        If ``None``, calibrator defaults to a string "sigmoid" instance.

        By default ``None``.

    cv: Optional[str]
        The cross-validation strategy to compute scores :

        - "split", performs a standard splitting into a calibration and a
          test set.

        - "prefit", assumes that ``estimator`` has been fitted already.
          All the data that are provided in the ``fit`` method are then used
          to calibrate the predictions through the score computation.

        By default "split".

    Attributes
    ----------
    classes_: NDArray
        Array with the name of each class.

    n_classes_: int
        Number of classes that are in the training dataset.

    uncalib_pred: NDArray
        Array of the uncalibrated predictions set by the ``estimator``.

    single_estimator_: ClassifierMixin
        Classifier fitted on the training data.

    calibrators: Dict[Union[int, str], RegressorMixin]
        Dictionnary of all the fitted calibrators.

    References
    ----------
    [1] Gupta, Chirag, and Aaditya K. Ramdas. "Top-label calibration
    and multiclass-to-binary reductions." arXiv preprint
    arXiv:2107.08353 (2021).


    Examples
    --------
    >>> import numpy as np
    >>> from mapie.calibration import TopLabelCalibrator
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> mapie = TopLabelCalibrator().fit(X_toy, y_toy, random_state=20)
    >>> y_calib = mapie.predict_proba(X_toy)
    >>> print(y_calib)
    [[0.84......        nan        nan]
     [0.75......        nan        nan]
     [0.62......        nan        nan]
     [       nan 0.33......        nan]
     [       nan 0.33......        nan]
     [       nan 0.33......        nan]
     [       nan        nan 0.33......]
     [       nan        nan 0.54......]
     [       nan        nan 0.66......]]
    """

    fit_attributes = [
        "estimator",
        "calibrators",
    ]

    named_calibrators = {
        "sigmoid": _SigmoidCalibration(),
        "isotonic": IsotonicRegression(out_of_bounds="clip"),
    }

    valid_cv = ["prefit", "split"]

    valid_inputs = ["multiclass", "binary"]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        calibrator: Optional[Union[str, RegressorMixin]] = None,
        cv: Optional[str] = "split",
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.cv = cv

    def _check_cv(
        self,
        cv: Optional[str],
    ) -> str:
        """
        Check if cross-validator is ``"prefit"`` or ``"split"``.
        Else raise error.

        Parameters
        ----------
        cv : str
            Cross-validator to check.

        Returns
        -------
        str
            'prefit' or 'split'.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv in self.valid_cv:
            return cv
        raise ValueError(f"Invalid cv argument. Allowed values are {self.valid_cv}.")

    def _check_calibrator(
        self,
        calibrator: Optional[Union[str, RegressorMixin]],
    ) -> RegressorMixin:
        """
        Check the input that has been provided for calibrator and
        check that the calibrator is a valid estimator to calibrate.

        Parameters
        ----------
        calibrator : Union[str, RegressorMixin]
            If calibrator is a string then it returns the corresponding
            estimator of ``named_calibrators``, else returns calibrator.

            By defaults ``None``. If ``None``, defaults to ``"sigmoid"``.

        Returns
        -------
        RegressorMixin
            RegressorMixin calibrator.

        Raises
        ------
        ValueError
            If calibrator is not a key of ``named_calibrators``.
        """
        if calibrator is None:
            calibrator = "sigmoid"

        if isinstance(calibrator, str):
            if calibrator in self.named_calibrators.keys():
                calibrator = self.named_calibrators[calibrator]
            else:
                raise ValueError(
                    "Please provide a string in: "
                    + (", ").join(self.named_calibrators.keys())
                    + "."
                )
        _check_estimator_fit_predict(calibrator)
        return calibrator

    def _get_labels(self, X: ArrayLike) -> Tuple[NDArray, NDArray]:
        """
        This method depends on the value of ``method`` and collects the labels
        that are needed to transform a multi-class calibration to multiple
        binary calibrations.

        - "top_label" method means that you take the maximum score and
        calibrate each maximum class separately in a one-versus all setting,
        see section 2 of [1].

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        Tuple[NDArray, NDArray] of shapes (n_samples,) and (n_samples,)
            The first element corresponds to the output scores that have
            to be calibrated and the second element corresponds to the
            predictions.

            In the "top_label" setting, this refers to the maximum scores
            and the class associated to the maximum score.
        """
        pred = self.single_estimator_.predict_proba(X=X)
        max_class_prob = np.max(pred, axis=1).reshape(-1, 1)
        y_pred = self.classes_[np.argmax(pred, axis=1)]
        return max_class_prob, y_pred

    def _check_type_of_target(self, y: ArrayLike):
        """
        Check type of target for calibration class.

        Parameters
        ----------
        y : ArrayLike of shape (n_samples,)
            Training labels.
        """
        if type_of_target(y) not in self.valid_inputs:
            raise ValueError(
                "Make sure to have one of the allowed targets: "
                + (", ").join(self.valid_inputs)
                + "."
            )

    def _fit_calibrator(
        self,
        label: Union[int, str],
        calibrator: RegressorMixin,
        y_calib: NDArray,
        top_class_prob: NDArray,
        y_pred: NDArray,
        sample_weight: Optional[ArrayLike],
    ) -> RegressorMixin:
        """
        Fitting the calibrator requires that we have a binary target, hence,
        we find the subset of values on which we want to apply this
        calibration and apply a binary calibration on these.

        Parameters
        ----------
        label : Union[int, str]
            The class for which we fit a calibrator.
        calibrator : RegressorMixin
            Calibrator to fit.
        y_calib : NDArray of shape (n_samples,)
            Training labels.
        top_class_prob : NDArray of shape (n_samples,)
            The highest score for each input of the method ``predict_proba``
            of the ``estimator`.
        y_pred : NDArray of shape (n_samples,)
            Predictions.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If ``None``, then samples are equally weighted.

        Returns
        -------
        RegressorMixin
            Calibrated estimator.
        """
        calibrator_ = clone(calibrator)
        sample_weight = cast(NDArray, sample_weight)
        given_label_indices = np.argwhere(y_pred.ravel() == label).ravel()
        y_calib_ = np.equal(y_calib[given_label_indices], label).astype(int)
        top_class_prob_ = top_class_prob[given_label_indices]

        if sample_weight is not None:
            sample_weight_ = sample_weight[given_label_indices]
            (sample_weight_, top_class_prob_, y_calib_) = _check_null_weight(
                sample_weight_, top_class_prob_, y_calib_
            )
        else:
            sample_weight_ = sample_weight
        calibrator_ = _fit_estimator(
            calibrator_, top_class_prob_, y_calib_, sample_weight_
        )
        return calibrator_

    def _fit_calibrators(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike],
        calibrator: RegressorMixin,
    ) -> Dict[Union[int, str], RegressorMixin]:
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
            Note that the sample weight defined are only for the training, not
            for the calibration.
            By default ``None``.
        calibrator : RegressorMixin
            Calibrator to fit.

        Returns
        -------
        Dict[Union[int, str], RegressorMixin]
            Dictionnary of fitted calibrators.
        """
        X, y = indexable(X, y)
        y = _check_y(y)
        max_prob, y_pred = self._get_labels(X)
        calibrators = {}
        for label in np.unique(y_pred):
            calibrator_ = self._fit_calibrator(
                label,
                calibrator,
                cast(NDArray, y),
                max_prob,
                y_pred,
                sample_weight,
            )
            calibrators[label] = calibrator_
        return calibrators

    def _pred_proba_calib(
        self,
        idx: int,
        label: Union[int, str],
        calibrated_values: NDArray,
        max_prob: NDArray,
        y_pred: NDArray,
    ) -> None:
        """
        Using the predicted scores, we calibrate the maximum score with the
        specifically fitted calibrator. Note that if there is no calibrator
        for the specific class, then we simply output the not calibrated
        values.


        Parameters
        ----------
        idx : int
            Position of the label for an enumerate of all labels.
        label : Union[int, str]
            The label to define the subset of values to be taken into account.
        calibrated_values : NDArray
            Array of calibrated values to be updated.
        max_prob : NDArray of shape (n_samples,)
            Values to be calibrated.
        y_pred : NDArray of shape (n_samples,)
            Indicator of the values to be calibrated.


        Raises
        ------
        Warnings
            If there has not been a fitted calibrator for the specific class.
        """
        idx_labels = np.where(y_pred.ravel() == label)[0].ravel()
        if label not in self.calibrators.keys():
            calibrated_values[idx_labels, idx] = max_prob[idx_labels].ravel()
            warnings.warn(
                f"WARNING: This predicted label {label} has not been seen "
                + " during the calibration and therefore scores will remain"
                + " unchanged."
            )
        else:
            calibrator_ = self.calibrators[label]
            preds_ = calibrator_.predict(max_prob[idx_labels])
            calibrated_values[idx_labels, idx] = preds_

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        calib_size: Optional[float] = 0.33,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
        **fit_params,
    ) -> TopLabelCalibrator:
        """
        Calibrate the estimator on given datasets, according to the chosen
        method.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,)
            Training labels.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.
            By default ``None``.
        calib_size : Optional[float]
            If ``cv == split`` and X_calib and y_calib are not defined, then
            the calibration dataset is created with the split defined by
            calib_size.
        random_state : int, RandomState instance or ``None``, default is
            ``None``
            See ``sklearn.model_selection.train_test_split`` documentation.
            Controls the shuffling applied to the data before applying the
            split.
            Pass an int for reproducible output across multiple function calls.
        shuffle : bool, default=True
            See ``sklearn.model_selection.train_test_split`` documentation.
            Whether or not to shuffle the data before splitting.
            If shuffle=False, then stratify must be ``None``.
        stratify : array-like, default=None
            See ``sklearn.model_selection.train_test_split`` documentation.
            If not ``None``, data is split in a stratified fashion, using this
            as the class label.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        TopLabelCalibrator
            The model itself.
        """
        cv = self._check_cv(self.cv)
        X, y = indexable(X, y)
        y = _check_y(y)
        self._check_type_of_target(y)
        estimator = _check_estimator_classification(X, y, cv, self.estimator)
        calibrator = self._check_calibrator(self.calibrator)
        sample_weight, X, y = _check_null_weight(sample_weight, X, y)
        self.n_features_in_ = _check_n_features_in(X, cv, estimator)
        random_state = check_random_state(random_state)

        if cv == "prefit":
            self.single_estimator_ = estimator
            self.classes_ = self.single_estimator_.classes_
            self.n_classes_ = len(self.classes_)
            self.calibrators = self._fit_calibrators(X, y, sample_weight, calibrator)
        if cv == "split":
            results = _get_calib_set(
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
            sw_train, X_train, y_train = _check_null_weight(sw_train, X_train, y_train)
            estimator = _fit_estimator(
                clone(estimator),
                X_train,
                y_train,
                sw_train,
                **fit_params,
            )
            self.single_estimator_ = estimator
            self.classes_ = self.single_estimator_.classes_
            self.n_classes_ = len(self.classes_)
            self.calibrators = self._fit_calibrators(
                X_calib, y_calib, sw_calib, calibrator
            )
        return self

    def predict_proba(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Prediction of the calibrated scores using fitted classifier and
        calibrator.

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
        self.uncalib_pred = self.single_estimator_.predict_proba(X=X)

        max_prob, y_pred = self._get_labels(X)

        n = _num_samples(max_prob)
        calibrated_test_values = np.full((n, self.n_classes_), np.nan)

        for idx, label in enumerate(np.unique(y_pred)):
            self._pred_proba_calib(
                idx,
                label,
                calibrated_test_values,
                max_prob,
                y_pred,
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
            The class from the scores.
        """
        check_is_fitted(self, self.fit_attributes)
        return self.single_estimator_.predict(X)


class VennAbersCalibrator(BaseEstimator, ClassifierMixin):
    """
    Venn-ABERS calibration for binary and multi-class problems.

    A class implementing binary [1] or multi-class [2] Venn-ABERS calibration.
    This calibrator provides well-calibrated probabilities with validity guarantees.
    The implementation is based on the reference implementation by the user ip200 [3].

    Can be used in 3 different forms:
    - Prefit Venn-ABERS: estimator is already fitted, only calibration is performed
    - Inductive Venn-ABERS (IVAP): splits data into training and calibration sets
    - Cross Venn-ABERS (CVAP): uses cross-validation for calibration

    Parameters
    ----------
    estimator : ClassifierMixin
        The classifier whose output needs to be calibrated to provide more
        accurate `predict_proba` outputs. Must be a scikit-learn compatible
        classifier with `fit` and `predict_proba` methods.

    cv : Optional[str], default=None
        The cross-validation strategy:

        - ``"prefit"``: Assumes that ``estimator`` has been fitted already.
            All data provided in ``fit`` are used for calibration only.
        - ``None``: Uses inductive or cross validation based on the
            ``inductive`` parameter.

    inductive : bool, default=True
        Determines the calibration strategy when ``cv=None``:

        - ``True``: Inductive Venn-ABERS (IVAP) - splits data into proper
            training and calibration sets.
        - ``False``: Cross Venn-ABERS (CVAP) - uses k-fold cross-validation.

    n_splits : Optional[int], default=None
        Number of folds for Cross Venn-ABERS (CVAP). Must be at least 2.
        Only used when ``inductive=False`` and ``cv=None``.
        Uses ``sklearn.model_selection.StratifiedKFold`` functionality.

    train_proper_size : Optional[float], default=None
        Proportion of the dataset to use for proper training in Inductive
        Venn-ABERS (IVAP). Only used when ``inductive=True`` and ``cv=None``.

        - If float, should be between 0.0 and 1.0.
        - If int, represents the absolute number of training samples.
        - If ``None``, automatically set to complement of ``cal_size``.

    random_state : Optional[int], default=None
        Controls the shuffling applied to the data before splitting.
        Pass an int for reproducible output across multiple function calls.
        Can be overridden in the ``fit`` method.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting.

        - For IVAP: if ``shuffle=False``, then ``stratify`` must be ``None``.
        - For CVAP: controls whether to shuffle each class's samples before
            splitting into batches.

        Can be overridden in the ``fit`` method.

    stratify : Optional[ArrayLike], default=None
        For Inductive Venn-ABERS (IVAP) only. If not ``None``, data is split
        in a stratified fashion, using this as the class labels.
        Can be overridden in the ``fit`` method.

    precision : Optional[int], default=None
        Number of decimal points to round Venn-ABERS calibration probabilities.
        Yields significantly faster computation for larger calibration datasets.
        Trade-off between speed and precision.

    Attributes
    ----------
    classes_ : NDArray
        Array with the name of each class.

    n_classes_ : int
        Number of classes in the training dataset.

    n_features_in_ : int
        Number of features seen during fit.

    va_calibrator_ : Union[VennAbersMultiClass, VennAbers, None]
        The fitted Venn-ABERS calibrator instance.
        May be None in prefit mode with multi-class classification.

    transformers_ : Optional[Pipeline]
        Trasnformers from sklearn pipeline to transform categorical attributes.

    single_estimator_ : Optional[ClassifierMixin]
        The fitted estimator (only for prefit mode).

    p_cal_ : Optional[NDArray]
        Calibration probabilities (only for prefit mode with multi-class).

    y_cal_ : Optional[NDArray]
        Calibration labels (only for prefit mode with multi-class).

    References
    ----------
    [1] Vovk, Vladimir, Ivan Petej, and Valentina Fedorova.
        "Large-scale probabilistic predictors with and without guarantees
        of validity." Advances in Neural Information Processing Systems 28
        (2015). https://arxiv.org/pdf/1511.00213.pdf

    [2] Manokhin, Valery. "Multi-class probabilistic classification using
        inductive and cross Venn–Abers predictors." In Conformal and
        Probabilistic Prediction and Applications, pp. 228-240. PMLR, 2017.

    [3] Reference implementation:
    https://github.com/ip200/venn-abers/blob/main/src/venn_abers.py

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie.calibration import VennAbersCalibrator

    **Example 1: Prefit mode**

    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=3, n_informative=10,
    ...                            random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>> # Fit the base classifier
    >>> clf = GaussianNB()
    >>> _ = clf.fit(X_train, y_train)
    >>> # Calibrate using prefit mode
    >>> va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    >>> _ = va_cal.fit(X_test, y_test)  # Use test set for calibration
    >>> # Get calibrated probabilities
    >>> calibrated_probs = va_cal.predict_proba(X_test)

    **Example 2: Inductive Venn-ABERS (IVAP)**

    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>> # Inductive mode with 30% calibration split
    >>> clf = GaussianNB()
    >>> va_cal = VennAbersCalibrator(
    ...     estimator=clf,
    ...     inductive=True,
    ...     random_state=42
    ... )
    >>> _ = va_cal.fit(X_train, y_train)
    >>> calibrated_probs = va_cal.predict_proba(X_test)
    >>> predictions = va_cal.predict(X_test)

    **Example 3: Cross Venn-ABERS (CVAP)**

    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_informative=10, n_classes=3,
    ...                            random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>> # Cross validation mode with 5 folds
    >>> clf = GaussianNB()
    >>> va_cal = VennAbersCalibrator(
    ...     estimator=clf,
    ...     inductive=False,
    ...     n_splits=5,
    ...     random_state=42
    ... )
    >>> _ = va_cal.fit(X_train, y_train)
    >>> calibrated_probs = va_cal.predict_proba(X_test)
    >>> predictions = va_cal.predict(X_test)

    Notes
    -----
    - Venn-ABERS calibration provides probabilistic predictions with
        validity guarantees under the exchangeability assumption.
    - For binary classification, the method produces well-calibrated
        probabilities with minimal assumptions.
    - For multi-class problems, the method uses a one-vs-one approach
        to extend binary Venn-ABERS to multiple classes.
    - The ``precision`` parameter can significantly speed up computation
        for large datasets with minimal impact on calibration quality.
    - When using ``cv="prefit"``, ensure the estimator is fitted on a
        different dataset than the one used for calibration to avoid
        overfitting.

    See Also
    --------
    TopLabelCalibrator : Top-label calibration for multi-class problems.
    sklearn.calibration.CalibratedClassifierCV : Scikit-learn's probability
        calibration with isotonic regression or Platt scaling.
    """

    fit_attributes = ["va_calibrator_", "classes_", "n_classes_"]

    valid_cv = ["prefit", None]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        cv: Optional[str] = None,
        inductive: bool = True,
        n_splits: Optional[int] = None,
        train_proper_size: Optional[float] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        stratify: Optional[ArrayLike] = None,
        precision: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.cv = cv
        self.inductive = inductive
        self.n_splits = n_splits
        self.train_proper_size = train_proper_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.precision = precision

        # Initialize attributes that will be set during fit
        self.va_calibrator_: Optional[Union[VennAbersMultiClass, VennAbers]] = None
        self.classes_: Optional[NDArray] = None
        self.n_classes_: Optional[int] = None
        self.transformers_: Optional[Pipeline] = None
        self.single_estimator_: Optional[ClassifierMixin] = None
        self.p_cal_: Optional[NDArray] = None
        self.y_cal_: Optional[NDArray] = None

    def _check_cv(self, cv: Optional[str]) -> Optional[str]:
        """
        Check if cross-validator is valid.

        Parameters
        ----------
        cv : Optional[str]
            Cross-validator to check.

        Returns
        -------
        Optional[str]
            'prefit' or None.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv in self.valid_cv:
            return cv
        raise ValueError("Invalid cv argument. " f"Allowed values are {self.valid_cv}.")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[NDArray] = None,
        calib_size: Optional[float] = 0.33,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
        **fit_params,
    ) -> "VennAbersCalibrator":
        """
        Fits the Venn-ABERS calibrator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight : Optional[NDArray] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.
            By default ``None``.

        calib_size : Optional[float], default=0.33
            Proportion of the dataset to use for calibration when using
            Inductive Venn-ABERS (IVAP) mode (``inductive=True`` and ``cv=None``).
            It should be between 0.0 and 1.0 and represents
            the proportion of the dataset to include in the calibration split.
            This parameter is ignored when ``cv="prefit"`` or when using
            Cross Venn-ABERS (``inductive=False``).

        random_state : Optional[Union[int, np.random.RandomState, None]], default=None
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.

        shuffle : Optional[bool], default=True
            Whether to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        stratify : Optional[ArrayLike], default=None
            If not None, data is split in a stratified fashion, using this as
            the class labels.

        **fit_params : dict
            Additional parameters for the underlying estimator.

        Returns
        -------
        VennAbersCalibrator
            The fitted calibrator.

        Raises
        ------
        ValueError
            If required parameters are missing for the chosen mode.
        """
        cv = self._check_cv(self.cv)

        # Check for manual mode (backward compatibility)
        # If estimator is None, we expect this to be manual mode
        if self.estimator is None:
            raise ValueError(
                "For VennAbersCalibrator, an estimator must be provided. "
                "For manual calibration with pre-computed probabilities, "
                "please use the VennAbers class directly from mapie._venn_abers"
            )

        # Validate inputs
        X, y = indexable(X, y)
        y = _check_y(y)
        sample_weight, X, y = _check_null_weight(sample_weight, X, y)
        # Handle categorical features

        from sklearn.pipeline import Pipeline

        last_estimator = self.estimator
        X_processed = X

        if isinstance(last_estimator, Pipeline):
            # Separate transformers and final estimator
            transformers = self.estimator[:-1]  # all steps except last
            last_estimator = self.estimator[-1]  # usually a classifier

            X_processed = transformers.fit_transform(X, y)
            self.transformers_ = transformers

        # Set up classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Prefit mode: estimator is already fitted, only calibrate
        if cv == "prefit":
            try:
                check_is_fitted(last_estimator)
            except NotFittedError:
                raise ValueError(
                    "For cv='prefit', the estimator must be already fitted"
                )

            # Set up classes from the fitted estimator
            self.single_estimator_ = last_estimator
            self.classes_ = self.single_estimator_.classes_

            # Type guard: ensure classes_ is not None
            if self.classes_ is None:
                raise RuntimeError(
                    "classes_ should not be None after fitting estimator"
                )

            self.n_classes_ = len(self.classes_)

            # Get predictions from the fitted estimator
            p_cal_pred = self.single_estimator_.predict_proba(X_processed)

            # Fit Venn-ABERS calibrator on these predictions
            if self.n_classes_ <= 2:
                self.va_calibrator_ = VennAbers()
                self.va_calibrator_.fit(p_cal_pred, y, self.precision)
            else:
                # For multi-class, store calibration data for later use
                self.p_cal_ = np.asarray(p_cal_pred)
                self.y_cal_ = np.asarray(y)
                self.va_calibrator_ = None  # Will be used in predict_proba

            return self

        # Standard inductive or cross validation mode
        # Integrity checks
        if not self.inductive and self.n_splits is None:
            raise ValueError("For Cross Venn-ABERS please provide n_splits")

        # Check random state
        random_state_to_use: Optional[Union[int, np.random.RandomState]] = None
        if random_state is not None:
            random_state_to_use = random_state
        else:
            random_state_to_use = self.random_state

        # Initialize and fit the Venn-ABERS calibrator
        self.va_calibrator_ = VennAbersMultiClass(
            estimator=last_estimator,
            inductive=self.inductive,
            n_splits=self.n_splits,
            cal_size=calib_size,
            train_proper_size=self.train_proper_size,
            random_state=random_state_to_use,
            shuffle=shuffle if shuffle is not None else self.shuffle,
            stratify=stratify if stratify is not None else self.stratify,
            precision=self.precision,
        )

        self.va_calibrator_.fit(X_processed, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X: ArrayLike, loss="log") -> NDArray:
        """
        Prediction of the calibrated scores using fitted classifier and
        Venn-ABERS calibrator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            Venn-ABERS calibrated probabilities.
        """
        check_is_fitted(self, self.fit_attributes)

        cv = self._check_cv(self.cv)

        # Process test data
        if self.transformers_ is not None:
            X_processed = self.transformers_.transform(X)
        else:
            X_processed = X
        # Prefit mode: use fitted estimator to get probabilities, then calibrate
        if cv == "prefit":
            if self.single_estimator_ is None:
                raise RuntimeError(
                    "single_estimator_ should not be None in prefit mode"
                )

            p_test_pred = self.single_estimator_.predict_proba(X_processed)

            # Type guard: ensure n_classes_ is not None after fit
            if self.n_classes_ is None:
                raise RuntimeError("n_classes_ should not be None after fitting")

            if self.n_classes_ <= 2:
                # Binary classification
                if self.va_calibrator_ is None:
                    raise RuntimeError(
                        "va_calibrator_ should not be None for binary classification"
                    )
                p_prime, _ = self.va_calibrator_.predict_proba(p_test_pred)
            else:
                # Multi-class classification
                p_prime, _ = predict_proba_prefitted_va(
                    self.p_cal_,
                    self.y_cal_,
                    p_test_pred,
                    precision=self.precision,
                    va_tpe="one_vs_one",
                )

            return p_prime

        # Standard inductive or cross validation mode
        if self.va_calibrator_ is None:
            raise RuntimeError(
                "va_calibrator_ should not be None in inductive/cross-validation mode"
            )

        # Type guard: ensure we have VennAbersMultiClass instance
        if not isinstance(self.va_calibrator_, VennAbersMultiClass):
            raise RuntimeError(
                "va_calibrator_ should be VennAbersMultiClass instance in "
                "inductive/cross-validation mode"
            )

        if "loss" in signature(self.va_calibrator_.predict_proba).parameters:
            p_prime = self.va_calibrator_.predict_proba(
                X_processed, loss=loss, p0_p1_output=False
            )
        else:
            p_prime = self.va_calibrator_.predict_proba(X_processed, p0_p1_output=False)

        return p_prime

    def predict(self, X: ArrayLike, loss="log") -> NDArray:
        """
        Predict the class of the estimator after Venn-ABERS calibration.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, self.fit_attributes)

        # Type guard: ensure n_classes_ is not None after fit
        if self.n_classes_ is None:
            raise RuntimeError("n_classes_ should not be None after fitting")

        # Type guard: ensure classes_ is not None after fit
        if self.classes_ is None:
            raise RuntimeError("classes_ should not be None after fitting")

        # Get calibrated probabilities
        p_prime = self.predict_proba(X, loss=loss)

        # Store classes_ in a local variable to help type checker
        classes: NDArray = self.classes_
        n_classes = self.n_classes_

        # Convert probabilities to class predictions
        if n_classes <= 2:
            # Binary classification
            y_pred = classes[(p_prime[:, 1] >= 0.5).astype(int)]
        else:
            # Multi-class classification
            y_pred = classes[np.argmax(p_prime, axis=1)]

        return y_pred
