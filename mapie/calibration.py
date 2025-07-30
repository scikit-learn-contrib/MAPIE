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

from numpy.typing import ArrayLike, NDArray
from .utils import (_check_estimator_classification,
                    _check_estimator_fit_predict, _check_n_features_in,
                    _check_null_weight, _fit_estimator, _get_calib_set)


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
        "isotonic": IsotonicRegression(out_of_bounds="clip")
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
        raise ValueError(
            "Invalid cv argument. "
            f"Allowed values are {self.valid_cv}."
        )

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
                    + (", ").join(self.named_calibrators.keys()) + "."
                )
        _check_estimator_fit_predict(calibrator)
        return calibrator

    def _get_labels(
        self,
        X: ArrayLike
    ) -> Tuple[NDArray, NDArray]:
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
                + (", ").join(self.valid_inputs) + "."
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
            (
                sample_weight_, top_class_prob_, y_calib_
            ) = _check_null_weight(
                sample_weight_,
                top_class_prob_,
                y_calib_
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
            calibrated_values[
                idx_labels, idx
                ] = max_prob[idx_labels].ravel()
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
            self.calibrators = self._fit_calibrators(
                X, y, sample_weight, calibrator
            )
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
            sw_train, X_train, y_train = _check_null_weight(
                sw_train,
                X_train,
                y_train
            )
            estimator = _fit_estimator(
                clone(estimator), X_train, y_train, sw_train, **fit_params,
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


class VennABERSCalibrator(BaseEstimator, ClassifierMixin):
    """
    Venn-ABERS calibration for binary classification problems.
    Implements the Inductive Venn-ABERS Predictors (IVAP) algorithm described in:
    "Large-scale probabilistic prediction with and without validity guarantees"
    by Vovk et al. (https://arxiv.org/pdf/1511.00213.pdf).
    This is a MAPIE wrapper for
    the implementation in https://github.com/ptocca/VennABERS/.
    Note that VennABERSCalibrator uses its own specific calibration algorithm.

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default
        ``None``.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    cv: Optional[str]
        The cross-validation strategy to compute scores:

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

    single_estimator_: ClassifierMixin
        Classifier fitted on the training data.

    calibration_points_: List
        List of calibration points used for Venn-ABERS calibration.

    References
    ----------
    [1] Vovk, V., Petej, I., & Fedorova, V. (2015). Large-scale probabilistic
    predictors with and without validity guarantees. Advances in Neural
    Information Processing Systems, 28.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.calibration import VennABERSCalibrator
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    >>> model = LogisticRegression().fit(X[:80], y[:80])
    >>> calibrator = VennABERSCalibrator(model, cv="prefit")
    >>> _ = calibrator.fit(X[80:], y[80:])
    >>> probs = calibrator.predict_proba(X[:5])
    >>> print(probs)
    [[0.14285714 0.85714286]
     [0.8        0.2       ]
     [0.1        0.9       ]
     [0.91666667 0.08333333]
     [0.91666667 0.08333333]]
    """

    fit_attributes = [
        "single_estimator_",
        "calibration_points_",
    ]

    valid_cv = ["prefit", "split"]

    valid_inputs = ["binary"]

    calibration_points_: list[Tuple[float, Union[int, float]]]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        cv: Optional[str] = "split",
    ) -> None:
        self.estimator = estimator
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
        raise ValueError(
            "Invalid cv argument. "
            f"Allowed values are {self.valid_cv}."
        )

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
                "VennABERSCalibrator only supports binary classification. "
                "Make sure to have one of the allowed targets: "
                + (", ").join(self.valid_inputs) + "."
            )

    def _prepare_data(self, calibr_points: list[Tuple[float, float]]) -> Tuple:
        """
        Prepare data for Venn-ABERS calibration.

        Parameters
        ----------
        calibr_points : List[Tuple[float, float]]
            List of calibration points (score, label).

        Returns
        -------
        Tuple
            Prepared data for Venn-ABERS calibration.
        """
        pts_sorted = sorted(calibr_points)

        xs = np.array([p[0] for p in pts_sorted])
        ys = np.array([p[1] for p in pts_sorted])
        pts_unique, pts_index, pts_inverse, pts_counts = np.unique(
            xs,
            return_index=True,
            return_counts=True,
            return_inverse=True
        )

        a = np.zeros(pts_unique.shape)
        np.add.at(a, pts_inverse, ys)

        w = pts_counts
        y_prime = a / w
        y_csd = np.cumsum(a)  # Equivalent to np.cumsum(w * y_prime)
        x_prime = np.cumsum(w)
        k_prime = len(x_prime)

        return y_prime, y_csd, x_prime, pts_unique, k_prime

    def _algorithm1(self, P: Dict, k_prime: int) -> list:
        """
        Algorithm 1 from Venn-ABERS paper.

        Parameters
        ----------
        P : Dict
            Dictionary of points.
        k_prime : int
            Number of unique calibration points.

        Returns
        -------
        List
            Stack of points.
        """
        S = []
        P[-1] = np.array((-1, -1))
        S.append(P[-1])
        S.append(P[0])

        for i in range(1, k_prime + 1):
            while len(S) > 1 and self._non_left_turn(S[-2], S[-1], P[i]):
                S.pop()
            S.append(P[i])
        return S

    def _algorithm2(self, P: Dict, S: list, k_prime: int) -> NDArray:
        """
        Algorithm 2 from Venn-ABERS paper.

        Parameters
        ----------
        P : Dict
            Dictionary of points.
        S : List
            Stack of points from Algorithm 1.
        k_prime : int
            Number of unique calibration points.

        Returns
        -------
        NDArray
            F1 function values.
        """
        S_prime = S[::-1]  # reverse the stack

        F1 = np.zeros((k_prime + 1,))
        for i in range(1, k_prime + 1):
            F1[i] = self._slope(S_prime[-1], S_prime[-2])
            P[i-1] = P[i-2] + P[i] - P[i-1]

            if self._not_below(P[i-1], S_prime[-1], S_prime[-2]):
                continue

            S_prime.pop()
            while len(S_prime) > 1 and \
                    self._non_left_turn(P[i-1], S_prime[-1], S_prime[-2]):
                S_prime.pop()
            S_prime.append(P[i-1])

        return F1

    def _algorithm3(self, P: Dict, k_prime: int) -> list:
        """
        Algorithm 3 from Venn-ABERS paper.

        Parameters
        ----------
        P : Dict
            Dictionary of points.
        k_prime : int
            Number of unique calibration points.

        Returns
        -------
        List
            Stack of points.
        """
        S = []
        S.append(P[k_prime + 1])
        S.append(P[k_prime])

        for i in range(k_prime - 1, -1, -1):  # k'-1, k'-2, ..., 0
            while len(S) > 1 and self._non_right_turn(S[-2], S[-1], P[i]):
                S.pop()
            S.append(P[i])

        return S

    def _algorithm4(self, P: Dict, S: list, k_prime: int) -> NDArray:
        """
        Algorithm 4 from Venn-ABERS paper.

        Parameters
        ----------
        P : Dict
            Dictionary of points.
        S : List
            Stack of points from Algorithm 3.
        k_prime : int
            Number of unique calibration points.

        Returns
        -------
        NDArray
            F0 function values.
        """
        S_prime = S[::-1]  # reverse the stack

        F0 = np.zeros((k_prime + 1,))
        for i in range(k_prime, 0, -1):  # k', k'-1, ..., 1
            F0[i] = self._slope(S_prime[-1], S_prime[-2])
            P[i] = P[i-1] + P[i+1] - P[i]

            if self._not_below(P[i], S_prime[-1], S_prime[-2]):
                continue

            S_prime.pop()
            while len(S_prime) > 1 and \
                    self._non_right_turn(P[i], S_prime[-1], S_prime[-2]):
                S_prime.pop()
            S_prime.append(P[i])

        return F0

    def _compute_F(self, x_prime: NDArray, y_csd: NDArray,
                   k_prime: int) -> Tuple[NDArray, NDArray]:
        """
        Compute F0 and F1 functions for Venn-ABERS calibration.

        Parameters
        ----------
        x_prime : NDArray
            Cumulative sum of weights.
        y_csd : NDArray
            Cumulative sum of weighted labels.
        k_prime : int
            Number of unique calibration points.

        Returns
        -------
        Tuple[NDArray, NDArray]
            F0 and F1 function values.
        """
        # Compute F1
        P = {0: np.array((0, 0))}
        P.update({i+1: np.array((k, v)) for i, (k, v)
                  in enumerate(zip(x_prime, y_csd))})

        S = self._algorithm1(P, k_prime)
        F1 = self._algorithm2(P, S, k_prime)

        # Compute F0
        P = {0: np.array((0, 0))}
        P.update({i+1: np.array((k, v)) for i, (k, v)
                  in enumerate(zip(x_prime, y_csd))})
        P[k_prime + 1] = P[k_prime] + np.array((1.0, 0.0))

        S = self._algorithm3(P, k_prime)
        F0 = self._algorithm4(P, S, k_prime)

        return F0, F1

    def _get_F_val(self, F0: NDArray, F1: NDArray,
                   pts_unique: NDArray,
                   test_objects: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Get F0 and F1 values for test objects.

        Parameters
        ----------
        F0 : NDArray
            F0 function values.
        F1 : NDArray
            F1 function values.
        pts_unique : NDArray
            Unique calibration points.
        test_objects : NDArray
            Test objects to calibrate.

        Returns
        -------
        Tuple[NDArray, NDArray]
            p0 and p1 probabilities.
        """
        pos0 = np.searchsorted(pts_unique, test_objects, side='left')
        pos1 = np.searchsorted(pts_unique[:-1], test_objects, side='right') + 1
        return F0[pos0], F1[pos1]

    def _scores_to_multi_probs(self, calibr_points: list[Tuple[float, float]],
                               test_objects: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Convert scores to multi-probabilities using Venn-ABERS calibration.

        Parameters
        ----------
        calibr_points : List[Tuple[float, float]]
            List of calibration points (score, label).
        test_objects : NDArray
            Test objects to calibrate.

        Returns
        -------
        Tuple[NDArray, NDArray]
            p0 and p1 probabilities.
        """
        # Prepare data
        y_prime, y_csd, x_prime, pts_unique, k_prime = self._prepare_data(calibr_points)

        # Compute F0 and F1 functions
        F0, F1 = self._compute_F(x_prime, y_csd, k_prime)

        # Get values for test objects
        p0, p1 = self._get_F_val(F0, F1, pts_unique, test_objects)

        return p0, p1

    def _non_left_turn(self, a: NDArray, b: NDArray, c: NDArray) -> bool:
        """
        Check if three points make a non-left turn.

        Parameters
        ----------
        a : NDArray
            First point.
        b : NDArray
            Second point.
        c : NDArray
            Third point.

        Returns
        -------
        bool
            True if non-left turn, False otherwise.
        """
        d1 = b - a
        d2 = c - b
        return np.cross(d1, d2) <= 0

    def _non_right_turn(self, a: NDArray, b: NDArray, c: NDArray) -> bool:
        """
        Check if three points make a non-right turn.

        Parameters
        ----------
        a : NDArray
            First point.
        b : NDArray
            Second point.
        c : NDArray
            Third point.

        Returns
        -------
        bool
            True if non-right turn, False otherwise.
        """
        d1 = b - a
        d2 = c - b
        return np.cross(d1, d2) >= 0

    def _slope(self, a: NDArray, b: NDArray) -> float:
        """
        Calculate slope between two points.

        Parameters
        ----------
        a : NDArray
            First point.
        b : NDArray
            Second point.

        Returns
        -------
        float
            Slope between points.
        """
        ax, ay = a
        bx, by = b
        return (by - ay) / (bx - ax)

    def _not_below(self, t: NDArray, p1: NDArray, p2: NDArray) -> bool:
        """
        Check if point t is not below the line defined by p1 and p2.

        Parameters
        ----------
        t : NDArray
            Point to check.
        p1 : NDArray
            First point of the line.
        p2 : NDArray
            Second point of the line.

        Returns
        -------
        bool
            True if t is not below the line, False otherwise.
        """
        p1x, p1y = p1
        p2x, p2y = p2
        tx, ty = t
        m = (p2y - p1y) / (p2x - p1x)
        b = (p2x * p1y - p1x * p2y) / (p2x - p1x)
        return (ty >= tx * m + b)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.33,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> VennABERSCalibrator:
        """
        Fit the calibrator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,)
            Training labels.
        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If ``None``, then samples are equally weighted.
        calib_size : Optional[float], default=0.33
            If ``cv == "split"``, the proportion of samples to use for calibration.
        random_state : Optional[Union[int, np.random.RandomState]]
            Random state for reproducibility.
        shuffle : bool, default=True
            Whether to shuffle the data before splitting.
        stratify : Optional[ArrayLike]
            If not None, data is split in a stratified fashion.

        Returns
        -------
        VennABERSCalibrator
            Fitted calibrator.
        """
        X, y = indexable(X, y)
        y = _check_y(y)
        self._check_type_of_target(y)

        cv = self._check_cv(self.cv)
        estimator = _check_estimator_classification(X, y, cv, self.estimator)
        sample_weight, X, y = _check_null_weight(sample_weight, X, y)

        if cv == "prefit":
            self.single_estimator_ = estimator
            self.classes_ = self.single_estimator_.classes_
            self.n_classes_ = len(self.classes_)

            # Get scores for calibration set
            scores = self.single_estimator_.predict_proba(X)[:, 1]

            # Create calibration points
            self.calibration_points_ = list(zip(scores, cast(NDArray, y)))
        else:  # cv == "split"
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

            # Fit estimator on training data
            estimator = _fit_estimator(
                clone(estimator), X_train, y_train, sw_train
            )
            self.single_estimator_ = estimator
            self.classes_ = self.single_estimator_.classes_
            self.n_classes_ = len(self.classes_)

            # Get scores for calibration set
            scores = self.single_estimator_.predict_proba(X_calib)[:, 1]

            # Create calibration points
            self.calibration_points_ = list(zip(scores, cast(NDArray, y_calib)))

        return self

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """
        Predict probabilities for test data using Venn-ABERS calibration.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples, 2)
            Calibrated probabilities.
        """
        check_is_fitted(self, self.fit_attributes)

        # Get scores for test data
        scores = self.single_estimator_.predict_proba(X)[:, 1]

        # Apply Venn-ABERS calibration
        p0, p1 = self._scores_to_multi_probs(self.calibration_points_, scores)

        # Normalize probabilities
        p1_normalized = p1 / (p1 + (1 - p0))

        # Return probabilities for both classes
        result = np.zeros((len(scores), 2))
        result[:, 0] = 1 - p1_normalized
        result[:, 1] = p1_normalized

        return result

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predict class labels for test data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        NDArray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, self.fit_attributes)

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
