from __future__ import annotations

from typing import Callable, List, Literal, Optional, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.risk_control.fwer_control import (
    FWER_METHODS,
    FWERProcedure,
)

from ._base_controller import _BaseLTTController
from .risks import BinaryRisk, ClassSpecificRisk, _OccurrenceConditionRisk


class MultiClassificationLTTController(_BaseLTTController):
    """
    Controls the risk or performance of a multiclass classifier, treating each
    class as a one-vs-rest (OVR) binary problem.

    MultiClassificationLTTController finds, among a set of candidate thresholds, the
    single threshold (λ) that statistically guarantees one or several
    class-specific risks (each defined by a `ClassSpecificRisk` instance, see
    e.g. :func:`mapie.risk_control.risks.build_precision_ovr`) to be below
    their target level(s) (the risks are "controlled"). It can be used to
    control performance metrics as well, such as the per-class precision. In
    that case, the threshold guarantees that the performance is above the
    target level(s).

    At prediction time, a sample is assigned the class with the highest
    predicted probability, unless that probability is below the selected
    threshold, in which case the controller abstains (returns `np.nan`).

    Usage:

    1. Instantiate a MultiClassificationLTTController, providing the predict_proba
       method of your fitted multiclass classifier and the risks to control, e.g.
       one `ClassSpecificRisk` per class (via
       :func:`mapie.risk_control.risks.build_precision_ovr`)
    2. Call the calibrate method to find the threshold
    3. Use the predict method to predict using the best threshold

    Note: for a given model, calibration dataset, target level(s), and
    confidence level, there may not be any threshold controlling the risk(s).

    Parameters
    ----------

    predict_function : Callable[..., NDArray]
        predict_proba method of a fitted multiclass classifier, with output
        of shape (len(X), n_classes).

        For multi-dimensional parameters (when `list_predict_params` is 2D),
        predict_function must instead take (X, \\*params) and output the
        predicted class labels directly, of shape (len(X),).

    risks : List[Union[ClassSpecificRisk, BinaryRisk]]
        The risks (or performance metrics) to control. These can be
        class-specific risks, one per tested class (or several per class), see
        e.g. :func:`mapie.risk_control.risks.build_precision_ovr`, which builds
        one-vs-rest precision risks for a list of class labels. They can also
        be class-agnostic `BinaryRisk` instances, such as
        `mapie.risk_control.abstention_rate`, which measures the proportion of
        abstentions (`np.nan` predictions) over all the samples.

    target_level : List[float]
        The maximum risk level (or minimum performance level) of each risk in
        `risks`. Must be a list of the same length as `risks`, with values
        between 0 and 1.

    confidence_level : float, default=0.9
        The confidence level with which the risks (or performance) are
        controlled. Must be between 0 and 1. See the documentation for
        detailed explanations.

    best_predict_param_choice : Literal["auto"], default="auto"
        How to select the best threshold from the valid thresholds that
        control the risks (or performance). "auto" is currently the only
        supported value: it uses the first risk in `risks` as the secondary
        objective.

    list_predict_params : NDArray, default=np.linspace(0, 0.99, 100)
        The set of thresholds (noted λ in [1]) to consider for controlling the
        risks (or performance), of shape (n_params,). The values threshold the
        predicted probability of the most likely class.
        Note that performance is degraded when `len(list_predict_params)` is
        large as it is used by the Bonferroni correction [1].

    classes : Optional[ArrayLike], default=None
        The class labels, in the order of the columns of `predict_function`'s
        output (e.g. the `classes_` attribute of your classifier). If None, the
        classes are inferred from the labels of the calibration set, which is
        only correct if every class appears in it.

    fwer_method : {"bonferroni", "bonferroni_holm", "split_fixed_sequence"} or FWERProcedure instance, default="bonferroni_holm"
        Method used to control the family-wise error rate (FWER).

        Supported methods:
        - `"bonferroni"` : Classical Bonferroni correction.
        It is valid in all settings but can be conservative, especially when the number of tested parameters is large.
        - `"bonferroni_holm"` : Sequential Graphical Testing corresponding
        to the Bonferroni–Holm procedure. This is the default method and is suitable for general settings.
        - `"split_fixed_sequence"` : Split Fixed Sequence Testing (SFST). Requires
        calling `learn_fixed_sequence_order` on independent data before `calibrate`.

        Note: `"fixed_sequence"` is only supported when a single risk is
        controlled and `list_predict_params` is 1D; otherwise, use
        `"split_fixed_sequence"` instead.

    Attributes
    ----------
    valid_predict_params : NDArray
        The valid thresholds that control the risks (or performance).
        Use the calibrate method to compute these.

    best_predict_param : Optional[float]
        The best threshold that controls the risks (or performance).
        Use the calibrate method to compute it.

    p_values : NDArray
        P-values associated with each tested threshold in
        `list_predict_params`, one row per risk in `risks`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from mapie.risk_control import MultiClassificationLTTController
    >>> from mapie.risk_control.risks import build_precision_ovr

    >>> X, y = make_classification(
    ...     n_samples=600,
    ...     n_features=5,
    ...     n_informative=5,
    ...     n_redundant=0,
    ...     n_classes=3,
    ...     n_clusters_per_class=1,
    ...     random_state=42,
    ... )
    >>> X_train, X_temp, y_train, y_temp = train_test_split(
    ...     X, y, test_size=0.4, random_state=42
    ... )
    >>> X_calib, X_test, y_calib, y_test = train_test_split(
    ...     X_temp, y_temp, test_size=0.1, random_state=42
    ... )

    >>> clf = LogisticRegression().fit(X_train, y_train)

    >>> controller = MultiClassificationLTTController(
    ...     predict_function=clf.predict_proba,
    ...     risks=build_precision_ovr(np.unique(y_train)),
    ...     target_level=[0.7] * len(np.unique(y_train)),
    ...     classes=clf.classes_,
    ... )

    >>> predictions = controller.calibrate(X_calib, y_calib).predict(X_test)

    References
    ----------
    [1] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al.
    "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022)
    """

    def __init__(
        self,
        predict_function: Callable[..., NDArray],
        risks: list[_OccurrenceConditionRisk],
        target_level: List[float],
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[
            Literal["auto"], _OccurrenceConditionRisk
        ] = "auto",
        list_predict_params: NDArray = np.linspace(0, 0.99, 100),
        classes: Optional[ArrayLike] = None,
        fwer_method: Union[FWER_METHODS, FWERProcedure] = "bonferroni_holm",
    ):
        self.is_multi_risk = self._check_if_multi_risk_control(risks, target_level)
        self._init_ltt(
            predict_function,
            risks,
            target_level,
            confidence_level,
            best_predict_param_choice,
            list_predict_params,
            fwer_method,
        )
        self._infer_classes = classes is None
        self._classes_: Optional[NDArray] = (
            None if classes is None else self._check_classes(classes)
        )

    def calibrate(  # pragma: no cover
        self, X_calibrate: ArrayLike, y_calibrate: ArrayLike
    ) -> MultiClassificationLTTController:
        """
        Calibrate the MultiClassificationLTTController.
        Sets attributes valid_predict_params and best_predict_param (if the risk
        or performance can be controlled at the target level).

        Parameters
        ----------
        X_calibrate : ArrayLike
            Features of the calibration set.

        y_calibrate : ArrayLike
            Labels of the calibration set.

        Returns
        -------
        MultiClassificationLTTController
            The calibrated controller instance.

        Notes
        -----
        When using `fwer_method="split_fixed_sequence"`,
        the learning step must be performed separately on independent data:

        1. bcc.learn_fixed_sequence_order(X_learn, y_learn)
        2. bcc.calibrate(X_calibrate, y_calibrate)

        Using the same data for both steps would invalidate guarantees.
        """
        self._calibrate(X_calibrate, y_calibrate)
        return self

    def _prepare_y(self, y: ArrayLike) -> NDArray:
        """
        Convert the labels into an array, infer the classes from them if they
        were not provided, otherwise check that they belong to the classes.
        """
        y_ = np.asarray(y)
        if self._infer_classes:
            self._classes_ = np.unique(y_)
        else:
            self._check_labels_in_classes(y_)
        return y_

    @staticmethod
    def _check_classes(classes: ArrayLike) -> NDArray:
        """
        Check that classes contains at least two distinct labels.
        """
        classes_ = np.asarray(classes)
        if (
            classes_.ndim != 1
            or len(classes_) < 2
            or len(np.unique(classes_)) != len(classes_)
        ):
            raise ValueError(
                "classes must be a 1D array of at least two distinct class labels."
            )
        return classes_

    def _check_labels_in_classes(self, y: NDArray) -> None:
        """
        Check that every label of y is one of the classes.
        """
        unknown_labels = np.setdiff1d(y, cast(NDArray, self._classes_))
        if len(unknown_labels) > 0:
            raise ValueError(
                f"y contains labels that are not in classes: {unknown_labels.tolist()}."
            )

    @staticmethod
    def _check_if_multi_risk_control(
        risks: list[_OccurrenceConditionRisk],
        target_level: List[float],
    ) -> bool:
        """
        Check that risks and target levels are lists of the same non-zero
        length, and return whether several risks are controlled.
        """
        if (
            isinstance(risks, list)
            and isinstance(target_level, list)
            and len(risks) == len(target_level)
            and len(risks) > 0
        ):
            return len(risks) > 1
        raise ValueError(
            "You must provide a list of risks and a list of target levels "
            "of the same non-zero length."
        )

    def _check_calibration_predictions(
        self, predictions_per_param: NDArray, y_calibrate_: NDArray
    ) -> None:
        self._check_predictions_type(predictions_per_param, y_calibrate_)

    @staticmethod
    def _check_predictions_type(
        predictions_per_param: NDArray, y_calibrate_: NDArray
    ) -> None:
        """
        check if predictions type is different than y_calibrate, else raise an error
        """
        predictions_dtype = np.asarray(predictions_per_param).dtype
        y_calibrate_dtype = np.asarray(y_calibrate_).dtype
        if np.issubdtype(predictions_dtype, np.number) != np.issubdtype(
            y_calibrate_dtype, np.number
        ):
            raise TypeError(
                f"predictions type ({predictions_dtype}) is different from "
                f"y_calibrate type ({y_calibrate_dtype})."
            )

    def _set_best_predict_param_choice(
        self,
        best_predict_param_choice: Union[Literal["auto"], _OccurrenceConditionRisk],
    ) -> _OccurrenceConditionRisk:
        if best_predict_param_choice == "auto":
            # when multi risk, we minimize the first risk in the list
            return cast(_OccurrenceConditionRisk, self._risk[0])
        if isinstance(best_predict_param_choice, (ClassSpecificRisk, BinaryRisk)):
            return best_predict_param_choice
        raise TypeError(
            f"Got object of type {type(best_predict_param_choice)}. "
            "best_predict_param_choice must be either 'auto' "
            "or a ClassSpecificRisk or BinaryRisk instance."
        )

    def _get_predictions_per_param_mono_param(
        self, X, params, is_calibration_step=False
    ):
        if self._is_predict_proba_method(X):
            y_pred = self._predict_pba_function(X, params)
        else:
            y_pred = self._predict_function(X, params)
        return y_pred

    def _is_predict_proba_method(self, X: ArrayLike) -> bool:
        """
        Check if self._predict_function behaves like a predict_proba method,
        by calling it with X alone and inspecting the shape of the output:
        (n_samples, n_classes) with n_classes > 1 means it is (True),
        (n_samples, 1) means it is not (False).
        """
        try:
            predictions = np.asarray(self._predict_function(X))
            return predictions.ndim == 2 and predictions.shape[1] > 1
        except TypeError as e:
            if "object is not callable" in str(e):
                raise TypeError(
                    "Error when calling the predict_function. "
                    "Maybe you provided a classifier to the "
                    "predict_function parameter of the MultiClassificationLTTController. "
                    "You should provide your classifier's predict_proba method instead."
                ) from e
            elif "positional argument" in str(e):
                # predict_function requires more than just X (e.g. params),
                # so it cannot be a predict_proba method.
                return False
            else:
                raise

    def _check_predictions(self, predictions_per_param: NDArray) -> None:
        if self.is_multi_dimensional_param:
            # In this mode, predict_function already outputs the final class
            # labels directly (not predict_proba output), so the probability
            # checks below don't apply.
            return
        if predictions_per_param.ndim != 2:
            raise ValueError(
                "Error when calling the predict_function. "
                "Maybe you provided a predict method instead of a "
                "predict_proba method to the predict_function parameter "
                "of the MultiClassificationLTTController. "
                "You should provide a predict function that returns the "
                "probabilities of each class, like scikit-learn's "
                "predict_proba method, with shape (n_samples, n_classes)."
            )
        if np.any((predictions_per_param < 0) | (predictions_per_param > 1)) or (
            not np.allclose(predictions_per_param.sum(axis=1), 1)
        ):
            raise ValueError(
                "Error when calling the predict_function. "
                "The values it returns must be valid probabilities: "
                "each value must lie in [0, 1] and each row must sum to 1. "
                "Maybe you provided a decision_function method or another "
                "scoring method instead of a predict_proba method to the "
                "predict_function parameter of the MultiClassificationLTTController."
            )

    def _predict_pba_function(self, X, params) -> NDArray:
        predictions_proba = np.asarray(self._predict_function(X))
        self._check_predictions(predictions_proba)
        if self._classes_ is not None and predictions_proba.shape[1] != len(
            self._classes_
        ):
            raise ValueError(
                f"predict_function returns {predictions_proba.shape[1]} columns "
                f"but there are {len(self._classes_)} classes. Provide `classes` "
                "matching the columns of predict_function's output."
            )
        max_proba = predictions_proba.max(axis=1)
        predicted_index = predictions_proba.argmax(axis=1)
        if self._classes_ is None:
            class_label = predicted_index
        elif np.issubdtype(self._classes_.dtype, np.number):
            class_label = self._classes_[predicted_index]
        else:
            class_label = self._classes_.astype(object)[predicted_index]
        is_above_param = max_proba[np.newaxis, :] >= params[:, np.newaxis]
        y_pred = np.where(is_above_param, class_label[np.newaxis, :], np.nan)
        return y_pred
