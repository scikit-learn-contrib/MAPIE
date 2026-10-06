from __future__ import annotations

from typing import Any, Callable, List, Literal, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.risk_control.fwer_control import (
    FWER_METHODS,
    FWERProcedure,
)
from mapie.utils import check_valid_ltt_params_index

from ._base_ltt_controller import _BaseLTTController
from .methods import ltt_procedure
from .risks import ClassSpecificRisk


class MultiClassificationController(_BaseLTTController):
    """
    Controls the risk or performance of a multiclass classifier, treating each
    class as a one-vs-rest (OVR) binary problem.

    MultiClassificationController finds, among a set of candidate thresholds, the
    single threshold (λ) that statistically guarantees one or several
    class-specific risks (each defined by a `ClassSpecificRisk` instance, see
    e.g. `build_precision_ovr`) to be below their target level(s) (the risks are
    "controlled"). It can be used to control performance metrics as well, such
    as the per-class precision. In that case, the threshold guarantees that the
    performance is above the target level(s).

    At prediction time, a sample is assigned the class with the highest
    predicted probability, unless that probability is below the selected
    threshold, in which case the controller abstains (returns `np.nan`).

    Usage:

    1. Instantiate a MultiClassificationController, providing the predict_proba
       method of your fitted multiclass classifier and one `ClassSpecificRisk`
       per class to control (e.g. via `build_precision_ovr`)
    2. Call the calibrate method to find the threshold
    3. Use the predict method to predict using the best threshold

    Note: for a given model, calibration dataset, target level(s), and
    confidence level, there may not be any threshold controlling the risk(s).

    Parameters
    ----------
    risk_combination_method : Union[int, str, Callable[[NDArray], NDArray]]
        How the per-class risk values are combined into the single secondary
        score used to select `best_predict_param` among the thresholds that
        control the risk(s).
        Valid options:

        - A class label (must be one of the labels found in `y_calibrate`):
          the threshold is chosen to minimize that class's own risk, ignoring
          the others.
        - A callable that takes an array of shape (n_risks, n_valid_params) --
          the value of each risk in `risks`, for every valid threshold -- and
          returns an array of shape (n_valid_params,) combining them into a
          single score per threshold. The threshold minimizing this score is
          selected.

    predict_function : Callable[[ArrayLike], NDArray]
        predict_proba method of a fitted multiclass classifier.
        Its output signature must be of shape (len(X), n_classes).

        Or, in the general case of multi-dimensional parameters, a function
        that takes (X, \\*params) and outputs predicted class labels. In that
        case, `list_predict_params` must be provided.

    risks : List[ClassSpecificRisk]
        The class-specific risks (or performance metrics) to control, one per
        tested class (or several per class). See e.g. `build_precision_ovr`,
        which builds one-vs-rest precision risks for a list of class labels.

    target_level : Union[float, List[float]]
        The maximum risk level (or minimum performance level). Must be between
        0 and 1. If a single float is provided, it applies to every risk in
        `risks`. Can be a list matching the length of `risks` to set a
        different target level per risk.

    confidence_level : float, default=0.9
        The confidence level with which the risks (or performance) are
        controlled. Must be between 0 and 1. See the documentation for
        detailed explanations.

    best_predict_param_choice : Literal["auto"], default="auto"
        How to select the best threshold from the valid thresholds that
        control the risks (or performance). "auto" is currently the only
        supported value: it uses every risk in `risks` as the secondary
        objective, combined through `risk_combination_method`.

    list_predict_params : NDArray, default=np.linspace(0, 0.99, 100)
        The set of thresholds (noted λ in [1]) to consider for controlling the
        risks (or performance). When `predict_function` is a `predict_proba`
        method, the shape is (n_params,) and the values threshold the
        predicted probability of the most likely class. When
        `predict_function` is a general function with multi-dimensional
        parameters, the shape is (n_params, params_dim).
        Note that performance is degraded when `len(list_predict_params)` is
        large as it is used by the Bonferroni correction [1].

    fwer_method : {"bonferroni", "bonferroni_holm", "split_fixed_sequence"} or FWERProcedure instance, default="bonferroni_holm"
        Method used to control the family-wise error rate (FWER).

        Supported methods:
        - `"bonferroni"` : Classical Bonferroni correction.
        It is valid in all settings but can be conservative, especially when the number of tested parameters is large.
        - `"bonferroni_holm"` : Sequential Graphical Testing corresponding
        to the Bonferroni–Holm procedure. This is the default method and is suitable for general settings.
        - `"split_fixed_sequence"` : Split Fixed Sequence Testing (SFST). Requires
        calling `learn_fixed_sequence_order` on independent data before `calibrate`.

        Note: `"fixed_sequence"` is not supported here, since
        MultiClassificationController always controls multiple risks (one per
        tested class); use `"split_fixed_sequence"` instead.

    Attributes
    ----------
    valid_predict_params : NDArray
        The valid thresholds that control the risks (or performance).
        Use the calibrate method to compute these.

    best_predict_param : Optional[float]
        The best threshold that controls the risks (or performance), selected
        according to `risk_combination_method`.
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
    >>> from mapie.risk_control import MultiClassificationController
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

    >>> controller = MultiClassificationController(
    ...     predict_function=clf.predict_proba,
    ...     risks=build_precision_ovr(np.unique(y_train)),
    ...     target_level=0.7,
    ...     risk_combination_method=1,
    ... )

    >>> predictions = controller.calibrate(X_calib, y_calib).predict(X_test)

    References
    ----------
    [1] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al.
    "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022)
    """

    def __init__(
        self,
        risk_combination_method: str | int | Callable[[NDArray], NDArray],
        predict_function: Callable[[ArrayLike], NDArray],
        risks: list[ClassSpecificRisk],
        target_level: Union[float, List[float]],
        confidence_level: float = 0.9,
        best_predict_param_choice: Literal["auto"] = "auto",
        list_predict_params: NDArray = np.linspace(0, 0.99, 100),
        fwer_method: Union[FWER_METHODS, FWERProcedure] = "bonferroni_holm",
    ):
        self.risk_combination_method = risk_combination_method
        self.is_multi_risk = True
        self._predict_function = predict_function
        self._risk = risks
        target_level_list = (
            target_level if isinstance(target_level, list) else [target_level]
        )
        self._alpha = self._convert_target_level_to_alpha(target_level_list)
        self._delta = 1 - confidence_level

        self._best_predict_param_choice = self._set_best_predict_param_choice(
            best_predict_param_choice
        )

        self._predict_params = list_predict_params
        self.is_multi_dimensional_param = self._check_if_multi_dimensional_param(
            self._predict_params
        )
        self.fwer_method = self._check_fwer_method(fwer_method)
        self._learned_fixed_sequence: Optional[NDArray[Any]] = None

        self.valid_predict_params: NDArray = np.array([])
        self.best_predict_param = None
        self.p_values: Optional[NDArray] = None

    def _check_risk_combination_method(
        self, risk_combination_method, y_calibrate
    ) -> Callable[[NDArray], NDArray]:
        list_class_label = np.unique(y_calibrate).tolist()
        dict_class_label = {
            value: index for index, value in enumerate(list_class_label)
        }
        if isinstance(risk_combination_method, int | str):
            # the user want the risk_combination_method to take only a specific class
            class_label = risk_combination_method
            assert risk_combination_method in list_class_label, (
                f""" risk_combination_method : {class_label} must be in list_class_label  : {list_class_label}"""
            )
            return lambda array: array[dict_class_label[class_label], :]

        if callable(risk_combination_method):
            return cast(
                Callable[[NDArray], NDArray],
                risk_combination_method,
            )

        raise TypeError("risk_combination_method must be a class label or a callable")

    # All subfunctions are unit-tested. To avoid having to write
    # tests just to make sure those subfunctions are called,
    # we don't include .calibrate in the coverage report
    def calibrate(  # pragma: no cover
        self, X_calibrate: ArrayLike, y_calibrate: ArrayLike
    ) -> MultiClassificationController:
        """
        Calibrate the MultiClassificationController.
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
        MultiClassificationController
            The calibrated controller instance.

        Notes
        -----
        When using `fwer_method="split_fixed_sequence"`,
        the learning step must be performed separately on independent data:

        1. bcc.learn_fixed_sequence_order(X_learn, y_learn)
        2. bcc.calibrate(X_calibrate, y_calibrate)

        Using the same data for both steps would invalidate guarantees.
        """
        y_calibrate_ = np.asarray(y_calibrate, dtype=int)

        self.risk_combination_method = self._check_risk_combination_method(
            self.risk_combination_method, y_calibrate
        )

        original_params = self._predict_params
        if self.fwer_method == "split_fixed_sequence":
            if self._learned_fixed_sequence is None:
                raise ValueError(
                    "You must call 'learn_fixed_sequence_order' before 'calibrate' "
                    "when using fwer_method='split_fixed_sequence'."
                )
            self._predict_params = self._learned_fixed_sequence

        predictions_per_param = self._get_predictions_per_param(
            X_calibrate, self._predict_params, is_calibration_step=True
        )
        # broadcast one _alpha_level per risk.
        if self._alpha.shape[0] != len(self._risk):
            self._alpha = np.repeat(self._alpha, len(self._risk))
        # aggregates all risks
        risk_values, eff_sample_sizes = self._get_risk_values_and_eff_sample_sizes(
            y_calibrate_, predictions_per_param, self._risk
        )
        (valid_index, p_values) = ltt_procedure(
            risk_values,
            np.expand_dims(self._alpha, axis=1),
            self._delta,
            eff_sample_sizes,
            True,
            fwer_method=self.fwer_method,
        )
        valid_params_index = valid_index[0]

        self.valid_predict_params = self._predict_params[valid_params_index]

        check_valid_ltt_params_index(
            predict_params=self._predict_params, valid_index=self.valid_predict_params
        )

        if len(self.valid_predict_params) == 0:
            self.best_predict_param = None
        else:
            self._set_best_predict_param(
                y_calibrate_,
                predictions_per_param,
                valid_params_index,
                risk_values,
            )

        self.p_values = p_values
        self._predict_params = original_params

        return self

    def _set_best_predict_param_choice(
        self,
        best_predict_param_choice: Literal["auto"] = "auto",
    ) -> Sequence[ClassSpecificRisk]:
        if best_predict_param_choice == "auto":
            # when multi risk, we minimize the first risk in the list
            return self._risk

    def _set_best_predict_param(
        self,
        y_calibrate_: NDArray,
        predictions_per_param: NDArray,
        valid_params_index: List[Any],
        risk_values: NDArray,
    ):
        secondary_risks_per_param, _ = self._get_risk_values_and_eff_sample_sizes(
            y_calibrate_,
            predictions_per_param[valid_params_index],
            self._best_predict_param_choice,
        )

        assert callable(self.risk_combination_method), (
            """ internal bug, self.risk_combination_method must be a callable"""
        )
        risk_score = self.risk_combination_method(secondary_risks_per_param)
        assert risk_score.shape[0] == secondary_risks_per_param.shape[1], (
            "risk_combination_method must be a function "
            f"that transform an input of shape ({secondary_risks_per_param.shape[0]}, {secondary_risks_per_param.shape[1]})"
            f" into an output ({secondary_risks_per_param.shape[1]}) "
        )
        best_index = np.flatnonzero(risk_score == risk_score.min())

        if len(best_index) > 1:
            # When several parameters reach the minimum secondary risk, break ties by
            # selecting the one that maximizes the first risk (to be less conservative).
            first_risk_values = risk_values[0, valid_params_index]
            best_index = best_index[np.argmax(first_risk_values[best_index])]
        else:
            best_index = best_index[0]
        best_predict_param = self.valid_predict_params[best_index]
        if isinstance(best_predict_param, np.ndarray):
            self.best_predict_param = tuple(best_predict_param.tolist())
        else:
            self.best_predict_param = float(best_predict_param)

    @staticmethod
    def default_agg_class_pba(params, predictions_proba) -> NDArray:
        max_proba = predictions_proba.max(axis=1)
        class_label = predictions_proba.argmax(axis=1)
        is_above_param = max_proba[np.newaxis, :] >= params[:, np.newaxis]
        # When no class reaches the threshold, we default to class np.nan.
        # This is probably to be changed.
        y_pred = np.where(is_above_param, class_label[np.newaxis, :], np.nan)
        return y_pred

    def _get_predictions_per_param(
        self,
        X: ArrayLike,
        params: NDArray,
        is_calibration_step=False,
        custom_agg_class_pba=default_agg_class_pba,
    ) -> NDArray:
        """Returns y_pred of shape (n_samples)"""
        n_params = len(params)
        n_samples = len(np.asarray(X))
        if self.is_multi_dimensional_param:
            y_pred: NDArray[np.float64] = np.empty((n_params, n_samples), dtype=float)
            for i in range(n_params):
                y_pred[i] = self._predict_function(X, *params[i])
            if is_calibration_step:
                self._check_predictions(y_pred)
        else:
            try:
                predictions_proba = self._predict_function(X)
            except TypeError as e:
                if "object is not callable" in str(e):
                    raise TypeError(
                        "Error when calling the predict_function. "
                        "Maybe you provided a classifier to the "
                        "predict_function parameter of the MultiClassificationController. "
                        "You should provide your classifier's predict_proba method instead."
                    ) from e
                else:
                    raise
            predictions_proba = np.asarray(predictions_proba)
            if predictions_proba.ndim != 2:
                raise ValueError(
                    "Error when calling the predict_function. "
                    "Maybe you provided a predict method instead of a "
                    "predict_proba method to the predict_function parameter "
                    "of the MultiClassificationController. "
                    "You should provide a predict function that returns the "
                    "probabilities of each class, like scikit-learn's "
                    "predict_proba method, with shape (n_samples, n_classes)."
                )
            if np.any((predictions_proba < 0) | (predictions_proba > 1)) or (
                not np.allclose(predictions_proba.sum(axis=1), 1)
            ):
                raise ValueError(
                    "Error when calling the predict_function. "
                    "The values it returns must be valid probabilities: "
                    "each value must lie in [0, 1] and each row must sum to 1. "
                    "Maybe you provided a decision_function method or another "
                    "scoring method instead of a predict_proba method to the "
                    "predict_function parameter of the MultiClassificationController."
                )
            if is_calibration_step:
                self._check_predictions(predictions_proba)
            y_pred = custom_agg_class_pba(params, predictions_proba)
        return y_pred
