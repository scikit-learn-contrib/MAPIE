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
        Calibrate the BinaryClassificationController.
        Sets attributes valid_predict_params and best_predict_param (if the risk
        or performance can be controlled at the target level).

        Parameters
        ----------
        X_calibrate : ArrayLike
            Features of the calibration set.

        y_calibrate : ArrayLike
            Binary labels of the calibration set.

        Returns
        -------
        BinaryClassificationController
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
        """
        only works for == auto
        """
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
                        "Maybe you provided a binary classifier to the "
                        "predict_function parameter of the BinaryClassificationController. "
                        "You should provide your classifier's predict_proba method instead."
                    ) from e
                else:
                    raise
            except IndexError as e:
                if "array is 1-dimensional, but 2 were indexed" in str(e):
                    raise IndexError(
                        "Error when calling the predict_function. "
                        "Maybe the predict function you provided returns only the "
                        "probability of the positive class. "
                        "You should provide a predict function that returns the "
                        "probabilities of both classes, like scikit-learn estimators."
                    ) from e
                else:
                    raise
            if is_calibration_step:
                self._check_predictions(predictions_proba)
            y_pred = custom_agg_class_pba(params, predictions_proba)
        return y_pred
