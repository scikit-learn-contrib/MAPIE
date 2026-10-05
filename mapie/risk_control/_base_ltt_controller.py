from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.risk_control.fwer_control import (
    FWER_IMPLEMENTED,
    FWERFixedSequenceTesting,
    FWERProcedure,
)

from .methods import compute_hoeffding_bentkus_p_value
from .risks import ClassSpecificRisk


class _BaseLTTController:
    """
    Base class factoring out the Learn-Then-Test (LTT) logic shared between
    `BinaryClassificationController` and `MultiClassificationController`.
    """

    # Attributes set by subclasses in __init__, declared here so that mypy can
    # check the method bodies below.
    is_multi_risk: bool
    is_multi_dimensional_param: bool
    _predict_params: NDArray
    _risk: Sequence[ClassSpecificRisk]
    _alpha: NDArray
    _learned_fixed_sequence: Optional[NDArray[Any]]
    best_predict_param: Optional[Union[float, Tuple[float, ...]]]

    def _get_predictions_per_param(
        self, X: ArrayLike, params: NDArray, is_calibration_step: bool = False
    ) -> NDArray:
        raise NotImplementedError  # pragma: no cover

    def _check_fwer_method(self, fwer_method):
        if isinstance(fwer_method, str):
            if fwer_method not in FWER_IMPLEMENTED:
                raise ValueError(
                    f"Unknown fwer_method '{fwer_method}'. Allowed: {sorted(FWER_IMPLEMENTED)}"
                )

        elif not isinstance(fwer_method, FWERProcedure):
            raise TypeError("fwer_method must be a string or FWERProcedure instance.")

        if (self.is_multi_risk or self.is_multi_dimensional_param) and (
            fwer_method == "fixed_sequence"
            or isinstance(fwer_method, FWERFixedSequenceTesting)
        ):
            raise ValueError(
                "Fixed sequence testing cannot be used with multiple risks "
                "or multidimensional parameters. Use 'split_fixed_sequence' instead."
            )

        return fwer_method

    def learn_fixed_sequence_order(
        self,
        X_learn: ArrayLike,
        y_learn: ArrayLike,
        beta_grid: NDArray = np.logspace(-25, 0, 1000),
        binary: bool = False,
    ) -> _BaseLTTController:
        """
        Learn an ordered sequence of prediction parameters for split fixed-sequence FWER control.

        This method performs the learning step of split fixed-sequence testing.
        It must be called before `calibrate` when `fwer_method="split_fixed_sequence"`.

        The data provided here must be independent from the calibration data used later in `calibrate`.
        Using the same data would invalidate the statistical guarantees.

        A typical workflow is to split your calibration dataset:

        - one subset for learning the parameter order
        - one subset for calibration

        For each value in `beta_grid`, the parameter whose p-value vector is
        closest to the constant vector beta is selected. Duplicate parameters are
        removed while preserving order, yielding a deterministic testing sequence.

        Parameters
        ----------
        X_learn : ArrayLike
            Features used only to learn the parameter order.

        y_learn : ArrayLike
            Binary labels associated with X_learn.

        beta_grid : NDArray, default=np.logspace(-25, 0, 1000)
            Grid of target p-values used to construct the ordering.
            Smaller values prioritize parameters with stronger evidence.

        binary : bool, default=False
            Whether the loss associated with the controlled risk is binary.

        Returns
        -------
        _BaseLTTController
            The controller instance with the learned sequence of ordered prediction parameters.

        Notes
        -----
        This method does NOT perform risk control.
        It only determines an order of parameters.
        Statistical guarantees are provided later when calling `calibrate`.
        """
        y_learn = np.asarray(y_learn, dtype=int)
        predictions_per_param = self._get_predictions_per_param(
            X_learn, self._predict_params, is_calibration_step=True
        )

        r_hat, n_obs = self._get_risk_values_and_eff_sample_sizes(
            y_learn, predictions_per_param, self._risk
        )
        alpha_np = np.expand_dims(self._alpha, axis=1)
        p_values = np.array(
            [
                compute_hoeffding_bentkus_p_value(r_hat_i, n_obs_i, alpha_np_i, binary)
                for r_hat_i, n_obs_i, alpha_np_i in zip(r_hat, n_obs, alpha_np)
            ]
        )

        n_risks, n_lambdas = p_values.shape[:2]
        ordered_predict_params: List[Any] = []

        for beta_value in beta_grid:
            beta_vector: NDArray[np.float64] = np.repeat(beta_value, n_risks)

            distances_to_beta: list[np.float64] = [
                np.max(np.abs(p_values[:, idx, 0] - beta_vector))
                for idx in range(n_lambdas)
            ]

            best_idx = np.argmin(distances_to_beta)
            candidate = self._predict_params[best_idx]

            if self.is_multi_dimensional_param:
                candidate = tuple(candidate.tolist())

            if candidate not in ordered_predict_params:
                ordered_predict_params.append(candidate)

        if self.is_multi_dimensional_param:
            ordered_predict_params = [list(p) for p in ordered_predict_params]

        self._learned_fixed_sequence = np.array(ordered_predict_params, dtype=object)

        return self

    def predict(self, X_test: ArrayLike) -> NDArray:
        """
        Predict using predict_function at the best threshold.

        Parameters
        ----------
        X_test : ArrayLike
            Features

        Returns
        -------
        NDArray
            NDArray of shape (n_samples,)

        Raises
        ------
        ValueError
            If the method .calibrate was not called,
            or if no valid thresholds were found during calibration.
        """
        if self.best_predict_param is None:
            raise ValueError(
                "Cannot predict. "
                "Either you forgot to calibrate the controller first, "
                "or calibration was not successful."
            )
        return cast(
            NDArray,
            self._get_predictions_per_param(
                X_test,
                np.array([self.best_predict_param]),
            )[0],
        )

    @staticmethod
    def _get_risk_values_and_eff_sample_sizes(
        y_true: NDArray,
        predictions_per_param: NDArray,
        risks: Sequence[ClassSpecificRisk],
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the values of risks and effective sample sizes for multiple risks
        and for multiple parameter values.
        Returns arrays with shape (n_risks, n_params).
        """

        risks_values_and_eff_sizes = np.array(
            [
                [
                    risk.get_value_and_effective_sample_size(y_true, predictions)
                    for predictions in predictions_per_param
                ]
                for risk in risks
            ]
        )

        risk_values = risks_values_and_eff_sizes[:, :, 0]
        effective_sample_sizes = risks_values_and_eff_sizes[:, :, 1]

        return risk_values, effective_sample_sizes

    def _convert_target_level_to_alpha(self, target_level: List[float]) -> NDArray:
        alpha = []
        for risk, target in zip(self._risk, target_level):
            if risk.higher_is_better:
                alpha.append(1 - target)
            else:
                alpha.append(target)
        return np.array(alpha)

    @staticmethod
    def _check_if_multi_dimensional_param(
        predict_params: NDArray,
    ) -> bool:
        """
        Check if the the parameters (the λ) are multi-dimensional.
        """
        if predict_params.ndim == 1:
            return False
        elif predict_params.ndim == 2:
            return True
        else:
            raise ValueError(
                "predict_params must be a 1D array of shape (n_params,) for one-dimensional parameters, "
                "or a 2D array of shape (n_params, params_dim) for multi-dimensional parameters "
                "(params_dim=1 is allowed for the case when a one-dimensional parameter is not used as a threshold)."
            )

    def _check_predictions(self, predictions_per_param: NDArray) -> None:
        """
        Checks if predictions are probabilities for one-dimensional parameters,
        or binary predictions for multi-dimensional parameters.
        """
        if (
            not self.is_multi_dimensional_param
            and np.logical_or(
                predictions_per_param == 0, predictions_per_param == 1
            ).all()
        ):
            warnings.warn(
                "All predictions are either 0 or 1 while the parameters are one-dimensional. "
                "Make sure that the provided predict_function is a "
                "predict_proba method or a function that outputs probabilities.",
            )

        if (
            self.is_multi_dimensional_param
            and not np.logical_or.reduce(
                (
                    predictions_per_param == 0,
                    predictions_per_param == 1,
                    np.isnan(predictions_per_param),
                )
            ).all()
        ):
            raise ValueError(
                "The provided predict_function with multi-dimensional "
                "parameters must return binary predictions (0, 1, np.nan)."
            )
