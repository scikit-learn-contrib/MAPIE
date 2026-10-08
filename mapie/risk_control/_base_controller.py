from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.risk_control.fwer_control import (
    FWER_METHODS,
    FWERFixedSequenceTesting,
    FWERProcedure,
    control_fwer,
)

from .methods import compute_hoeffding_bentkus_p_value
from .risks import ClassSpecificRisk


class _BaseLTTController:
    """
    Base class factoring out the Learn-Then-Test (LTT) logic shared between
    `BinaryClassificationController` and `MultiClassificationLTTController`.
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
        raise NotImplementedError  # pragma: no cover


def ltt_procedure(
    r_hat: NDArray,
    alpha_np: NDArray,
    delta: float,
    n_obs: NDArray,
    binary: bool = False,
    fwer_method: Union[FWER_METHODS, FWERProcedure] = "bonferroni_holm",
) -> Tuple[List[List[Any]], NDArray]:
    """
    Apply the Learn-Then-Test procedure for risk control.
    Note that we will do a multiple test for `r_hat` that are
    less than level `alpha_np`.
    The procedure follows the instructions in [1]:
        - Calculate p-values for each lambdas discretized
        - Apply a family wise error rate algorithm, here Bonferonni correction
        - Return the index lambdas that give you the control at alpha level

    Note that in the case of multi-risk, the arrays r_hat, alpha_np, and n_obs
    should have the same length for the first dimension which corresponds
    to the number of risks. In the case of a single risk, the length should be 1.

    Parameters
    ----------
    r_hat: NDArray of shape (n_risks, n_lambdas).
        Empirical risk with respect to the lambdas.
        Here lambdas are thresholds that impact decision-making,
        therefore empirical risk.

    alpha_np: NDArray of shape (n_risks, n_alpha).
        Contains the different alphas control level.
        The empirical risk should be less than alpha with
        probability 1-delta.
        For MultiLabelClassificationController, the shape should be (1, n_alpha).
        For BinaryClassificationController, the shape should be (n_risks, 1).

    delta: float.
        Probability of not controlling empirical risk.
        Correspond to proportion of failure we don't
        want to exceed.

    n_obs: NDArray of shape (n_risks, n_lambdas).
        Correspond to the number of observations used to compute the risk.
        In the case of a conditional loss, n_obs must be the
        number of effective observations used to compute the empirical risk
        for each lambda.

    binary: bool, default=False
        Must be True if the loss associated to the risk is binary.

    fwer_method : {"bonferroni", "bonferroni_holm", "fixed_sequence", "split_fixed_sequence"} or FWERProcedure instance, default="bonferroni_holm"
        FWER control strategy.

    Returns
    -------
    valid_index: List[List[Any]].
        Contain the valid index that satisfy FWER control
        for each alpha (length aren't the same for each alpha).

    p_values : NDArray of shape (n_lambdas, n_alpha)
        P-values associated with each tested parameter. In the multi-risk setting,
        they correspond to the maximum over the tested risks.

    Notes
    -----
    fwer_method="fixed_sequence" corresponds to the fixed sequence testing procedure with one start.
    However, users can use multi-start by instantiating FWERFixedSequenceTesting with
    any desired number of starts and passing the instance to control_fwer.

    fwer_method="split_fixed_sequence" behaves identically to "fixed_sequence" at this stage.
    The ordering must have been learned beforehand on independent data (typically by the controller).

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".
    """
    if not (r_hat.shape[0] == n_obs.shape[0] == alpha_np.shape[0]):
        raise ValueError("r_hat, n_obs, and alpha_np must have the same length.")
    p_values = np.array(
        [
            compute_hoeffding_bentkus_p_value(r_hat_i, n_obs_i, alpha_np_i, binary)
            for r_hat_i, n_obs_i, alpha_np_i in zip(r_hat, n_obs, alpha_np)
        ]
    )
    p_values = p_values.max(
        axis=0
    )  # to handle multiple risks, take max over risks (no effect if mono risk)

    # Fixed Sequence Testing (FST) only supports a single monotonic risk.
    # - If non-monotonic: raise warning.
    # - If decreasing: reverse order so FST tests easiest -> hardest;
    #   store permutation to remap indices afterward.
    order = None
    p_values_original = p_values
    if (fwer_method == "fixed_sequence") or (
        isinstance(fwer_method, FWERFixedSequenceTesting)
    ):
        if r_hat.shape[0] > 1:
            raise ValueError("fixed_sequence cannot be used with multiple risks.")

        direction = _check_risk_monotonicity(r_hat[0])

        if direction == "none":
            warnings.warn(
                "Fixed sequence testing requires a monotonic risk over lambdas (thresholds) to find "
                "optimal solutions but this hypothesis is not verified here. "
                "We recommand you try split_fixed_sequence instead if the hypothesis ordering is not known a priori.",
                UserWarning,
            )
            average_variation = np.mean(np.diff(r_hat[0]))
            direction = "increasing" if average_variation > 0 else "decreasing"

        if direction == "decreasing":
            order = np.arange(len(p_values))[::-1]
            p_values = p_values[order]

        # To have 100% coverage
        if direction == "increasing":
            pass

    valid_index = []
    for i in range(alpha_np.shape[1]):
        idx = control_fwer(p_values[:, i], delta, fwer_method=fwer_method)
        if order is not None:
            idx = order[idx]
        l_index = idx.tolist()
        valid_index.append(l_index)
    return valid_index, p_values_original
