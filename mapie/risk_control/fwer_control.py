import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Literal, Union

if TYPE_CHECKING:
    from mapie.risk_control.binary_classification import BinaryClassificationController

import numpy as np
from numpy.typing import ArrayLike, NDArray

FWER_IMPLEMENTED = [
    "bonferroni",
    "fixed_sequence",
    "bonferroni_holm",
    "split_fixed_sequence",
]
FWER_METHODS = Literal[
    "bonferroni", "fixed_sequence", "bonferroni_holm", "split_fixed_sequence"
]


class FWERProcedure(ABC):
    """
    Base class for procedures controlling the Family-Wise Error Rate (FWER).

    This class defines a unified interface for sequential multiple testing
    procedures that allocate and update a global error budget `delta`
    across a set of hypotheses.

    Subclasses implement the strategy that determines:

    - how the error budget is initialized,
    - which hypothesis is tested next,
    - how local significance levels are computed,
    - how the state evolves after a rejection.

    The main entry point is ``run`` which executes the procedure and returns
    the indices of rejected hypotheses.

    Methods to implement
    --------------------
    _init_state(n_lambdas, delta)
        Initialize internal state.

    _select_next_hypothesis(p_values)
        Return index of next hypothesis to test, or None if no test remains.

    _local_significance_levels()
        Return current local significance levels.

    _update_on_reject(hypothesis_index)
        Update state after a rejection.
    """

    def run(self, p_values: NDArray, delta: float) -> NDArray[np.int_]:
        """
        Execute the multiple testing procedure.

        Parameters
        ----------
        p_values : NDArray of shape (n_lambdas,)
            P-values associated with hypotheses.
        delta : float
            Target family-wise error rate.

        Returns
        -------
        NDArray[int]
            Sorted indices of rejected hypotheses.
        """
        p_values = np.asarray(p_values, float)
        n_lambdas = len(p_values)

        self._init_state(n_lambdas, delta)
        rejected_mask: NDArray[np.bool_] = np.zeros(n_lambdas, dtype=bool)

        while True:
            hypothesis_index = self._select_next_hypothesis(p_values)
            if hypothesis_index is None:
                break

            if (
                p_values[hypothesis_index]
                <= self._local_significance_levels()[hypothesis_index]
            ):
                rejected_mask[hypothesis_index] = True
                self._update_on_reject(hypothesis_index)
            else:  # Ignore coverage: Python 3.9 fails to detect this line although it is tested.
                break  # pragma: no cover

        return np.flatnonzero(rejected_mask)

    @abstractmethod
    def _init_state(self, n_lambdas: int, delta: float):
        raise NotImplementedError

    @abstractmethod
    def _select_next_hypothesis(self, p_values: NDArray) -> Union[int, None]:
        raise NotImplementedError

    @abstractmethod
    def _local_significance_levels(self) -> NDArray:
        raise NotImplementedError

    @abstractmethod
    def _update_on_reject(self, hypothesis_index: int):
        raise NotImplementedError


class FWERBonferroniCorrection:
    """
    Bonferroni procedure for controlling the FWER [1].

    Each hypothesis is tested independently at level delta / n_lambdas.
    The procedure stops as soon as one hypothesis is not rejected.

    Notes
    -----
    This is the simplest FWER-controlling method. It does not adapt
    to p-values and does not redistribute error budget after rejections.

    [1] Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle probabilità.
    """

    def run(self, p_values: NDArray, delta: float) -> NDArray[np.int_]:
        """
        Execute the multiple testing procedure.

        Parameters
        ----------
        p_values : NDArray of shape (n_lambdas,)
            P-values associated with hypotheses.
        delta : float
            Target family-wise error rate.

        Returns
        -------
        NDArray[int]
            Sorted indices of rejected hypotheses.
        """
        p_values = np.asarray(p_values, float)
        n_lambdas = len(p_values)
        rejected_mask = p_values <= delta / n_lambdas
        return np.flatnonzero(rejected_mask)


class FWERBonferroniHolm(FWERProcedure):
    """
    Holm step-down procedure for controlling the FWER [1].

    At each step, the hypothesis with the smallest p-value among the
    remaining ones is tested at level delta / k, where k is the number
    of hypotheses still active.

    The procedure stops when the current hypothesis is not rejected.

    Notes
    -----
    This method strictly dominates Bonferroni in power while preserving
    strong FWER control.

    [1] Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian journal of statistics, 65-70.
    """

    def _init_state(self, n_lambdas: int, delta: float):
        self.delta = delta
        self.n_lambdas = n_lambdas
        self.active_hypotheses: NDArray[np.bool_] = np.ones(n_lambdas, dtype=bool)

    def _select_next_hypothesis(self, p_values: NDArray) -> Union[int, None]:
        active_indices = np.flatnonzero(self.active_hypotheses)
        if len(active_indices) == 0:
            return None
        return active_indices[np.argmin(p_values[active_indices])]

    def _local_significance_levels(self) -> NDArray:
        remaining = self.active_hypotheses.sum()
        local_deltas = np.zeros(self.n_lambdas)
        local_deltas[self.active_hypotheses] = self.delta / remaining
        return local_deltas

    def _update_on_reject(self, hypothesis_index: int):
        self.active_hypotheses[hypothesis_index] = False


class FWERFixedSequenceTesting(FWERProcedure):
    """
    Fixed Sequential Testing (ascending) procedure with multi-start
    for controlling the Family-Wise Error Rate (FWER) [1].

    Hypotheses are assumed to be ordered according to a parameter grid
    such that rejection becomes progressively easier along the sequence.

    If multiple starts are used, each start explores a disjoint segment
    of hypotheses. Starts falling inside already rejected regions are
    automatically discarded.

    Parameters
    ----------
    n_starts : int, default=1
        Number of equally spaced starting points used in the multi-start procedure.

    [1] P. Bauer, “Multiple testing in clinical trials,” Statistics in Medicine, vol. 10, no. 6, pp. 871–890, 1991.
    """

    def __init__(self, n_starts: int = 1):
        if n_starts <= 0:
            raise ValueError("n_starts must be a positive integer.")
        self.n_starts = n_starts

    def _init_state(self, n_lambdas: int, delta: float):
        self.n_lambdas = n_lambdas

        if self.n_starts > n_lambdas:
            warnings.warn(
                "n_starts is greater than the number of tests. Thus, it is set to n_lambdas.",
                UserWarning,
            )

        self._effective_starts = min(self.n_starts, n_lambdas)
        self.local_delta = delta / self._effective_starts

        branch_size = n_lambdas // self._effective_starts
        self.start_positions = [i * branch_size for i in range(self._effective_starts)]

    def _select_next_hypothesis(self, p_values):
        while self.start_positions:
            idx = self.start_positions[0]
            level = self.local_delta

            if p_values[idx] <= level:
                return idx

            self.start_positions.pop(0)

        return None

    def _local_significance_levels(self):
        levels = np.zeros(self.n_lambdas)
        for start in self.start_positions:
            levels[start] = self.local_delta
        return levels

    def _update_on_reject(self, hypothesis_index: int):
        new_start_positions = []

        for start in self.start_positions:
            if start < hypothesis_index:
                new_start_positions.append(start)
            elif start == hypothesis_index:
                start += 1

            if start < self.n_lambdas:
                new_start_positions.append(start)

        self.start_positions = new_start_positions


def learn_fixed_sequence_order(
    bcc: "BinaryClassificationController",
    X_learn: ArrayLike,
    y_learn: ArrayLike,
    beta_grid: NDArray = np.logspace(-25, 0, 1000),
    binary: bool = False,
) -> "BinaryClassificationController":
    """
    Learn an ordered sequence of prediction parameters for split fixed-sequence FWER control [1].

    This method performs the learning step of split fixed-sequence testing.
    It must be called before ``calibrate`` when ``fwer_method="split_fixed_sequence"``.

    The data provided here must be independent from the calibration data used later in ``calibrate``.
    Using the same data would invalidate the statistical guarantees.

    A typical workflow is to split your calibration dataset:

    - one subset for learning the parameter order
    - one subset for calibration

    For each value in ``beta_grid``, the parameter whose p-value vector is
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
    BinaryClassificationController
        The controller instance with the learned sequence of ordered prediction parameters.

    Notes
    -----
    This method does NOT perform risk control.
    It only determines an order of parameters.
    Statistical guarantees are provided later when calling ``calibrate``.

    [1] [1] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al. "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022).
    """
    from mapie.risk_control.methods import (
        compute_hoeffding_bentkus_p_value,  # Import here to avoid circular imports
    )

    y_learn = np.asarray(y_learn, dtype=int)
    predictions_per_param = bcc._get_predictions_per_param(
        X_learn, bcc._predict_params, is_calibration_step=True
    )

    r_hat, n_obs = bcc._get_risk_values_and_eff_sample_sizes(
        y_learn, predictions_per_param, bcc._risk
    )
    alpha_np = np.expand_dims(bcc._alpha, axis=1)
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
        candidate = bcc._predict_params[best_idx]

        if bcc.is_multi_dimensional_param:
            candidate = tuple(candidate.tolist())

        if candidate not in ordered_predict_params:
            ordered_predict_params.append(candidate)

    if bcc.is_multi_dimensional_param:
        ordered_predict_params = [list(p) for p in ordered_predict_params]

    bcc._learned_fixed_sequence = np.array(ordered_predict_params, dtype=object)
    return bcc


def control_fwer(
    p_values: NDArray,
    delta: float,
    fwer_method: Union[FWER_METHODS, FWERProcedure] = "bonferroni",
) -> NDArray:
    """
    Apply a Family-Wise Error Rate (FWER) control procedure.

    This function applies a multiple testing correction to a collection
    of p-values in order to control the family-wise error rate (FWER)
    at level ``delta``.

    The correction method is selected via the ``fwer_method`` argument.

    Supported methods are:
    - ``"bonferroni"``: classical Bonferroni correction,
    - ``"bonferroni_holm"``: Sequential Graphical Testing corresponding
      to the Bonferroni-Holm procedure.
    - ``"fixed_sequence"``: Fixed Sequence Testing (FST),
    - ``"split_fixed_sequence"``: Split Fixed Sequence Testing (SFST).
    - Custom procedures can also be implemented by subclassing ``FWERProcedure``
      and passing an instance to ``fwer_method``.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with each tested hypothesis.
    delta : float
        Target family-wise error rate. Must be in (0, 1].
    fwer_method : {"bonferroni", "bonferroni_holm", "fixed_sequence", "split_fixed_sequence"} or FWERProcedure instance, default="bonferroni"
        FWER control strategy.

    Returns
    -------
    valid_index : NDArray
        Sorted indices of hypotheses rejected under FWER control.

    Notes
    -----
    fwer_method="fixed_sequence" corresponds to the fixed sequence testing procedure with one start.
    However, users can use multi-start by instantiating FWERFixedSequenceTesting with
    any desired number of starts and passing the instance to control_fwer.

    If fwer_method="split_fixed_sequence", this function behaves exactly as
    "fixed_sequence". The distinction exists only upstream, where the ordering
    of hypotheses may have been learned from separate data.
    """
    p_values = np.asarray(p_values, dtype=float)
    n_lambdas = len(p_values)

    if n_lambdas == 0:
        raise ValueError("p_values must be non-empty.")
    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0, 1].")

    if isinstance(fwer_method, FWERProcedure):
        procedure: Union[FWERProcedure, FWERBonferroniCorrection] = fwer_method
    elif fwer_method == "bonferroni":
        procedure = FWERBonferroniCorrection()
    elif fwer_method in ["fixed_sequence", "split_fixed_sequence"]:
        procedure = FWERFixedSequenceTesting(n_starts=1)
    elif fwer_method == "bonferroni_holm":
        procedure = FWERBonferroniHolm()
    else:
        raise ValueError(
            f"Unknown FWER control method: {fwer_method}. "
            f"Supported methods are: {FWER_IMPLEMENTED}, "
            "or an instance of FWERProcedure."
        )

    return procedure.run(p_values, delta)
