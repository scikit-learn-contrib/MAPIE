import warnings
from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray


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
            else:
                break

        return np.flatnonzero(rejected_mask)

    @abstractmethod
    def _init_state(self, n_lambdas: int, delta: float):
        pass

    @abstractmethod
    def _select_next_hypothesis(self, p_values: NDArray) -> Union[int, None]:
        pass

    @abstractmethod
    def _local_significance_levels(self) -> NDArray:
        pass

    @abstractmethod
    def _update_on_reject(self, hypothesis_index: int):
        pass


class FWERBonferroniCorrection(FWERProcedure):
    """
    Bonferroni procedure for controlling the FWER.

    Each hypothesis is tested independently at level delta / n_lambdas.
    The procedure stops as soon as one hypothesis is not rejected.

    Notes
    -----
    This is the simplest FWER-controlling method. It does not adapt
    to p-values and does not redistribute error budget after rejections.
    """

    def _init_state(self, n_lambdas: int, delta: float):
        self.local_deltas = np.full(n_lambdas, delta / n_lambdas)
        self.active_hypotheses: NDArray[np.bool_] = np.ones(n_lambdas, dtype=bool)

    def _select_next_hypothesis(self, p_values: NDArray) -> Union[int, None]:
        active_indices = np.flatnonzero(self.active_hypotheses)
        return None if len(active_indices) == 0 else active_indices[0]

    def _local_significance_levels(self) -> NDArray:
        return self.local_deltas

    def _update_on_reject(self, hypothesis_index: int):
        self.active_hypotheses[hypothesis_index] = False


class FWERBonferroniHolm(FWERProcedure):
    """
    Holm step-down procedure for controlling the FWER.

    At each step, the hypothesis with the smallest p-value among the
    remaining ones is tested at level delta / k, where k is the number
    of hypotheses still active.

    The procedure stops when the current hypothesis is not rejected.

    Notes
    -----
    This method strictly dominates Bonferroni in power while preserving
    strong FWER control.
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
    for controlling the Family-Wise Error Rate (FWER).

    Hypotheses are assumed to be ordered according to a parameter grid
    such that rejection becomes progressively easier along the sequence.

    If multiple starts are used, each start explores a disjoint segment
    of hypotheses. Starts falling inside already rejected regions are
    automatically discarded.

    Parameters
    ----------
    n_starts : int, default=1
        Number of equally spaced starting points used in the multi-start procedure.
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

        self.start_positions = list(
            np.linspace(0, n_lambdas - 1, self._effective_starts, dtype=int)
        )

    def _select_next_hypothesis(self, p_values):
        if len(self.start_positions) == 0:
            return None

        return min(self.start_positions)

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


def _build_fwer(
    method: Union[
        Literal[
            "bonferroni",
            "fixed_sequence",
            "bonferroni_holm",
        ],
        FWERProcedure,
    ],
) -> FWERProcedure:
    """
    Build an instance of FWERProcedure based on the specified method.

    Parameters
    ----------
    method : {"bonferroni", "fixed_sequence", "bonferroni_holm"}, or FWERProcedure instance
        FWER control strategy. If a string is provided, it must be one of the supported methods.
        If an instance of FWERProcedure is provided, it will be used directly.

    Notes
    -----
    When method is "fixed_sequence", the number of starts is set to 1 by default.
    However, users can use multi-start by instantiating FWERFixedSequenceTesting with
    any desired number of starts and passing the instance to control_fwer.
    """
    if isinstance(method, FWERProcedure):
        return method

    if method == "bonferroni":
        return FWERBonferroniCorrection()

    if method == "fixed_sequence":
        return FWERFixedSequenceTesting(n_starts=1)

    if method == "bonferroni_holm":
        return FWERBonferroniHolm()

    raise ValueError(
        f"Unknown FWER control method: {method}. "
        "Supported methods are {'bonferroni','fixed_sequence','bonferroni_holm'}, "
        "or an instance of FWERProcedure."
    )


def control_fwer(
    p_values: NDArray,
    delta: float,
    fwer_method: Union[
        Literal[
            "bonferroni",
            "fixed_sequence",
            "bonferroni_holm",
        ],
        FWERProcedure,
    ] = "bonferroni",
) -> NDArray:
    """
    Apply a Family-Wise Error Rate (FWER) control procedure.

    This function applies a multiple testing correction to a collection
    of p-values in order to control the family-wise error rate (FWER)
    at level ``delta``.

    The correction method is selected via the ``fwer_method`` argument.

    Supported methods are:
    - ``"bonferroni"``: classical Bonferroni correction,
    - ``"fixed_sequence"``: Fixed Sequence Testing (FST),
    - ``"bonferroni_holm"``: Sequential Graphical Testing corresponding
      to the Bonferroni-Holm procedure.
    - Custom procedures can also be implemented by subclassing ``FWERProcedure``
      and passing an instance to ``fwer_method``.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with each tested hypothesis.
    delta : float
        Target family-wise error rate. Must be in (0, 1].
    fwer_method : {"bonferroni", "fixed_sequence", "bonferroni_holm"} or FWERProcedure instance, default="bonferroni"
        FWER control strategy.
    **fwer_kwargs
        Additional keyword arguments used only when ``fwer_method="fixed_sequence"``.
        Currently supported keyword:
        - ``n_starts`` (int): number of equally spaced starting points used in
          the multi-start Fixed Sequence Testing procedure.

    Returns
    -------
    valid_index : NDArray
        Sorted indices of hypotheses rejected under FWER control.

    Notes
    -----
    This function is a thin dispatcher that instantiates the requested
    FWERProcedure and executes it.
    """
    p_values = np.asarray(p_values, dtype=float)
    n_lambdas = len(p_values)

    if n_lambdas == 0:
        raise ValueError("p_values must be non-empty.")
    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0, 1].")

    procedure = _build_fwer(method=fwer_method)

    return procedure.run(p_values, delta)
