import warnings
from abc import ABC, abstractmethod
from typing import Literal, Union

import numpy as np
from numpy.typing import NDArray


def control_fwer(
    p_values: NDArray,
    delta: float,
    fwer_method: Literal[
        "bonferroni",
        "fst_ascending",
        "bonferroni_holm",
    ] = "bonferroni",
    **fwer_kwargs,
) -> NDArray:
    """
    Apply a Family-Wise Error Rate (FWER) control procedure.

    This function applies a multiple testing correction to a collection
    of p-values in order to control the family-wise error rate (FWER)
    at level ``delta``.

    The correction method is selected via the ``fwer_method`` argument.

    Supported methods are:
    - ``"bonferroni"``: classical Bonferroni correction,
    - ``"fst_ascending"``: Fixed Sequence Testing (ascending, multi-start),
    - ``"bonferroni_holm"``: Sequential Graphical Testing corresponding
      to the Bonferroni-Holm procedure.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with each tested hypothesis.
    delta : float
        Target family-wise error rate. Must be in (0, 1].
    fwer_method : {"bonferroni", "fst_ascending", "bonferroni_holm"}, default="bonferroni"
        FWER control strategy.
    **fwer_kwargs
        Additional keyword arguments used only when ``fwer_method="fst_ascending"``.
        Currently supported keyword:
        - ``n_starts`` (int): number of equally spaced starting points used in
          the multi-start Fixed Sequence Testing procedure.

    Returns
    -------
    valid_index : NDArray
        Sorted indices of hypotheses rejected under FWER control.
    """
    p_values = np.asarray(p_values, dtype=float)
    n_lambdas = len(p_values)

    if n_lambdas == 0:
        raise ValueError("p_values must be non-empty.")
    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0, 1].")

    if fwer_method == "bonferroni":
        threshold = delta / n_lambdas
        valid_index = np.nonzero(p_values <= threshold)[0]
        return valid_index

    if fwer_method == "fst_ascending":
        return fst_ascending(p_values, delta, **fwer_kwargs)

    if fwer_method == "bonferroni_holm":
        return sgt_bonferroni_holm(p_values, delta)

    raise ValueError(
        f"Unknown FWER control method: {fwer_method}. "
        "Supported methods are {'bonferroni', 'fst_ascending', 'bonferroni_holm'}."
    )


def fst_ascending(
    p_values: NDArray,
    delta: float,
    n_starts: int = 1,
) -> NDArray:
    """
    Apply Fixed Sequential Testing (FST) with multi-start to control
    the Family-Wise Error Rate (FWER).

    This procedure tests hypotheses sequentially starting from multiple
    equally spaced entry points along the ordered p_values according to
    the hypothesis order grid. For each starting point, hypotheses are
    tested in ascending order until a p-value exceeds the locally
    adjusted significance level.

    The final rejection set is defined as the union of all hypotheses
    rejected across the different starting points.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with the hypotheses, ordered according to
        the lambda grid (from most conservative to least conservative).
    delta : float
        Target family-wise error rate.
    n_starts : int, default=1
        Number of equally spaced starting points used in the multi-start procedure.

    Returns
    -------
    valid_index : NDArray
        Sorted indices of hypotheses rejected under FWER control.
        It contains the indices of valid lambdas for which the null hypothesis is rejected.

    Notes
    -----
    This procedure assumes that the hypotheses are ordered according to a
    parameter grid such that the associated risk is monotonic along this
    ordering. In particular, the null hypotheses are assumed to become
    progressively easier to reject when moving forward along the grid,
    which justifies the sequential testing strategy.
    """
    p_values = np.asarray(p_values, dtype=float)
    p_values = np.nan_to_num(
        p_values, nan=1.0
    )  # NaN p-values are treated as non-significant
    n_lambdas = len(p_values)

    if n_lambdas == 0:
        raise ValueError("p_values must be non-empty.")
    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0, 1].")
    if n_starts <= 0:
        raise ValueError("n_starts must be a positive integer.")
    if n_starts > n_lambdas:
        warnings.warn(
            "n_starts is greater than the number of tests (n_lambdas). "
            "Hence, it will be set to n_lambdas.",
            UserWarning,
        )
        n_starts = n_lambdas

    start_indices = np.linspace(0, n_lambdas - 1, n_starts, dtype=int)
    rejected = set()

    for j in start_indices:
        if j in rejected:
            continue
        while j < n_lambdas and p_values[j] <= delta / n_starts:
            rejected.add(j)
            j += 1

    return np.array(sorted(rejected), dtype=int)


def sgt_bonferroni_holm(
    p_values: NDArray,
    delta: float,
) -> NDArray:
    """
    Apply Sequential Graphical Testing (SGT) with Bonferroni-Holm
    correction to control the Family-Wise Error Rate (FWER).

    This procedure implements the Bonferroni-Holm method as a special
    case of Sequential Graphical Testing. Each hypothesis is associated
    with an initial local significance level equal to delta / n_lambdas, where
    n_lambdas is the number of hypotheses. Hypotheses are tested sequentially,
    and whenever a hypothesis is rejected, its local significance level
    is redistributed uniformly among the remaining hypotheses.

    At each step, the hypothesis with the smallest p-value among the
    remaining ones is tested against its current local significance
    level. The procedure stops when no further rejection is possible.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with the hypotheses.
    delta : float
        Target family-wise error rate.

    Returns
    -------
    valid_index : NDArray
        Sorted indices of hypotheses rejected under FWER control.
        It contains the indices of valid lambdas for which the null hypothesis is rejected.

    Notes
    -----
    This procedure is equivalent to the classical Bonferroni-Holm
    correction, but expressed in the Sequential Graphical Testing
    framework. It dynamically redistributes the error budget after
    each rejection, while preserving the overall FWER guarantee.

    The total allocated significance level is conserved at each step,
    i.e., the sum of the local significance levels always equals delta.
    """
    p_values = np.asarray(p_values, dtype=float)
    p_values = np.nan_to_num(
        p_values, nan=1.0
    )  # NaN p-values are treated as non-significant
    n_lambdas = len(p_values)

    if n_lambdas == 0:
        raise ValueError("p_values must be non-empty.")
    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0, 1].")

    active_hypotheses: NDArray[np.bool_] = np.ones(n_lambdas, dtype=bool)
    local_delta = np.full(n_lambdas, delta / n_lambdas)

    rejected_indices = set()

    while True:
        remaining_indices = np.where(active_hypotheses)[0]
        if len(remaining_indices) == 0:
            break

        i = remaining_indices[np.argmin(p_values[remaining_indices])]
        if p_values[i] > local_delta[i]:
            break

        rejected_indices.add(i)

        released_delta = local_delta[i]
        local_delta[i] = 0.0
        active_hypotheses[i] = False

        remaining_indices = np.where(active_hypotheses)[0]
        if len(remaining_indices) > 0:
            local_delta[remaining_indices] += released_delta / len(remaining_indices)
    return np.array(sorted(rejected_indices), dtype=int)


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

    Parameters
    ----------
    n_starts : int, default=1
        Number of equally spaced starting points used in the multi-start procedure.
    """

    def __init__(self, n_starts: int = 1):
        self.n_starts = n_starts

    def run(self, p_values: NDArray, delta: float) -> NDArray[np.int_]:
        """
        Apply Fixed Sequential Testing (FST) with multi-start to control
        the family-wise error rate.

        This procedure tests hypotheses sequentially starting from multiple
        equally spaced entry points along the ordered ``p_values``.
        For each starting point, hypotheses are tested in ascending index
        order until a p-value exceeds the locally adjusted significance level.

        The final rejection set is defined as the union of all hypotheses
        rejected across the different starting points.

        Parameters
        ----------
        p_values : NDArray of shape (n_lambdas,)
            P-values associated with the hypotheses, ordered according to
            the lambda grid (from most conservative to least conservative).
        delta : float
            Target family-wise error rate.

        Returns
        -------
        NDArray[int]
            Sorted indices of hypotheses rejected under FWER control.
            These correspond to valid grid positions where the null
            hypothesis is rejected.

        Notes
        -----
        This procedure assumes that hypotheses are ordered so that their
        associated risk is monotonic along the grid. In particular,
        null hypotheses are assumed to become progressively easier to reject
        as the index increases, which justifies the sequential testing rule.

        NaN p-values are treated as non-significant and replaced by 1.0.
        """

        p_values = np.asarray(p_values, dtype=float)
        p_values = np.nan_to_num(
            p_values, nan=1.0
        )  # NaN p-values are treated as non-significant
        n_lambdas = len(p_values)

        if n_lambdas == 0:
            raise ValueError("p_values must be non-empty.")
        if not (0 < delta <= 1):
            raise ValueError("delta must be in (0, 1].")
        if self.n_starts <= 0:
            raise ValueError("n_starts must be a positive integer.")

        n_starts = min(self.n_starts, n_lambdas)

        start_indices = np.linspace(0, n_lambdas - 1, n_starts, dtype=int)

        rejected = set()
        local_delta = delta / n_starts

        for j in start_indices:
            if j in rejected:
                continue

            while j < n_lambdas and p_values[j] <= local_delta:
                rejected.add(j)
                j += 1

        return np.array(sorted(rejected), dtype=int)

    def _init_state(self, n_hypotheses: int, delta: float):
        pass

    def _select_next_hypothesis(self, p_values):
        pass

    def _local_significance_levels(self):
        pass

    def _update_on_reject(self, hypothesis_index: int):
        pass
