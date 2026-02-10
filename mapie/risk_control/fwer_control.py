import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def control_fwer(
    p_values: NDArray,
    delta: float,
    fwer_method: Literal[
        "bonferroni",
        "fst_ascending",
        "sgt_bonferroni_holm",
    ] = "bonferroni",
    **kwargs,
) -> NDArray:
    """
    Apply a Family-Wise Error Rate (FWER) control procedure.

    This function applies a multiple testing correction to a collection
    of p-values in order to control the family-wise error rate (FWER)
    at level ``delta``.

    Depending on the value of ``fwer_method``, the correction can be:
    - a standard Bonferroni correction,
    - a fixed-sequence testing (ascending) procedure,
    - or a general graphical FWER control procedure.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with each tested hypothesis (lambda).
    delta : float
        Target family-wise error rate.
    fwer_method : Union[FWERGraph, {"bonferroni", "fst_ascending"}]
        FWER control strategy. Either a predefined string strategy
        ("bonferroni", "fst_ascending") or a custom graphical procedure.
    lambdas : Optional[NDArray], default=None
        Optional array of tested parameters (λ). This argument is not
        used by the procedure itself and is only provided for traceability
        or debugging purposes.

    Returns
    -------
    valid_index : NDArray
        Indices of hypotheses (lambdas) rejected under FWER control.
    """
    p_values = np.asarray(p_values, dtype=float)
    n_lambdas = len(p_values)

    if isinstance(fwer_method, str):
        if fwer_method == "bonferroni":
            threshold = delta / n_lambdas
            valid_index = np.nonzero(p_values <= threshold)[0]
            return valid_index

        #     elif fwer_method == "fst_ascending":
        #         # Placeholder for Fixed Sequence Testing (ascending)
        #         raise NotImplementedError(
        #             "Fixed Sequence Testing (ascending) is not implemented yet."
        #         )

        else:
            raise ValueError(f"Unknown FWER control strategy: {fwer_method}")
    # elif isinstance(fwer_method, FWERGraph):
    # # Generic graphical FWER control
    # graph = fwer_method
    # graph.reset()

    # rejected: List[int] = []
    # remaining_p_values = p_values.copy()

    # while True:
    #     # Local significance levels for each hypothesis
    #     local_alpha = delta * graph.delta_np

    #     # Identify rejectable hypotheses
    #     candidates = np.where(remaining_p_values <= local_alpha)[0]
    #     if len(candidates) == 0:
    #         break

    #     # Reject the hypothesis with the smallest p-value
    #     idx = candidates[np.argmin(remaining_p_values[candidates])]
    #     rejected.append(idx)

    #     # Update the graph and remove the rejected hypothesis
    #     graph.step(idx)
    #     remaining_p_values[idx] = np.inf

    # return np.array(rejected, dtype=int)
    else:
        raise ValueError(
            "fwer_method must be either a string or an instance of FWERGraph."
        )


def fst_ascending_multistart(
    p_values: NDArray,
    delta: float,
    n_starts: int = 20,
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
    n_starts : int, default=20
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
    n_lambdas = len(p_values)

    if n_lambdas == 0:
        raise ValueError("p_values must be non-empty.")
    if not (0 < delta <= 1):
        raise ValueError("delta must be in (0, 1].")

    active_hypotheses: NDArray[np.bool_] = np.ones(n_lambdas, dtype=bool)
    local_delta = np.full(n_lambdas, delta / n_lambdas, dtype=float)

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
