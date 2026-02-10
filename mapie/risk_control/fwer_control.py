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

    The correction method is selected via the ``fwer_method`` argument.

    Supported methods are:
    - ``"bonferroni"``: classical Bonferroni correction,
    - ``"fst_ascending"``: Fixed Sequence Testing (ascending, multi-start),
    - ``"sgt_bonferroni_holm"``: Sequential Graphical Testing corresponding
      to the Bonferroni-Holm procedure.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with each tested hypothesis.
    delta : float
        Target family-wise error rate. Must be in (0, 1].
    fwer_method : {"bonferroni", "fst_ascending", "sgt_bonferroni_holm"}, default="bonferroni"
        FWER control strategy.
    **kwargs
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
        return fst_ascending(p_values, delta, **kwargs)

    if fwer_method == "sgt_bonferroni_holm":
        return sgt_bonferroni_holm(p_values, delta)

    raise ValueError(
        f"Unknown FWER control method: {fwer_method}. "
        "Supported methods are {'bonferroni', 'fst_ascending', 'sgt_bonferroni_holm'}."
    )


def fst_ascending(
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
