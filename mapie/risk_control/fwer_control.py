from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray


def fwer_control(
    p_values: NDArray,
    delta: float,
    fwer_graph: Union["FWERGraph", Literal["bonferroni"], Literal["fst_ascending"]],
    lambdas: Optional[NDArray] = None,
) -> NDArray:
    """
    Apply a Family-Wise Error Rate (FWER) control procedure.

    This function applies a multiple testing correction to a collection
    of p-values in order to control the family-wise error rate (FWER)
    at level ``delta``.

    Depending on the value of ``fwer_graph``, the correction can be:
    - a standard Bonferroni correction,
    - a fixed-sequence testing (ascending) procedure,
    - or a general graphical FWER control procedure.

    Parameters
    ----------
    p_values : NDArray of shape (n_lambdas,)
        P-values associated with each tested hypothesis (lambda).
    delta : float
        Target family-wise error rate.
    fwer_graph : Union[FWERGraph, {"bonferroni", "fst_ascending"}]
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

    if isinstance(fwer_graph, str):
        if fwer_graph == "bonferroni":
            threshold = delta / n_lambdas
            valid_index = np.nonzero(p_values <= threshold)[0]
            return valid_index

        elif fwer_graph == "fst_ascending":
            # Placeholder for Fixed Sequence Testing (ascending)
            raise NotImplementedError(
                "Fixed Sequence Testing (ascending) is not implemented yet."
            )

        else:
            raise ValueError(f"Unknown FWER control strategy: {fwer_graph}")

    # Generic graphical FWER control
    graph = fwer_graph
    graph.reset()

    rejected: List[int] = []
    remaining_p_values = p_values.copy()

    while True:
        # Local significance levels for each hypothesis
        local_alpha = delta * graph.delta_np

        # Identify rejectable hypotheses
        candidates = np.where(remaining_p_values <= local_alpha)[0]
        if len(candidates) == 0:
            break

        # Reject the hypothesis with the smallest p-value
        idx = candidates[np.argmin(remaining_p_values[candidates])]
        rejected.append(idx)

        # Update the graph and remove the rejected hypothesis
        graph.step(idx)
        remaining_p_values[idx] = np.inf

    return np.array(rejected, dtype=int)


class FWERGraph(ABC):
    """
    Abstract base class for graphical Family-Wise Error Rate (FWER)
    control procedures.

    A graphical FWER procedure represents a multiple testing strategy
    through:
    - an initial allocation of the global error budget across hypotheses,
    - and a set of transition weights defining how the error budget
      is redistributed after each rejection.

    This formulation follows the graphical approach introduced in
    Bretz et al. (2009).

    Subclasses must implement the methods ``reset`` and ``step``
    to define how the graph state is initialized and updated.

    Parameters
    ----------
    delta_np : NDArray of shape (n_hypotheses,)
        Initial allocation of the FWER budget across hypotheses.
        The values must sum to 1.
    transition_matrix : NDArray of shape (n_hypotheses, n_hypotheses)
        Transition matrix defining how the error budget of a rejected
        hypothesis is redistributed to the remaining hypotheses.
        Each row must sum to a value less than or equal to 1.
    """

    def __init__(
        self,
        delta_np: NDArray,
        transition_matrix: NDArray,
    ):
        self.delta_np = delta_np.astype(float)
        self.W = transition_matrix.astype(float)
        self._check_valid_graph()

    def _check_valid_graph(self) -> None:
        """
        Validate the structure of the graphical FWER procedure.

        Raises
        ------
        ValueError
            If the initial risk budgets do not sum to 1,
            if the transition matrix contains negative values,
            or if any row of the transition matrix sums to more than 1.
        """
        if not np.isclose(self.delta_np.sum(), 1.0):
            raise ValueError("Initial risk budgets must sum to 1.")
        if np.any(self.W < 0):
            raise ValueError("Transition matrix must be non-negative.")
        if np.any(self.W.sum(axis=1) > 1):
            raise ValueError("Row sums of transition matrix must be <= 1.")

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the graph to its initial state.

        This method is called before starting a new FWER control
        procedure.
        """

    @abstractmethod
    def step(self, rejected_index: int) -> None:
        """
        Update the graph after rejection of a hypothesis.

        Parameters
        ----------
        rejected_index : int
            Index of the rejected hypothesis.
        """


class FixedSequenceGraph(FWERGraph):
    """
    Graphical representation of a Fixed Sequence Testing procedure.
    """

    def reset(self) -> None:
        raise NotImplementedError

    def step(self, rejected_index: int) -> None:
        raise NotImplementedError


class FallbackGraph(FWERGraph):
    """
    Graphical representation of a Fallback FWER control procedure.
    """

    def reset(self) -> None:
        raise NotImplementedError

    def step(self, rejected_index: int) -> None:
        raise NotImplementedError


class HolmGraph(FWERGraph):
    """
    Graphical representation of the Holm step-down procedure.
    """

    def reset(self) -> None:
        raise NotImplementedError

    def step(self, rejected_index: int) -> None:
        raise NotImplementedError
