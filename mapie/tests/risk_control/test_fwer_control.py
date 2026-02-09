import numpy as np
import pytest

from mapie.risk_control import control_fwer, fst_ascending_multistart
from mapie.risk_control.fwer_control import FWERGraph


def test_fwer_control_bonferroni():
    p_values = np.array([0.001, 0.02, 0.2, 0.8])
    delta = 0.05

    valid_index = control_fwer(
        p_values=p_values,
        delta=delta,
        fwer_graph="bonferroni",
    )
    assert np.array_equal(valid_index, np.array([0]))


def test_fst_multistart_multiple_starts():
    p_values = np.array([0.001, 0.003, 0.01, 0.02, 0.2, 0.6])
    delta = 0.1
    n_starts = 3
    rejected = fst_ascending_multistart(p_values, delta, n_starts)
    assert rejected.tolist() == [0, 1, 2, 3]


def test_fst_multistart_empty_pvalues():
    with pytest.raises(ValueError, match=r"p_values must be non-empty."):
        fst_ascending_multistart(np.array([]), delta=0.1)


def test_fst_multistart_invalid_n_starts():
    p_values = np.array([0.01, 0.02])
    with pytest.raises(ValueError, match=r".*n_starts must be a positive integer.*"):
        fst_ascending_multistart(p_values, delta=0.1, n_starts=0)


def test_fwer_control_wrong_graph():
    p_values = np.array([0.001, 0.02, 0.2, 0.8])
    delta = 0.05
    with pytest.raises(ValueError, match="Unknown FWER control strategy:"):
        control_fwer(p_values, delta, fwer_graph="invalid_strategy")


def test_fwer_control_wrong_graph_type():
    p_values = np.array([0.001, 0.02, 0.2, 0.8])
    delta = 0.05
    with pytest.raises(
        ValueError,
        match="fwer_graph must be either a string or an instance of FWERGraph.",
    ):
        control_fwer(p_values, delta, fwer_graph=123)


def test_graph_correct_init():
    delta = np.array([0.5, 0.5])
    transition_matrix = np.array([[0, 1], [0, 0]])
    graph = FWERGraph(delta, transition_matrix)
    assert np.array_equal(graph.delta_np, delta)
    assert np.array_equal(graph.W, transition_matrix)


def test_graph_init_wrong_delta():
    delta = np.array([0.5, 0.1])
    transition_matrix = np.array([[0, 1], [0, 0]])
    with pytest.raises(ValueError, match="Initial risk budgets must sum to 1."):
        FWERGraph(delta, transition_matrix)


def test_graph_init_negative_transition():
    delta = np.array([0.5, 0.5])
    transition_matrix = np.array([[0, -1], [0, 0]])
    with pytest.raises(ValueError, match="Transition matrix must be non-negative."):
        FWERGraph(delta, transition_matrix)


def test_graph_init_row_sum_exceeds_one():
    delta = np.array([0.5, 0.5])
    transition_matrix = np.array([[0, 1.5], [0, 0]])
    with pytest.raises(ValueError, match="Row sums of transition matrix must be <= 1."):
        FWERGraph(delta, transition_matrix)
