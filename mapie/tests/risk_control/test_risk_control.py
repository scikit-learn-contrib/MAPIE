"""
Testing for risk_control module.
Testing for now risks for multilabel classification
"""

from typing import List, Union

import numpy as np
import pytest
from numpy.typing import NDArray

from mapie.risk_control.methods import (
    compute_hoeffding_bentkus_p_value,
    find_precision_lambda_star,
    ltt_procedure,
)
from mapie.risk_control.risks import compute_risk_precision, compute_risk_recall

lambdas = np.array([0.5, 0.9])

y_toy = np.stack(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]
)

y_preds_proba = np.stack(
    [[0.2, 0.6, 0.9], [0.8, 0.2, 0.6], [0.4, 0.8, 0.1], [0.6, 0.8, 0.7]]
)

y_preds_proba = np.expand_dims(y_preds_proba, axis=2)

test_recall = np.array([[1 / 2, 1.0], [1.0, 1.0], [1 / 2, 1.0], [0.0, 1.0]])

test_precision = np.array([[1 / 2, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

r_hat = np.array([[0.5, 0.8]])

n = np.array([[1100]])

alpha = np.array([[0.6]])

valid_index = [[0, 1]]

wrong_alpha = 0

wrong_alpha_shape = np.array([[0.1, 0.2], [0.3, 0.4]])

random_state = 42
prng = np.random.RandomState(random_state)


def test_compute_recall_equal() -> None:
    """Test that compute_recall give good result"""
    recall = compute_risk_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(recall, test_recall)


def test_compute_precision() -> None:
    """Test that compute_precision give good result"""
    precision = compute_risk_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(precision, test_precision)


def test_recall_with_zero_sum_is_equal_nan() -> None:
    """Test compute_recall with nan values"""
    y_toy = np.zeros((4, 3))
    y_preds_proba = prng.rand(4, 3, 1)
    with pytest.warns(RuntimeWarning, match=r".*invalid value encountered.*"):
        recall = compute_risk_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_array_equal(recall, np.full_like(recall, np.nan))


def test_precision_with_zero_sum_is_equal_ones() -> None:
    """Test compute_precision with nan values"""
    y_toy = prng.rand(4, 3)
    y_preds_proba = np.zeros((4, 3, 1))
    precision = compute_risk_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_array_equal(precision, np.ones_like(precision))


def test_compute_recall_shape() -> None:
    """Test shape when using _compute_recall"""
    recall = compute_risk_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(recall.shape, test_recall.shape)


def test_compute_precision_shape() -> None:
    """Test shape when using _compute_precision"""
    precision = compute_risk_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(precision.shape, test_precision.shape)


def test_compute_recall_with_wrong_shape() -> None:
    """Test error when wrong shape in _compute_recall"""
    with pytest.raises(ValueError, match=r".*y_pred_proba should be a 3d*"):
        compute_risk_recall(lambdas, y_preds_proba.squeeze(), y_toy)
    with pytest.raises(ValueError, match=r".*y should be a 2d*"):
        compute_risk_recall(lambdas, y_preds_proba, np.expand_dims(y_toy, 2))
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        compute_risk_recall(lambdas, y_preds_proba, y_toy[:-1])


def test_compute_precision_with_wrong_shape() -> None:
    """Test shape when using _compute_precision"""
    with pytest.raises(ValueError, match=r".*y_pred_proba should be a 3d*"):
        compute_risk_precision(lambdas, y_preds_proba.squeeze(), y_toy)
    with pytest.raises(ValueError, match=r".*y should be a 2d*"):
        compute_risk_precision(lambdas, y_preds_proba, np.expand_dims(y_toy, 2))
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        compute_risk_precision(lambdas, y_preds_proba, y_toy[:-1])


@pytest.mark.parametrize("alpha", [0.5, [0.5], [0.5, 0.9]])
def test_p_values_different_alpha(alpha: Union[float, NDArray]) -> None:
    """Test type for different alpha for p_values"""
    result = compute_hoeffding_bentkus_p_value(r_hat[0], n[0], alpha)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("delta", [0.1, 0.2])
def test_ltt_different_delta(delta: float) -> None:
    """Test _ltt_procedure for different delta"""
    assert ltt_procedure(r_hat, alpha, delta, n)


def test_find_precision_lambda_star() -> None:
    """Test _find_precision_lambda_star"""
    assert find_precision_lambda_star(r_hat, valid_index, lambdas)


@pytest.mark.parametrize("delta", [0.1, 0.8])
@pytest.mark.parametrize("alpha", [np.array([[0.5]]), np.array([[0.6, 0.8]])])
def test_ltt_type_output_alpha_delta(alpha: NDArray, delta: float) -> None:
    """Test type output _ltt_procedure"""
    valid_index = ltt_procedure(r_hat, alpha, delta, n)
    assert isinstance(valid_index, list)


@pytest.mark.parametrize("valid_index", [[[0, 1]]])
def test_find_precision_lambda_star_output(valid_index: List[List[int]]) -> None:
    """Test _find_precision_lambda_star with a list of list"""
    assert find_precision_lambda_star(r_hat, valid_index, lambdas)


def test_warning_valid_index_empty() -> None:
    """Test warning sent when empty list"""
    valid_index = [[]]  # type: List[List[int]]
    with pytest.warns(UserWarning, match=r".*Warning: the risk couldn'*"):
        find_precision_lambda_star(r_hat, valid_index, lambdas)


def test_invalid_alpha_hb() -> None:
    """Test error message when invalid alpha"""
    with pytest.raises(ValueError, match=r".*Invalid confidence_level"):
        compute_hoeffding_bentkus_p_value(r_hat, n, wrong_alpha)


def test_invalid_shape_alpha_hb() -> None:
    """Test error message when invalid alpha shape"""
    with pytest.raises(ValueError, match=r".*Invalid confidence_level"):
        compute_hoeffding_bentkus_p_value(r_hat, n, wrong_alpha_shape)


def test_hb_p_values_n_obs_int_vs_array() -> None:
    """Test that using n_obs as an array gives the same values as an int"""
    r_hat = np.array([0.5, 0.8])
    n_obs = np.array([1100, 1200])
    alpha = np.array([0.6, 0.7])

    pval_0 = compute_hoeffding_bentkus_p_value(
        np.array([r_hat[0]]), int(n_obs[0]), alpha
    )
    pval_1 = compute_hoeffding_bentkus_p_value(
        np.array([r_hat[1]]), int(n_obs[1]), alpha
    )
    pval_manual = np.vstack([pval_0, pval_1])

    pval_array = compute_hoeffding_bentkus_p_value(r_hat, n_obs, alpha)

    np.testing.assert_allclose(pval_manual, pval_array, rtol=1e-12)


def test_ltt_procedure_n_obs_negative() -> None:
    """
    Test ltt_procedure with negative n_obs.
     This happens when the risk, defined as the conditional expectation of
     a loss, is undefined because the condition is never met.
     This should return an invalid lambda.
    """
    r_hat = np.array([[0.5]])
    n_obs = np.array([[-1]])
    alpha_np = np.array([[0.6]])
    binary = True
    assert ltt_procedure(r_hat, alpha_np, 0.1, n_obs, binary) == [[]]


def test_ltt_multi_risk() -> None:
    """Test _ltt_procedure for multi risk scenario"""
    assert ltt_procedure(
        np.repeat(r_hat, 2, axis=0),
        np.repeat(alpha, 2, axis=0),
        0.1,
        np.repeat(n, 2, axis=0),
    )


def test_ltt_multi_risk_error() -> None:
    """Test _ltt_procedure for multi risk scenario error where n_risks differ"""
    with pytest.raises(
        ValueError, match=r"r_hat, n_obs, and alpha_np must have the same length."
    ):
        ltt_procedure(
            np.repeat(r_hat, 2, axis=0),
            np.repeat(alpha, 1, axis=0),
            0.1,
            np.repeat(n, 2, axis=0),
        )
