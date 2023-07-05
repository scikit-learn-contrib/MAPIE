"""
Testing for control_risk module.
Testing for now risks for multilabel classification
"""
import numpy as np
# from numpy.typing import NDArray
from mapie.control_risk.risks import (_compute_precision,
                                      _compute_recall)
import pytest
from typing import Union, List
from numpy.typing import NDArray
from mapie.control_risk.p_values import hoefdding_bentkus_p_value
from mapie.control_risk.ltt import (_ltt_procedure,
                                    _find_lambda_control_star)


lambdas = np.array([0.5, 0.9])
y_toy = np.stack([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
])
y_preds_proba = np.stack([
    [0.2, 0.6, 0.9],
    [0.8, 0.2, 0.6],
    [0.4, 0.8, 0.1],
    [0.6, 0.8, 0.7]
])
y_preds_proba = np.expand_dims(y_preds_proba, axis=2)
test_recall = np.array([
    [1/2, 1.],
    [1., 1.],
    [1/2, 1.],
    [0., 1.]
])
test_precision = np.array([
    [1/2, 1.],
    [1., 1.],
    [0., 1.],
    [0., 1.]
])
r_hat = np.array([
    0.5, 0.8
])
n = 1100
alpha = np.array([0.6])
valid_index = [[
    0, 1
]]
wrong_alpha = 0
wrong_alpha_shape = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])
wrong_delta = None


def test_compute_recall_equal() -> None:
    recall = _compute_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(recall, test_recall)


def test_compute_precision() -> None:
    precision = _compute_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(precision, test_precision)


def test_recall_with_zero_sum_is_equal_nan() -> None:
    y_toy = np.zeros((4, 3))
    y_preds_proba = np.random.rand(4, 3, 1)
    recall = _compute_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_array_equal(recall, np.empty_like(recall))


def test_precision_with_zero_sum_is_equal_ones() -> None:
    y_toy = np.random.rand(4, 3)
    y_preds_proba = np.zeros((4, 3, 1))
    precision = _compute_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_array_equal(precision, np.ones_like(precision))


def test_compute_recall_shape() -> None:
    recall = _compute_recall(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(recall.shape, test_recall.shape)


def test_compute_shape() -> None:
    precision = _compute_precision(lambdas, y_preds_proba, y_toy)
    np.testing.assert_equal(precision.shape, test_precision.shape)


def test_compute_recall_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r".*y_pred_proba should be a 3d*"):
        _compute_recall(lambdas, y_preds_proba.squeeze(), y_toy)
    with pytest.raises(ValueError, match=r".*y should be a 2d*"):
        _compute_recall(lambdas, y_preds_proba, np.expand_dims(y_toy, 2))
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        _compute_recall(lambdas, y_preds_proba, y_toy[:-1])


def test_compute_precision_with_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r".*y_pred_proba should be a 3d*"):
        _compute_precision(lambdas, y_preds_proba.squeeze(), y_toy)
    with pytest.raises(ValueError, match=r".*y should be a 2d*"):
        _compute_precision(lambdas, y_preds_proba, np.expand_dims(y_toy, 2))
    with pytest.raises(ValueError, match=r".*could not be broadcast*"):
        _compute_precision(lambdas, y_preds_proba, y_toy[:-1])


@pytest.mark.parametrize("alpha", [0.5, [0.5], [0.5, 0.9]])
def test_p_values_different_alpha(alpha: Union[float, NDArray]) -> None:
    result = hoefdding_bentkus_p_value(r_hat, n, alpha)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("delta", [0.1, 0.2])
def test_ltt_different_delta(delta: float) -> None:
    assert _ltt_procedure(r_hat, alpha, delta, n)


def test_find_lambda_control_star() -> None:
    assert _find_lambda_control_star(r_hat, valid_index, lambdas)


@pytest.mark.parametrize("delta", [0.1, 0.8])
@pytest.mark.parametrize("alpha", [[0.5], [0.6, 0.8]])
def test_ltt_type_output_alpha_delta(
    alpha: NDArray,
    delta: float
) -> None:
    valid_index, p_values = _ltt_procedure(r_hat, alpha, delta, n)
    assert isinstance(valid_index, list)
    assert isinstance(p_values, np.ndarray)


@pytest.mark.parametrize("valid_index", [[[0, 1]]])
def test_find_lambda_control_star_output(valid_index: List[List[int]]) -> None:
    assert _find_lambda_control_star(r_hat, valid_index, lambdas)


def test_warning_valid_index_empty() -> None:
    valid_index = [[]]  # type: List[List[int]]
    with pytest.warns(UserWarning,
                      match=r".*At least one sequence is empty*"):
        _find_lambda_control_star(r_hat, valid_index, lambdas)


def test_invalid_alpha_hb() -> None:
    with pytest.raises(ValueError, match=r".*Invalid alpha"):
        hoefdding_bentkus_p_value(r_hat, n, wrong_alpha)


def test_invalid_shape_alpha_hb() -> None:
    with pytest.raises(ValueError, match=r".*Invalid alpha"):
        hoefdding_bentkus_p_value(r_hat, n, wrong_alpha_shape)


def test_delta_none_ltt() -> None:
    with pytest.raises(ValueError, match=r".*Invalid delta"):
        _ltt_procedure(r_hat, alpha, wrong_delta, n)
