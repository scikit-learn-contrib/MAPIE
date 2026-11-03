"""
Testing for risk_control module.
Testing for now risks for multilabel classification
"""

from typing import List, Union

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import binom

from mapie.risk_control.fwer_control import FWERFixedSequenceTesting
from mapie.risk_control.methods import (
    _check_risk_monotonicity,
    compute_hoeffding_bentkus_p_value,
    find_precision_best_predict_param,
    get_r_hat_plus,
)
from mapie.risk_control._base_controller import ltt_procedure
from mapie.risk_control import abstention_rate
from mapie.risk_control.risks import (
    ClassSpecificRisk,
    build_recall_ovr,
)

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


@pytest.mark.parametrize("alpha", [0.5, [0.5], [0.5, 0.9]])
def test_p_values_different_alpha(alpha: Union[float, NDArray]) -> None:
    """Test type for different alpha for p_values"""
    result = compute_hoeffding_bentkus_p_value(r_hat[0], n[0], alpha)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("delta", [0.1, 0.2])
def test_ltt_different_delta(delta: float) -> None:
    """Test _ltt_procedure for different delta"""
    assert ltt_procedure(r_hat, alpha, delta, n)


def test_find_precision_best_predict_param() -> None:
    """Test _find_precision_best_predict_param"""
    assert find_precision_best_predict_param(r_hat, valid_index, lambdas)


@pytest.mark.parametrize("delta", [0.1, 0.8])
@pytest.mark.parametrize("alpha", [np.array([[0.5]]), np.array([[0.6, 0.8]])])
def test_ltt_type_output_alpha_delta(alpha: NDArray, delta: float) -> None:
    """Test type output _ltt_procedure"""
    valid_index, _ = ltt_procedure(r_hat, alpha, delta, n)
    assert isinstance(valid_index, list)


@pytest.mark.parametrize("valid_index", [[[0, 1]]])
def test_find_precision_best_predict_param_output(valid_index: List[List[int]]) -> None:
    """Test _find_precision_best_predict_param with a list of list"""
    assert find_precision_best_predict_param(r_hat, valid_index, lambdas)


def test_warning_valid_index_empty() -> None:
    """Test warning sent when empty list"""
    valid_index = [[]]  # type: List[List[int]]
    with pytest.warns(UserWarning, match=r".*Warning: the risk couldn'*"):
        find_precision_best_predict_param(r_hat, valid_index, lambdas)


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


@pytest.mark.parametrize(
    "risk_hat, binary",
    [
        (r, b)
        for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for b in [True, False]
    ],
)
def test_computed_hb_p_value_matches_manual_calculation(risk_hat, binary):
    """
    Test that `compute_hoeffding_bentkus_p_value` returns the same result
    as the manual implementation of the Hoeffding–Bentkus bound term
    for different risk levels and binary/non-binary settings.
    """
    n = 10
    alpha = 0.1

    # manual calculation
    factor = 1 if binary else np.e

    def h1(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    term1 = np.exp(-n * h1(min(risk_hat, alpha), alpha))
    term2 = factor * binom.cdf(np.ceil(n * risk_hat), n, alpha)
    manual_bound = min(term1, term2)

    # function calculation
    hb_bound = compute_hoeffding_bentkus_p_value(
        r_hat=np.array([risk_hat]),
        n_obs=n,
        alpha=alpha,
        binary=binary,
    )

    assert np.isclose(manual_bound, hb_bound, atol=1e-12)


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
    valid_index, _ = ltt_procedure(r_hat, alpha_np, 0.1, n_obs, binary)
    assert valid_index == [[]]


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


def test_check_risk_monotonicity_all_cases():
    """Test _check_risk_monotonicity for all cases"""
    assert _check_risk_monotonicity(np.array([1, 2, 3])) == "increasing"
    assert _check_risk_monotonicity(np.array([3, 2, 1])) == "decreasing"
    assert _check_risk_monotonicity(np.array([1, 3, 2])) == "none"


def test_ltt_fst_multirisk_error():
    r_hat = np.array([[0.1, 0.2], [0.1, 0.2]])
    n_obs = np.ones_like(r_hat)
    alpha_np = np.array([[0.5], [0.5]])
    delta = 0.1
    with pytest.raises(ValueError, match=r".*fixed_sequence cannot be used.*"):
        ltt_procedure(r_hat, alpha_np, delta, n_obs, fwer_method="fixed_sequence")


@pytest.mark.parametrize(
    "r_hat",
    [
        np.array([[0.1, 0.4, 0.2]]),  # increasing on average
        np.array([[0.4, 0.1, 0.2]]),  # decreasing on average
    ],
)
def test_ltt_fst_non_monotone_warns(r_hat):
    n_obs = np.ones_like(r_hat)
    alpha_np = np.array([[0.5]])

    with pytest.warns(UserWarning, match=r".*requires a monotonic risk.*"):
        ltt_procedure(r_hat, alpha_np, 0.1, n_obs, fwer_method="fixed_sequence")


def test_ltt_fst_increasing():
    r_hat = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    n_obs = np.ones_like(r_hat)
    alpha_np = np.array([[0.1]])

    valid_index_1, _ = ltt_procedure(
        r_hat, alpha_np, 0.1, n_obs, fwer_method="fixed_sequence"
    )
    valid_index_2, _ = ltt_procedure(
        r_hat, alpha_np, 0.1, n_obs, fwer_method=FWERFixedSequenceTesting()
    )
    assert np.array_equal(valid_index_1, valid_index_2)


@pytest.mark.parametrize(
    "class_labels, y_true, y_pred",
    [
        ([0, 1, 2], np.array([0, 1, 2, 1, 0]), np.array([0, 1, 1, 1, 2])),
        (
            ["a", "b"],
            np.array(["a", "b", "a", "b"]),
            np.array(["a", "a", "a", "b"]),
        ),
    ],
)
def test_build_recall_ovr(class_labels, y_true, y_pred):
    """
    build_recall_ovr's ClassSpecificRisk for a given class_label must return
    1 - recall (recall is a performance metric, so higher_is_better=True),
    where recall = (correctly predicted samples among y_true == class_label)
    / (number of samples with y_true == class_label).
    """
    risks = build_recall_ovr(class_labels)
    for risk, class_label in zip(risks, class_labels):
        condition = y_true == class_label
        n_effective = condition.sum()
        recall = (y_pred[condition] == y_true[condition]).sum() / n_effective

        value, n = risk.get_value_and_effective_sample_size(y_true, y_pred)

        assert n == n_effective
        assert value == pytest.approx(1 - recall)


@pytest.mark.parametrize(
    "class_labels, y_pred",
    [
        # int class labels: default_agg_class_pba's np.where upcasts to a
        # plain float array, NaN marking abstention.
        ([0, 1, 2], np.array([0.0, np.nan, 2.0])),
        # str class labels: default_agg_class_pba's np.where keeps an
        # object array mixing strings and NaN, which np.isnan can't handle.
        (
            ["a", "b", "c"],
            np.where(
                [True, False, True],
                np.array(["a", "b", "c"], dtype=object),
                np.nan,
            ),
        ),
    ],
)
def test_abstention_rate(class_labels, y_pred):
    """
    abstention_rate's risk_occurrence must flag abstained (NaN) samples
    without raising, both for a numeric float y_pred (int class labels) and
    for an object-dtype y_pred mixing strings and NaN (str class labels) --
    the exact shapes produced by default_agg_class_pba.
    """
    y_true = np.repeat(class_labels[0], len(y_pred))
    risk = abstention_rate

    value, n = risk.get_value_and_effective_sample_size(y_true, y_pred)

    assert n == len(y_pred)
    assert value == 1 / 3


def test_bernstein_bound_uses_each_lambda_variance() -> None:
    """
    Test that each lambda is used to compute the Bernstein bound.
    If the bound is not computed over every lambda, r_hat_plus will be NaN.
    """
    risks = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]])
    delta = 0.1
    _, r_hat_plus = get_r_hat_plus(
        risks, np.array([0.1]), "rcps", "bernstein", delta, None
    )
    assert not np.isnan(r_hat_plus).any()


def test_ltt_fst_decreasing_reorder():
    r_hat = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    n_obs = np.ones_like(r_hat)
    alpha_np = np.array([[0.1]])

    valid_index_1, _ = ltt_procedure(
        r_hat, alpha_np, 0.1, n_obs, fwer_method="fixed_sequence"
    )
    valid_index_2, _ = ltt_procedure(
        r_hat, alpha_np, 0.1, n_obs, fwer_method=FWERFixedSequenceTesting()
    )
    assert np.array_equal(valid_index_1, valid_index_2)


def test_class_specific_risk_non_binary_occurrence_raises() -> None:
    """
    A ClassSpecificRisk is controlled with the binary Hoeffding-Bentkus
    p-values too, so an occurrence which is not a 0/1/bool indicator must raise.
    """
    bad_risk = ClassSpecificRisk(
        class_label=1,
        risk_occurrence=lambda y_true, y_pred: 0.5 * np.ones(len(y_true)),  # type: ignore[arg-type,return-value]
        risk_condition=lambda y_true, y_pred: np.repeat(True, len(y_true)),
        higher_is_better=False,
    )
    y_true, y_pred = np.array([0, 1, 2, 1]), np.array([0, 1, 1, 1])
    with pytest.raises(ValueError, match=r".*must be binary indicators.*"):
        bad_risk.get_value_and_effective_sample_size(y_true, y_pred)
    with pytest.raises(ValueError, match=r".*must be binary indicators.*"):
        bad_risk.get_risk_sequence(y_true, y_pred)


@pytest.mark.parametrize(
    "y_pred",
    [
        np.array([0.0, np.nan, 2.0]),
        np.where(
            [True, False, True],
            np.array(["a", "b", "c"], dtype=object),
            np.nan,
        ),
    ],
)
def test_abstention_rate_binary_risk(y_pred) -> None:
    """
    The binary abstention_rate risk must flag abstained (NaN) samples for a
    float y_pred as well as for an object-dtype y_pred mixing strings and NaN.
    """
    y_true = np.repeat(y_pred[0], len(y_pred))
    value, n = abstention_rate.get_value_and_effective_sample_size(y_true, y_pred)

    assert n == len(y_pred)
    assert value == 1 / 3
