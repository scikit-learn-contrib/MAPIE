import numpy as np
import pytest

from mapie.risk_control import control_fwer, fst_ascending, sgt_bonferroni_holm


def test_fst_multistart_multiple_starts():
    p_values = np.array([0.001, 0.003, 0.01, 0.02, 0.2, 0.6])
    delta = 0.1
    n_starts = 3
    rejected = fst_ascending(p_values, delta, n_starts=n_starts)
    assert rejected.tolist() == [0, 1, 2, 3]


def test_fst_ascending_invalid_inputs():
    with pytest.raises(ValueError, match=r".*p_values must be non-empty.*"):
        fst_ascending(np.array([]), delta=0.1)

    with pytest.raises(ValueError, match=r".*delta must be in \(0, 1].*"):
        fst_ascending(np.array([0.1, 0.2]), delta=0.0)

    with pytest.raises(ValueError, match=r".*delta must be in \(0, 1].*"):
        fst_ascending(np.array([0.1, 0.2]), delta=1.5)

    with pytest.raises(ValueError, match=r".*n_starts must be a positive integer.*"):
        fst_ascending(np.array([0.1, 0.2]), delta=0.1, n_starts=0)

    with pytest.warns(
        UserWarning, match=r".*n_starts is greater than the number of tests.*"
    ):
        fst_ascending(np.array([0.1, 0.2]), delta=0.1, n_starts=5)


def test_sgt_bonferroni_holm_invalid_inputs():
    with pytest.raises(ValueError, match=r".*p_values must be non-empty.*"):
        sgt_bonferroni_holm(np.array([]), delta=0.1)

    with pytest.raises(ValueError, match=r".*delta must be in \(0, 1].*"):
        sgt_bonferroni_holm(np.array([0.1, 0.2]), delta=0.0)

    with pytest.raises(ValueError, match=r".*delta must be in \(0, 1].*"):
        sgt_bonferroni_holm(np.array([0.1, 0.2]), delta=1.5)


def test_sgt_bonferroni_holm_no_rejection():
    p_values = np.array([0.5, 0.6, 0.7])
    delta = 0.1

    valid_index = sgt_bonferroni_holm(p_values, delta)

    assert len(valid_index) == 0


def test_sgt_bonferroni_holm_single_rejection():
    p_values = np.array([0.001, 0.4, 0.6])
    delta = 0.05

    valid_index = sgt_bonferroni_holm(p_values, delta)

    assert np.array_equal(valid_index, np.array([0]))


def test_sgt_bonferroni_holm_multiple_rejections():
    p_values = np.array([0.001, 0.01, 0.2])
    delta = 0.05

    valid_index = sgt_bonferroni_holm(p_values, delta)

    # Test behavior:
    # - first rejection at index 0
    # - redistribution allows rejection at index 1
    assert np.array_equal(valid_index, np.array([0, 1]))


def test_sgt_bonferroni_holm_all_rejected():
    p_values = np.array([0.001, 0.002, 0.003])
    delta = 0.05

    valid_index = sgt_bonferroni_holm(p_values, delta)

    assert np.array_equal(valid_index, np.array([0, 1, 2]))


def test_control_fwer_bonferroni():
    p_values = np.array([0.001, 0.02, 0.2, 0.8])
    delta = 0.05

    valid_index = control_fwer(
        p_values=p_values,
        delta=delta,
        fwer_method="bonferroni",
    )
    assert np.array_equal(valid_index, np.array([0]))


def test_control_fwer_fst():
    p_values = np.array([0.001, 0.003, 0.01, 0.02, 0.2])
    delta = 0.1

    valid_index = control_fwer(
        p_values,
        delta,
        fwer_method="fst_ascending",
        n_starts=3,
    )

    assert np.array_equal(valid_index, np.array([0, 1, 2, 3]))


def test_control_fwer_sgt():
    p_values = np.array([0.001, 0.01, 0.2])
    delta = 0.05

    valid_index = control_fwer(
        p_values,
        delta,
        fwer_method="sgt_bonferroni_holm",
    )

    assert np.array_equal(valid_index, np.array([0, 1]))


def test_control_fwer_invalid_inputs():
    with pytest.raises(ValueError, match=r".*p_values must be non-empty.*"):
        control_fwer(np.array([]), delta=0.1)

    with pytest.raises(ValueError, match=r".*delta must be in \(0, 1].*"):
        control_fwer(np.array([0.1, 0.2]), delta=0.0)

    with pytest.raises(ValueError, match=r".*delta must be in \(0, 1].*"):
        control_fwer(np.array([0.1, 0.2]), delta=1.5)


def test_control_fwer_unknown_method():
    p_values = np.array([0.001, 0.02])
    delta = 0.05

    with pytest.raises(ValueError, match=r".*Unknown FWER control method.*"):
        control_fwer(p_values, delta, fwer_method="invalid")
