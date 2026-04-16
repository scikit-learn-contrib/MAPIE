import warnings

import numpy as np
import pytest

import mapie.exhangeability_testing.online as omt_module
from mapie.exhangeability_testing.online import OnlineMartingaleTest


def dummy_score(y_true, y_pred, X=None):
    return np.asarray(y_pred)


def test_init_validation_errors():
    """Test that invalid initialization parameters raise ValueError."""
    with pytest.raises(ValueError, match=r"test_level must lie in \(0, 1\)"):
        OnlineMartingaleTest(dummy_score, test_level=1.0)

    with pytest.raises(ValueError, match=r".*test_method must be one of.*"):
        OnlineMartingaleTest(dummy_score, test_method="invalid")

    with pytest.raises(ValueError, match=r"jump_size must lie in \(0, 1\)"):
        OnlineMartingaleTest(dummy_score, jump_size=-0.1)


def test_reject_threshold_computation():
    """Test that reject_threshold is computed correctly from test_level."""
    omt = OnlineMartingaleTest(dummy_score, test_level=0.1)
    assert omt.reject_threshold == pytest.approx(10.0)

    omt2 = OnlineMartingaleTest(dummy_score, test_level=0.05)
    assert omt2.reject_threshold == pytest.approx(20.0)


def test_to_1d_array_zero_dim_and_multi_dim():
    """Test flattening of zero-dimensional and multi-dimensional arrays."""
    assert np.array_equal(
        OnlineMartingaleTest._to_1d_array(np.array(1.0)), np.array([1.0])
    )
    assert np.array_equal(
        OnlineMartingaleTest._to_1d_array(np.array([[1.0, 2.0], [3.0, 4.0]])),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )


def test_compute_p_value_without_history_is_reproducible():
    """Test that p-value computation without history is reproducible."""
    omt = OnlineMartingaleTest(dummy_score, random_state=1234)
    rng = np.random.default_rng(1234)
    rng.uniform()
    expected = float(rng.uniform())

    actual = omt.compute_p_value(
        current_conformity_score=0.5, conformity_score_history=np.asarray([])
    )

    assert actual == pytest.approx(expected)


def test_compute_p_value_with_history_and_ties():
    """Test p-value computation with history and tied scores."""
    omt = OnlineMartingaleTest(dummy_score, random_state=1234)
    history = np.array([1.0, 2.0, 2.0, 3.0])

    rng = np.random.default_rng(1234)
    expected = float((1.0 + 1 + rng.uniform() * 2) / 5.0)

    actual = omt.compute_p_value(
        current_conformity_score=2.0, conformity_score_history=history
    )

    assert actual == pytest.approx(expected)
    assert 0.0 <= actual <= 1.0


def test_is_exchangeable_returns_false_for_one_value_above_threshold():
    """Test is_exchangeable returns False when at least one value is above threshold."""
    omt = OnlineMartingaleTest(dummy_score, burn_in=1)
    omt.pvalue_history = [0.1]  # 1 pvalue
    omt.martingale_value_history = [21.0]  # One value above threshold (20.0)
    omt.current_martingale_value = 1.0  # Current value below threshold

    assert omt.is_exchangeable is False


def test_estimate_pvalues_density_returns_uniform_for_empty_history():
    """Test that density estimate returns uniform for empty p-value history."""
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    assert omt._estimate_pvalues_density(0.5) == 1.0


def test_estimate_pvalues_density_invalid_pvalue():
    """Test that invalid p-value raises ValueError."""
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    omt.pvalue_history = [0.1, 0.2]

    with pytest.raises(ValueError, match=r"pvalue must lie in \[0, 1\]"):
        omt._estimate_pvalues_density(-0.1)


def test_estimate_pvalues_density_returns_uniform_when_normalization_invalid(
    monkeypatch,
):
    """Test that density estimate returns uniform when normalization is invalid."""
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    omt.pvalue_history = [0.1, 0.2]

    class DummyKDE:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return np.zeros_like(np.asarray(x), dtype=float)

    monkeypatch.setattr(omt_module, "gaussian_kde", DummyKDE)

    assert omt._estimate_pvalues_density(0.5) == 1.0


def test_estimate_pvalues_density_returns_uniform_when_density_nonfinite(monkeypatch):
    """Test that density estimate returns uniform when density is non-finite."""
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    omt.pvalue_history = [0.1, 0.2]

    class DummyKDE:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            x = np.asarray(x)
            if x.size == 1:
                return np.array([np.inf])
            return np.ones_like(x, dtype=float)

    monkeypatch.setattr(omt_module, "gaussian_kde", DummyKDE)

    assert omt._estimate_pvalues_density(0.5) == 1.0


def test_update_simple_jumper_martingale_history_and_return_value():
    """Test that jumper martingale update returns correct value and updates history."""
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    result = omt.update_simple_jumper_martingale(0.3)

    assert result == pytest.approx(omt.current_martingale_value)
    assert omt.martingale_value_history[-1] == pytest.approx(result)
    assert len(omt.martingale_value_history) == 1
    assert omt._jumper_wealth_by_expert.shape == (3,)


def test_update_simple_jumper_martingale_invalid_pvalue():
    """Test that invalid p-value raises ValueError in jumper martingale update."""
    omt = OnlineMartingaleTest(dummy_score, random_state=0)

    with pytest.raises(ValueError, match=r"pvalue must lie in \[0, 1\]"):
        omt.update_simple_jumper_martingale(1.5)


def test_update_plugin_martingale_uses_density_estimate():
    """Test that plugin martingale update uses density estimate."""
    omt = OnlineMartingaleTest(
        dummy_score,
        test_method="plugin_martingale",
        burn_in=1,
        random_state=0,
    )
    omt.pvalue_history = [0.1, 0.5, 0.9]
    omt.current_martingale_value = 1.0

    result = omt.update_plugin_martingale(0.2)

    assert result == pytest.approx(omt.current_martingale_value)
    assert len(omt.martingale_value_history) == 1
    assert result > 0.0


def test_update_warns_on_rejection():
    """Test that update raises warning when exchangeability is rejected."""
    omt = OnlineMartingaleTest(
        dummy_score,
        test_method="plugin_martingale",
        test_level=0.5,
        burn_in=1,
        warn=True,
        random_state=0,
    )
    omt.current_martingale_value = 100.0
    omt.pvalue_history = [0.1, 0.2, 0.3]
    omt.conformity_score_history = [0.1, 0.2, 0.3]

    with pytest.warns(
        UserWarning,
        match=r".*The online martingale test has rejected exchangeability.*",
    ):
        omt.update(np.array([1.0]), np.array([1.0]))

    assert omt._warning_already_raised is True


def test_update_unsupported_method_raises():
    """Test that update raises ValueError for unsupported test method."""
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    omt.test_method = "best_martingale"

    with pytest.raises(ValueError, match=r".*Unsupported test method.*"):
        omt.update(np.array([1.0]), np.array([1.0]))


def test_update_appends_scores_and_pvalues():
    """Test that update appends scores and p-values to history."""

    def score_function(_, y_pred, __):
        return np.asarray(y_pred)

    omt = OnlineMartingaleTest(
        score_function, burn_in=1, random_state=1234
    )
    omt.update(np.array([1.0, 2.0]), np.array([0.1, 0.2]))

    assert len(omt.conformity_score_history) == 2
    assert len(omt.pvalue_history) == 2
    assert omt.current_martingale_value == pytest.approx(1.0)


def test_summary_without_values():
    """Test that summary returns None values when no martingale values exist."""
    omt = OnlineMartingaleTest(dummy_score)
    summary = omt.summary()

    assert summary["test_method"] == "jumper_martingale"
    assert summary["burn_in"] == 100
    assert summary["test_level"] == pytest.approx(0.05)
    assert summary["is_exchangeable"] is None
    assert summary["stopping_time"] is None
    assert summary["martingale_value_at_decision"] is None
    assert summary["last_martingale_value"] == pytest.approx(1.0)
    assert summary["martingale_statistics"]["min"] is None


def test_summary_last_index_when_non_rejection():
    """Test that summary stopping_time is last index when no threshold crossing."""
    omt = OnlineMartingaleTest(
        dummy_score, test_level=0.05, burn_in=1
    )
    omt.martingale_value_history = [1.0, 2.0, 3.0]
    omt.current_martingale_value = 3.0
    omt.pvalue_history = [0.1, 0.2, 0.3]

    summary = omt.summary()

    assert summary["stopping_time"] == 3
    assert summary["martingale_value_at_decision"] == pytest.approx(3.0)
    assert summary["last_martingale_value"] == pytest.approx(3.0)
    assert summary["is_exchangeable"] is True


def test_summary_rejection_requires_sustained_block():
    """Test that summary detects first threshold crossing for both rejection cases."""
    omt = OnlineMartingaleTest(dummy_score, test_level=0.1, burn_in=1)
    omt.martingale_value_history = [1.0, 5.0, 11.0, 9.5, 9.0, 8.0, 7.0]
    omt.current_martingale_value = 7.0
    omt.pvalue_history = [0.1] * len(omt.martingale_value_history)

    summary = omt.summary()

    assert summary["stopping_time"] == 3
    assert summary["martingale_value_at_decision"] == pytest.approx(11.0)
    assert summary["last_martingale_value"] == pytest.approx(7.0)
    assert summary["is_exchangeable"] is False


def test_summary_non_rejection_no_threshold_crossing():
    """Test that summary with no threshold crossing indicates non-rejection."""
    omt = OnlineMartingaleTest(dummy_score, test_level=0.1, burn_in=1)
    omt.martingale_value_history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    omt.current_martingale_value = 9.0
    omt.pvalue_history = [0.1] * len(omt.martingale_value_history)

    summary = omt.summary()

    assert summary["stopping_time"] == 9
    assert summary["martingale_value_at_decision"] == pytest.approx(9.0)
    assert summary["last_martingale_value"] == pytest.approx(9.0)
    assert summary["is_exchangeable"] is True


def test_summary_detects_sustained_rejection_start():
    """Test that summary detects sustained rejection when all values exceed threshold."""
    omt = OnlineMartingaleTest(dummy_score, test_level=0.1, burn_in=1)
    omt.martingale_value_history = [11.0] * 6
    omt.current_martingale_value = 11.0
    omt.pvalue_history = [0.1] * 6

    summary = omt.summary()

    assert summary["stopping_time"] == 1
    assert summary["martingale_value_at_decision"] == pytest.approx(11.0)
    assert summary["is_exchangeable"] is False


def test_update_without_warn():
    """Test that no warning is raised when warn=False."""
    omt = OnlineMartingaleTest(
        dummy_score,
        test_method="plugin_martingale",
        test_level=0.5,
        burn_in=1,
        warn=False,
        random_state=0,
    )
    omt.current_martingale_value = 100.0
    omt.pvalue_history = [0.1, 0.2, 0.3]
    omt.conformity_score_history = [0.1, 0.2, 0.3]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        omt.update(np.array([1.0]), np.array([1.0]))
        user_warnings = [
            warning for warning in w if issubclass(warning.category, UserWarning)
        ]
        assert len(user_warnings) == 0


def test_is_exchangeable_with_exactly_one_value_above_threshold():
    """Test that exactly one value above threshold returns False."""
    omt = OnlineMartingaleTest(
        dummy_score, test_level=0.05, burn_in=1
    )
    omt.pvalue_history = [0.1]  # Above threshold (20.0)
    omt.martingale_value_history = [21.0]  # One value above threshold
    omt.current_martingale_value = 1.0  # Current value below threshold

    assert omt.is_exchangeable is False


def test_is_exchangeable_with_two_values_above_threshold():
    """Test that two or more values above threshold returns False."""
    omt = OnlineMartingaleTest(
        dummy_score, test_level=0.05, burn_in=1
    )
    omt.pvalue_history = [0.1, 0.2]  # At least 2 pvalues
    omt.martingale_value_history = [21.0, 22.0]  # Two values above threshold (20.0)
    omt.current_martingale_value = 1.0  # Current value below threshold doesn't matter

    assert omt.is_exchangeable is False


def test_summary_martingale_statistics_completeness():
    """Test that all martingale statistics are returned with values."""
    omt = OnlineMartingaleTest(
        dummy_score, test_level=0.05, burn_in=1
    )
    omt.martingale_value_history = [1.0, 2.0, 3.0, 4.0, 5.0]
    omt.current_martingale_value = 5.0
    omt.pvalue_history = [0.1, 0.2, 0.3, 0.4, 0.5]

    summary = omt.summary()

    stats = summary["martingale_statistics"]
    assert stats["min"] is not None
    assert stats["q025"] is not None
    assert stats["q25"] is not None
    assert stats["median"] is not None
    assert stats["mean"] is not None
    assert stats["q75"] is not None
    assert stats["q975"] is not None
    assert stats["max"] is not None
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_compute_p_value_with_strict_greater_than():
    """Test p-value computation with specific values to verify formula."""
    omt = OnlineMartingaleTest(dummy_score, random_state=42)
    history = np.array([0.5, 1.0, 1.5, 2.0])

    pvalue = omt.compute_p_value(
        current_conformity_score=1.2, conformity_score_history=history
    )

    assert 0.0 <= pvalue <= 1.0
    assert pvalue != 0.0
    assert pvalue != 1.0
