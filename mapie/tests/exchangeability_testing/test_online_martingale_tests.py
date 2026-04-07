import numpy as np
import pytest

from mapie.exhangeability_testing.online_martingale_tests import OnlineMartingaleTest


def dummy_score(y_true, y_pred, X=None):
    return np.asarray(y_pred)


def test_init_validation_errors():
    with pytest.raises(ValueError, match=r"confidence_level must lie in \(0, 1\)"):
        OnlineMartingaleTest(dummy_score, confidence_level=1.0)

    with pytest.raises(ValueError, match=r"test_method must be one of"):
        OnlineMartingaleTest(dummy_score, test_method="invalid")

    with pytest.raises(ValueError, match=r"jump_size must lie in \[0, 1\]"):
        OnlineMartingaleTest(dummy_score, jump_size=-0.1)


def test_alpha_level_and_reject_threshold():
    omt = OnlineMartingaleTest(dummy_score, confidence_level=0.9)
    assert omt.alpha_level == pytest.approx(0.1)
    assert omt.reject_threshold == pytest.approx(10.0)


def test_to_1d_array_zero_dim_and_multi_dim():
    assert np.array_equal(
        OnlineMartingaleTest._to_1d_array(np.array(1.0)), np.array([1.0])
    )
    assert np.array_equal(
        OnlineMartingaleTest._to_1d_array(np.array([[1.0, 2.0], [3.0, 4.0]])),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )


def test_compute_p_value_without_history_is_reproducible():
    omt = OnlineMartingaleTest(dummy_score, random_state=1234)
    rng = np.random.default_rng(1234)
    rng.uniform()
    expected = float(rng.uniform())

    actual = omt.compute_p_value(
        current_non_conformity_score=0.5, non_conformity_score_history=np.asarray([])
    )

    assert actual == pytest.approx(expected)


def test_compute_p_value_with_history_and_ties():
    omt = OnlineMartingaleTest(dummy_score, random_state=1234)
    history = np.array([1.0, 2.0, 2.0, 3.0])

    rng = np.random.default_rng(1234)
    expected = float((1.0 + 1 + rng.uniform() * 2) / 5.0)

    actual = omt.compute_p_value(
        current_non_conformity_score=2.0, non_conformity_score_history=history
    )

    assert actual == pytest.approx(expected)
    assert 0.0 <= actual <= 1.0


def test_estimate_pvalues_density_returns_uniform_for_empty_history():
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    assert omt._estimate_pvalues_density(0.5) == 1.0


def test_estimate_pvalues_density_invalid_pvalue():
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    omt.pvalue_history = [0.1, 0.2]

    with pytest.raises(ValueError, match=r"pvalue must lie in \[0, 1\]"):
        omt._estimate_pvalues_density(-0.1)


def test_update_simple_jumper_martingale_history_and_return_value():
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    result = omt.update_simple_jumper_martingale(0.3)

    assert result == pytest.approx(omt.current_martingale_value)
    assert omt.martingale_value_history[-1] == pytest.approx(result)
    assert len(omt.martingale_value_history) == 1
    assert omt._jumper_wealth_by_expert.shape == (3,)


def test_update_plugin_martingale_uses_density_estimate():
    omt = OnlineMartingaleTest(
        dummy_score,
        test_method="plugin_martingale",
        min_sample_size_to_decide=1,
        random_state=0,
    )
    omt.pvalue_history = [0.1, 0.5, 0.9]
    omt.current_martingale_value = 1.0

    result = omt.update_plugin_martingale(0.2)

    assert result == pytest.approx(omt.current_martingale_value)
    assert len(omt.martingale_value_history) == 1
    assert result > 0.0


def test_update_warns_on_rejection():
    omt = OnlineMartingaleTest(
        dummy_score,
        test_method="plugin_martingale",
        confidence_level=0.5,
        min_sample_size_to_decide=1,
        warn=True,
        random_state=0,
    )
    omt.current_martingale_value = 100.0
    omt.pvalue_history = [0.1, 0.2, 0.3]
    omt.non_conformity_score_history = [0.1, 0.2, 0.3]

    with pytest.warns(
        UserWarning, match=r"The online martingale test has rejected exchangeability"
    ):
        omt.update(np.array([1.0]), np.array([1.0]))

    assert omt._warning_already_raised is True


def test_update_unsupported_method_raises():
    omt = OnlineMartingaleTest(dummy_score, random_state=0)
    omt.test_method = "unsupported"

    with pytest.raises(ValueError, match=r"Unsupported test method"):
        omt.update(np.array([1.0]), np.array([1.0]))


def test_update_appends_scores_and_pvalues():
    def score_function(_, y_pred, __):
        return np.asarray(y_pred)

    omt = OnlineMartingaleTest(
        score_function, min_sample_size_to_decide=1, random_state=1234
    )
    omt.update(np.array([1.0, 2.0]), np.array([0.1, 0.2]))

    assert len(omt.non_conformity_score_history) == 2
    assert len(omt.pvalue_history) == 2
    assert omt.current_martingale_value == pytest.approx(1.0)


def test_summary_without_values():
    omt = OnlineMartingaleTest(dummy_score)
    summary = omt.summary()

    assert summary["test_method"] == "jumper_martingale"
    assert summary["min_sample_size_to_decide"] == 100
    assert summary["test_level"] == pytest.approx(0.05)
    assert summary["is_exchangeable"] is None
    assert summary["stopping_time"] is None
    assert summary["martingale_value_at_decision"] is None
    assert summary["last_martingale_value"] == pytest.approx(1.0)
    assert summary["martingale_statistics"]["min"] is None


def test_summary_last_index_when_non_rejection():
    omt = OnlineMartingaleTest(
        dummy_score, confidence_level=0.95, min_sample_size_to_decide=1
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
    omt = OnlineMartingaleTest(
        dummy_score, confidence_level=0.9, min_sample_size_to_decide=1
    )
    omt.martingale_value_history = [1.0, 5.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    omt.current_martingale_value = 16.0
    omt.pvalue_history = [0.1] * len(omt.martingale_value_history)

    summary = omt.summary()

    assert summary["stopping_time"] == 3
    assert summary["martingale_value_at_decision"] == pytest.approx(11.0)
    assert summary["last_martingale_value"] == pytest.approx(16.0)
    assert summary["is_exchangeable"] is False
