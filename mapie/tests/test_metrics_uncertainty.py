"""
Tests for mapie/metrics/uncertainty.py

All expected values are computed by hand or from first principles
so that tests are independent of the implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mapie.metrics.uncertainty import auroc, auarc


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Perfect case: confidence perfectly ranks correct predictions above incorrect
Y_CORRECT_PERFECT = np.array([1, 1, 0, 0], dtype=float)
Y_CONF_PERFECT = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)

# Random case: confidence uncorrelated with correctness
Y_CORRECT_RANDOM = np.array([1, 0, 1, 0], dtype=float)
Y_CONF_RANDOM = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

# All correct predictions
Y_CORRECT_ALL_CORRECT = np.ones(4, dtype=float)
Y_CONF_UNIFORM = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)


# ---------------------------------------------------------------------------
# auroc — known-answer tests
# ---------------------------------------------------------------------------


class TestAuroc:
    def test_perfect_discriminator(self):
        """Confidence perfectly separates correct from incorrect → AUROC = 1."""
        result = auroc(Y_CORRECT_PERFECT, Y_CONF_PERFECT)
        assert_allclose(result, 1.0)

    def test_returns_float(self):
        result = auroc(Y_CORRECT_PERFECT, Y_CONF_PERFECT)
        assert isinstance(result, float)

    def test_score_in_unit_interval(self):
        result = auroc(Y_CORRECT_RANDOM, Y_CONF_RANDOM)
        assert 0.0 <= result <= 1.0

    def test_accepts_list_input(self):
        """Should accept plain Python lists, not just numpy arrays."""
        result = auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        assert_allclose(result, 1.0)

    def test_known_value_random_case(self):
        """
        correctness = [1, 0, 1, 0], confidence = [0.1, 0.2, 0.3, 0.4]
        sklearn roc_auc_score ground truth.
        """
        result = auroc(Y_CORRECT_RANDOM, Y_CONF_RANDOM)
        expected = np.float64(0.25)
        assert_allclose(result, expected)

    def test_all_correct_predictions(self):
        """
        When all predictions are correct, AUROC is not defined
        (single class) and should raise ValueError.
        """
        with pytest.raises(ValueError, match="both 0 and 1"):
            auroc(Y_CORRECT_ALL_CORRECT, Y_CONF_UNIFORM)

    def test_all_incorrect_predictions(self):
        """All-incorrect also raises for single-class AUROC."""
        with pytest.raises(ValueError, match="both 0 and 1"):
            auroc(np.zeros(4, dtype=float), np.array([0.1, 0.2, 0.3, 0.4]))

    def test_raises_on_non_binary_labels(self):
        with pytest.raises(ValueError, match="binary"):
            auroc([0, 1, 2], [0.1, 0.2, 0.3])

    def test_raises_on_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            auroc([0, np.nan, 1], [0.1, 0.2, 0.3])

    def test_raises_on_inf(self):
        with pytest.raises(ValueError, match="Inf"):
            auroc([0, 1, 1], [0.1, np.inf, 0.3])

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            auroc([0, 1], [0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# auarc — known-answer tests
# ---------------------------------------------------------------------------


class TestAuarc:
    def test_perfect_confidence(self):
        """
        confidence perfectly ranks correct above incorrect →
        AUARC > overall accuracy.
        """
        result = auarc(Y_CORRECT_PERFECT, Y_CONF_PERFECT)
        # correctness=[1,1,0,0], confidence=[0.9,0.8,0.2,0.1]
        # sorted descending: k=1→1.0, k=2→1.0, k=3→0.667, k=4→0.5
        expected = np.mean([1.0, 1.0, 2 / 3, 0.5])
        assert_allclose(result, expected, rtol=1e-6)

    def test_all_correct_predictions(self):
        """
        If all predictions are correct, accuracy is always 1.0
        regardless of rejection order → AUARC = 1.0.
        """
        result = auarc(
            np.ones(5, dtype=float),
            np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=float),
        )
        assert_allclose(result, 1.0)

    def test_returns_float(self):
        result = auarc(Y_CORRECT_PERFECT, Y_CONF_PERFECT)
        assert isinstance(result, float)

    def test_score_in_unit_interval(self):
        result = auarc(Y_CORRECT_RANDOM, Y_CONF_RANDOM)
        assert 0.0 <= result <= 1.0

    def test_accepts_list_input(self):
        result = auarc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        assert isinstance(result, float)

    def test_worst_confidence(self):
        """
        Correct samples have lowest confidence → we retain least
        confident first → accuracy degrades. AUARC should be lower
        than the perfect case.
        """
        correctness = np.array([0, 0, 1, 1], dtype=float)
        conf_bad = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
        conf_good = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        assert auarc(correctness, conf_bad) < auarc(correctness, conf_good)

    def test_raises_on_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            auarc([0, np.nan, 1], [0.1, 0.2, 0.3])

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            auarc([0, 1], [0.1, 0.2, 0.3])

    def test_raises_on_inf(self):
        with pytest.raises(ValueError, match="Inf"):
            auarc([0, 1, 1], [0.1, np.inf, 0.3])

    def test_raises_on_non_binary_labels(self):
        with pytest.raises(ValueError, match="binary"):
            auarc([0, 1, 2], [0.1, 0.2, 0.3])
