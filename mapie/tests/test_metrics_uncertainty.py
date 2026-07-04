"""
Tests for mapie/metrics/uncertainty.py

All expected values are computed by hand or from first principles
so that tests are independent of the implementation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mapie.metrics.uncertainty import aucroc_score, auarc_score


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Perfect case: uncertainty perfectly ranks wrong predictions above correct ones
Y_WRONG_PERFECT = np.array([0, 0, 1, 1], dtype=float)
Y_UNC_PERFECT = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)

# Random case: uncertainty uncorrelated with errors
Y_WRONG_RANDOM = np.array([0, 1, 0, 1], dtype=float)
Y_UNC_RANDOM = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

# All correct predictions
Y_WRONG_ALL_CORRECT = np.zeros(4, dtype=float)
Y_UNC_UNIFORM = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)


# ---------------------------------------------------------------------------
# aucroc_score — known-answer tests
# ---------------------------------------------------------------------------


class TestAucrocScore:
    def test_perfect_discriminator(self):
        """Uncertainty perfectly separates wrong from correct → AUCROC = 1."""
        result = aucroc_score(Y_WRONG_PERFECT, Y_UNC_PERFECT)
        assert_allclose(result, 1.0)

    def test_returns_float(self):
        result = aucroc_score(Y_WRONG_PERFECT, Y_UNC_PERFECT)
        assert isinstance(result, float)

    def test_score_in_unit_interval(self):
        result = aucroc_score(Y_WRONG_RANDOM, Y_UNC_RANDOM)
        assert 0.0 <= result <= 1.0

    def test_accepts_list_input(self):
        """Should accept plain Python lists, not just numpy arrays."""
        result = aucroc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert_allclose(result, 1.0)

    def test_known_value_random_case(self):
        """
        y_wrong = [0, 1, 0, 1], y_unc = [0.1, 0.2, 0.3, 0.4]
        sklearn roc_auc_score ground truth.
        Wrong indices: 1, 3 (uncertainty 0.2, 0.4)
        Correct indices: 0, 2 (uncertainty 0.1, 0.3)
        By hand: TPR/FPR pairs give AUC = 0.75
        """
        result = aucroc_score(Y_WRONG_RANDOM, Y_UNC_RANDOM)
        assert_allclose(result, 0.75)

    # --- input validation ---

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            aucroc_score([0, 1], [0.1, 0.2, 0.3])

    def test_raises_on_nan_in_y_wrong(self):
        with pytest.raises(ValueError, match="NaN"):
            aucroc_score([0, np.nan, 1, 1], Y_UNC_PERFECT)

    def test_raises_on_nan_in_y_uncertainty(self):
        with pytest.raises(ValueError, match="NaN"):
            aucroc_score(Y_WRONG_PERFECT, [0.1, np.nan, 0.8, 0.9])

    def test_raises_on_inf_in_y_wrong(self):
        with pytest.raises(ValueError, match="Inf"):
            aucroc_score([0, np.inf, 1, 1], Y_UNC_PERFECT)

    def test_raises_on_inf_in_y_uncertainty(self):
        with pytest.raises(ValueError, match="Inf"):
            aucroc_score(Y_WRONG_PERFECT, [0.1, np.inf, 0.8, 0.9])

    def test_raises_on_non_binary_y_wrong(self):
        with pytest.raises(ValueError, match="binary"):
            aucroc_score([0, 0.5, 1, 1], Y_UNC_PERFECT)

    def test_raises_on_negative_uncertainty(self):
        with pytest.raises(ValueError, match="non-negative"):
            aucroc_score(Y_WRONG_PERFECT, [-0.1, 0.2, 0.8, 0.9])


# ---------------------------------------------------------------------------
# auarc_score — known-answer tests
# ---------------------------------------------------------------------------


class TestAuarcScore:
    def test_perfect_rejection(self):
        """
        Wrong samples have highest uncertainty, so rejecting by uncertainty
        removes all errors first → accuracy reaches 1.0 quickly → AUARC = 1.0.

        y_wrong      = [1, 1, 0, 0]
        y_uncertainty= [0.9, 0.8, 0.2, 0.1]
        rejection order (desc unc): indices [0,1,2,3]
        y_wrong_sorted = [1,1,0,0]

        accuracy_curve (retaining n, n-1, ..., 1 samples):
          retain 4: (0+0+0+0... wait — correct = 1-wrong = [0,0,1,1])
          cumsum from tail of [0,0,1,1] = [2,2,2,1]  NO let me redo:
          y_correct_sorted = [0,0,1,1]
          cumsum from tail: cumsum([1,1,0,0]) reversed → [2,2,1,1] reversed
          Actually cumsum([0,0,1,1] reversed=[1,1,0,0]) = [1,2,2,2], reversed=[2,2,2,1]
          retained_counts = [4,3,2,1]
          accuracy_curve = [2/4, 2/3, 2/2, 1/1] = [0.5, 0.667, 1.0, 1.0]
          mean = (0.5 + 0.667 + 1.0 + 1.0) / 4 = 0.792
        """
        y_wrong = np.array([1, 1, 0, 0], dtype=float)
        y_unc = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
        result = auarc_score(y_wrong, y_unc)
        expected = np.mean([0.5, 2 / 3, 1.0, 1.0])
        assert_allclose(result, expected, rtol=1e-6)

    def test_all_correct_predictions(self):
        """
        If no predictions are wrong, accuracy is always 1.0 regardless
        of rejection order → AUARC = 1.0.
        """
        y_wrong = np.zeros(5, dtype=float)
        y_unc = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=float)
        result = auarc_score(y_wrong, y_unc)
        assert_allclose(result, 1.0)

    def test_returns_float(self):
        result = auarc_score(Y_WRONG_PERFECT, Y_UNC_PERFECT)
        assert isinstance(result, float)

    def test_score_in_unit_interval(self):
        result = auarc_score(Y_WRONG_RANDOM, Y_UNC_RANDOM)
        assert 0.0 <= result <= 1.0

    def test_accepts_list_input(self):
        result = auarc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert isinstance(result, float)

    def test_worst_rejection(self):
        """
        Correct samples have highest uncertainty → we reject correct ones first
        → accuracy degrades. AUARC should be lower than the perfect case.
        """
        y_wrong = np.array([0, 0, 1, 1], dtype=float)
        y_unc_bad = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
        y_unc_good = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
        assert auarc_score(y_wrong, y_unc_bad) < auarc_score(y_wrong, y_unc_good)

    # --- input validation ---

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError):
            auarc_score([0, 1], [0.1, 0.2, 0.3])

    def test_raises_on_nan_in_y_wrong(self):
        with pytest.raises(ValueError, match="NaN"):
            auarc_score([0, np.nan, 1, 1], Y_UNC_PERFECT)

    def test_raises_on_nan_in_y_uncertainty(self):
        with pytest.raises(ValueError, match="NaN"):
            auarc_score(Y_WRONG_PERFECT, [0.1, np.nan, 0.8, 0.9])

    def test_raises_on_inf_in_y_wrong(self):
        with pytest.raises(ValueError, match="Inf"):
            auarc_score([0, np.inf, 1, 1], Y_UNC_PERFECT)

    def test_raises_on_inf_in_y_uncertainty(self):
        with pytest.raises(ValueError, match="Inf"):
            auarc_score(Y_WRONG_PERFECT, [0.1, np.inf, 0.8, 0.9])

    def test_raises_on_non_binary_y_wrong(self):
        with pytest.raises(ValueError, match="binary"):
            auarc_score([0, 0.5, 1, 1], Y_UNC_PERFECT)

    def test_raises_on_negative_uncertainty(self):
        with pytest.raises(ValueError, match="non-negative"):
            auarc_score(Y_WRONG_PERFECT, [-0.1, 0.2, 0.8, 0.9])
