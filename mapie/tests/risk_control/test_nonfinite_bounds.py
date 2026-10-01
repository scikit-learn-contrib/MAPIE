import warnings

import numpy as np

from mapie.risk_control.methods import find_best_predict_param


class TestNonFiniteBounds:
    def test_nan_bound_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.5]), np.array([np.nan]), np.array([0.1])
            )
        np.testing.assert_array_equal(result, [0.5])
        assert any("The risk cannot be controlled" in str(w.message) for w in caught)

    def test_negative_infinite_bound_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.5]), np.array([-np.inf]), np.array([0.1])
            )
        np.testing.assert_array_equal(result, [0.5])
        assert any("The risk cannot be controlled" in str(w.message) for w in caught)

    def test_positive_infinite_bound_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.5]), np.array([np.inf]), np.array([0.1])
            )
        np.testing.assert_array_equal(result, [0.5])
        assert any("The risk cannot be controlled" in str(w.message) for w in caught)

    def test_nan_interrupts_increasing_valid_prefix(self) -> None:
        bounds = np.array([0.05, np.nan, 0.08, 0.2])
        original = bounds.copy()
        result = find_best_predict_param(
            np.array([0.1, 0.3, 0.5, 0.7]), bounds, np.array([0.1])
        )
        np.testing.assert_array_equal(result, [0.1])
        np.testing.assert_array_equal(bounds, original)

    def test_nan_interrupts_decreasing_valid_prefix(self) -> None:
        result = find_best_predict_param(
            np.array([0.1, 0.3, 0.5, 0.7]),
            np.array([0.2, 0.08, np.nan, 0.05]),
            np.array([0.1]),
        )
        np.testing.assert_array_equal(result, [0.7])

    def test_warning_when_one_risk_level_cannot_be_controlled(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.1, 0.3, 0.5]),
                np.array([0.05, np.nan, 0.2]),
                np.array([0.01, 0.1]),
            )
        np.testing.assert_array_equal(result, [0.1, 0.1])
        assert any("The risk cannot be controlled" in str(w.message) for w in caught)

    def test_finite_increasing_bounds_unchanged(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.1, 0.3, 0.5]),
                np.array([0.05, 0.1, 0.2]),
                np.array([0.15]),
            )
        np.testing.assert_array_equal(result, [0.3])
        assert not caught

    def test_finite_bounds_warn_when_one_risk_level_has_no_valid_parameter(
        self,
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.1, 0.2]),
                np.array([0.5, 0.6]),
                np.array([0.1, 0.9]),
            )
        np.testing.assert_array_equal(result, [0.1, 0.2])
        assert any("The risk cannot be controlled" in str(w.message) for w in caught)

    def test_leading_nan_preserves_endpoint_direction_inference(self) -> None:
        # Characterize the unchanged direction inference, not a risk guarantee.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.1, 0.2, 0.3]),
                np.array([np.nan, 0.05, 0.08]),
                np.array([0.1]),
            )
        np.testing.assert_array_equal(result, [0.2])
        assert not caught

    def test_trailing_nan_preserves_extreme_fallback(self) -> None:
        # An invalid endpoint can still be returned by the existing fallback.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.1, 0.2, 0.3]),
                np.array([0.08, 0.05, np.nan]),
                np.array([0.1]),
            )
        np.testing.assert_array_equal(result, [0.3])
        assert not caught

    def test_finite_decreasing_bounds_unchanged(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = find_best_predict_param(
                np.array([0.1, 0.3, 0.5]),
                np.array([0.2, 0.1, 0.05]),
                np.array([0.15]),
            )
        np.testing.assert_array_equal(result, [0.3])
        assert not caught
