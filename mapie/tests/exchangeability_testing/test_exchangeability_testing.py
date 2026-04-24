import math

import numpy as np
import pytest
from numpy.typing import NDArray

from mapie.exchangeability_testing import RiskMonitoring
from mapie.exchangeability_testing.confidence_bounds import (
    GammaExponentialMixtureBound,
    conjugate_mixture_empirical_bernstein_bound,
    hoeffding_bound,
)
from mapie.risk_control.risks import BinaryRiskNames


class TestGammaExponentialMixtureBound:
    def test_init_validates_parameters(self) -> None:
        with pytest.raises(ValueError, match="v_opt must be > 0"):
            GammaExponentialMixtureBound(v_opt=0, c=1)

        with pytest.raises(ValueError, match="c must be > 0"):
            GammaExponentialMixtureBound(v_opt=1, c=0)

        with pytest.raises(ValueError, match="alpha_opt must be in"):
            GammaExponentialMixtureBound(v_opt=1, c=1, alpha_opt=0.5)

    def test_two_sided_best_rho_validates_inputs(self) -> None:
        with pytest.raises(ValueError, match="v must be > 0"):
            GammaExponentialMixtureBound._two_sided_best_rho(v=0, alpha=0.1)

        with pytest.raises(ValueError, match="alpha must be in"):
            GammaExponentialMixtureBound._two_sided_best_rho(v=1, alpha=1.0)

    def test_log_lower_regularized_gamma_returns_minus_inf_at_zero(self) -> None:
        result = GammaExponentialMixtureBound._log_lower_regularized_gamma(a=1.0, x=0.0)
        assert result == float("-inf")

    def test_log_supermg_and_bound_and_call(self) -> None:
        bound = GammaExponentialMixtureBound(v_opt=1.0, c=1.0, alpha_opt=0.05)

        log_value = bound.log_supermg(s=0.5, v=0.2)
        assert isinstance(log_value, float)
        assert math.isfinite(log_value)

        direct_bound = bound.bound(v=0.2, alpha=0.1)
        call_bound = bound(v=0.2, alpha=0.1)
        assert direct_bound > 0
        assert call_bound == pytest.approx(direct_bound)

    def test_bound_can_stop_on_max_iter_without_convergence_break(self) -> None:
        bound = GammaExponentialMixtureBound(v_opt=1.0, c=1.0, alpha_opt=0.05)

        result = bound.bound(v=0.2, alpha=0.1, tol=0.0, max_iter=1)

        assert result > 0

    def test_log_supermg_and_bound_validate_inputs(self) -> None:
        bound = GammaExponentialMixtureBound(v_opt=1.0, c=1.0, alpha_opt=0.05)

        with pytest.raises(ValueError, match="v must be >= 0"):
            bound.log_supermg(s=0.5, v=-1.0)

        with pytest.raises(ValueError, match="v must be >= 0"):
            bound.bound(v=-1.0, alpha=0.1)

        with pytest.raises(ValueError, match="alpha must be in"):
            bound.bound(v=0.5, alpha=0.0)

    def test_bound_raises_when_no_upper_limit_is_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bound = GammaExponentialMixtureBound(v_opt=1.0, c=1.0, alpha_opt=0.05)
        monkeypatch.setattr(bound, "log_supermg", lambda s, v: -1e9)

        with pytest.raises(RuntimeError, match="Failed to find an upper limit"):
            bound.bound(v=0.5, alpha=0.1)


class TestHoeffdingBound:
    def test_returns_lower_and_upper_bounds(self) -> None:
        empirical_risk_sequence = np.array([0.0, 1.0, 0.0, 1.0])

        lower = hoeffding_bound(empirical_risk_sequence, delta=0.1, bound_side="lower")
        upper = hoeffding_bound(empirical_risk_sequence, delta=0.1, bound_side="upper")

        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_rejects_invalid_bound_side(self) -> None:
        with pytest.raises(
            ValueError, match="bound_side must be either 'upper' or 'lower'"
        ):
            hoeffding_bound(np.array([0.0, 1.0]), delta=0.1, bound_side="invalid")  # type: ignore[arg-type]

    def test_rejects_empty_sequence(self) -> None:
        with pytest.raises(
            ValueError, match="empirical_risk_sequence must contain at least one value"
        ):
            hoeffding_bound(np.array([]), delta=0.1)

    def test_rejects_invalid_delta(self) -> None:
        with pytest.raises(ValueError, match="delta must be in"):
            hoeffding_bound(np.array([0.0, 1.0]), delta=1.0)


class TestConjugateMixtureEmpiricalBernsteinBound:
    def test_returns_lower_bound_with_running_intersection(self) -> None:
        empirical_risk_sequence = np.array([1.0, 0.0, 1.0, 1.0, 0.0])

        bound = conjugate_mixture_empirical_bernstein_bound(
            empirical_risk_sequence,
            v_opt=1.0,
            alpha=0.1,
            bound_side="lower",
            running_intersection=True,
        )

        assert bound.shape == empirical_risk_sequence.shape
        assert np.all(bound >= 0)
        assert np.all(np.diff(bound) >= -1e-12)

    def test_returns_lower_bound_without_running_intersection(self) -> None:
        empirical_risk_sequence = np.array([1.0, 0.0, 1.0, 1.0, 0.0])

        bound = conjugate_mixture_empirical_bernstein_bound(
            empirical_risk_sequence,
            v_opt=1.0,
            alpha=0.1,
            bound_side="lower",
            running_intersection=False,
        )

        assert bound.shape == empirical_risk_sequence.shape
        assert np.all(bound >= 0)

    def test_returns_upper_bound_without_running_intersection(self) -> None:
        empirical_risk_sequence = np.array([0.0, 1.0, 0.0, 1.0, 1.0])

        bound = conjugate_mixture_empirical_bernstein_bound(
            empirical_risk_sequence,
            v_opt=1.0,
            alpha=0.1,
            bound_side="upper",
            running_intersection=False,
        )

        assert bound.shape == empirical_risk_sequence.shape
        assert np.all(bound <= 1)

    def test_returns_upper_bound_with_running_intersection(self) -> None:
        empirical_risk_sequence = np.array([0.0, 1.0, 0.0, 1.0, 1.0])

        bound = conjugate_mixture_empirical_bernstein_bound(
            empirical_risk_sequence,
            v_opt=1.0,
            alpha=0.1,
            bound_side="upper",
            running_intersection=True,
        )

        assert bound.shape == empirical_risk_sequence.shape
        assert np.all(bound <= 1)
        assert np.all(np.diff(bound) <= 1e-12)

    def test_rejects_invalid_bound_side(self) -> None:
        with pytest.raises(
            ValueError, match="bound_side must be either 'upper' or 'lower'"
        ):
            conjugate_mixture_empirical_bernstein_bound(
                np.array([0.0, 1.0]),
                v_opt=1.0,
                alpha=0.1,
                bound_side="invalid",  # type: ignore[arg-type]
            )

    def test_rejects_empty_sequence(self) -> None:
        with pytest.raises(
            ValueError, match="empirical_risk_sequence must contain at least one value"
        ):
            conjugate_mixture_empirical_bernstein_bound(
                np.array([]),
                v_opt=1.0,
                alpha=0.1,
            )


class TestRiskMonitoring:
    @staticmethod
    def _binary_data() -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        return y_true, y_pred

    def test_is_reexported_from_package_init(self) -> None:
        monitor = RiskMonitoring(risk="accuracy")
        assert isinstance(monitor, RiskMonitoring)

    def test_init_rejects_invalid_string_risk(self) -> None:
        with pytest.raises(ValueError, match="When risk is provided as a string"):
            RiskMonitoring(risk="unknown_risk")  # type: ignore[arg-type]

    def test_init_rejects_non_scalar_risk(self) -> None:
        invalid_risk: list[BinaryRiskNames] = ["accuracy"]
        with pytest.raises(TypeError, match="risk must be a single BinaryRisk"):
            RiskMonitoring(risk=invalid_risk)

    def test_init_rejects_invalid_test_level(self) -> None:
        with pytest.raises(ValueError, match="test_level must be in"):
            RiskMonitoring(risk="accuracy", test_level=1.0)
        with pytest.raises(ValueError, match="test_level must be in"):
            RiskMonitoring(risk="accuracy", test_level=0.0)
        with pytest.raises(ValueError, match="test_level must be in"):
            RiskMonitoring(risk="accuracy", test_level=-0.1)
        with pytest.raises(ValueError, match="test_level must be in"):
            RiskMonitoring(risk="accuracy", test_level=1.1)

    def test_harmful_shift_detected_requires_online_bound(self) -> None:
        monitor = RiskMonitoring(risk="accuracy")

        with pytest.raises(
            ValueError, match="Online risk lower bound must be computed"
        ):
            _ = monitor.harmful_shift_detected

    def test_harmful_shift_detected_requires_threshold(self) -> None:
        monitor = RiskMonitoring(risk="accuracy")
        monitor.online_risk_lower_bound_latest = 0.1

        with pytest.raises(ValueError, match="Threshold must be computed"):
            _ = monitor.harmful_shift_detected

    def test_compute_threshold_absolute_tolerance(self) -> None:
        y_true, y_pred = self._binary_data()
        monitor = RiskMonitoring(
            risk="accuracy", tolerance=0.2, tolerance_type="absolute"
        )

        returned = monitor.compute_threshold(y_true, y_pred)

        assert returned is monitor
        assert monitor.reference_risk_upper_bound is not None
        assert monitor.threshold == pytest.approx(
            monitor.reference_risk_upper_bound + 0.2
        )

    def test_compute_threshold_relative_tolerance(self) -> None:
        y_true, y_pred = self._binary_data()
        monitor = RiskMonitoring(
            risk="accuracy", tolerance=0.2, tolerance_type="relative"
        )

        monitor.compute_threshold(y_true, y_pred)

        assert monitor.reference_risk_upper_bound is not None
        assert monitor.threshold == pytest.approx(
            monitor.reference_risk_upper_bound * 1.2
        )

    def test_compute_threshold_rejects_invalid_tolerance_type(self) -> None:
        y_true, y_pred = self._binary_data()
        monitor = RiskMonitoring(risk="accuracy")
        monitor.tolerance_type = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Invalid tolerance type"):
            monitor.compute_threshold(y_true, y_pred)

    def test_compute_threshold_warns_and_replaces_existing_threshold(self) -> None:
        monitor = RiskMonitoring(risk="accuracy", threshold=0.1)
        y_true, y_pred = self._binary_data()

        with pytest.warns(
            UserWarning, match="Threshold is already computed and will be replaced"
        ):
            monitor.compute_threshold(y_true, y_pred)

        assert monitor.reference_risk_upper_bound is not None
        assert monitor.threshold == pytest.approx(
            monitor.reference_risk_upper_bound + monitor.tolerance
        )

    def test_compute_threshold_rejects_empty_effective_sample(self) -> None:
        y_true: NDArray[np.int_] = np.array([1, 0, 1, 0])
        y_pred: NDArray[np.int_] = np.array([0, 0, 0, 0])
        monitor = RiskMonitoring(risk="precision", warn=False)

        with pytest.raises(
            ValueError, match="Reference risk is undefined because no samples"
        ):
            monitor.compute_threshold(y_true, y_pred)

    def test_update_online_risk_requires_threshold(self) -> None:
        monitor = RiskMonitoring(risk="accuracy")
        y_true, y_pred = self._binary_data()

        with pytest.raises(ValueError, match="Threshold must be computed"):
            monitor.update_online_risk(y_true, y_pred)

    def test_update_online_risk_updates_histories_without_warning(self) -> None:
        y_true, y_pred = self._binary_data()
        monitor = RiskMonitoring(risk="accuracy", threshold=1.0, warn=False)

        returned = monitor.update_online_risk(y_true, y_pred)

        assert returned is monitor
        assert monitor.online_risk_sequence_history.size > 0
        assert monitor.online_risk_lower_bound_sequence_history.size > 0
        assert monitor.online_risk_lower_bound_latest is not None
        assert monitor.harmful_shift_detected is False

    def test_update_online_risk_warns_when_harmful_shift_detected(self) -> None:
        y_true: NDArray[np.int_] = np.zeros(20, dtype=int)
        y_pred: NDArray[np.int_] = np.ones(20, dtype=int)
        monitor = RiskMonitoring(risk="accuracy", threshold=-0.1, warn=True)

        with pytest.warns(UserWarning, match="Harmful shift detected"):
            monitor.update_online_risk(y_true, y_pred)

        assert monitor.harmful_shift_detected is True

    def test_update_online_risk_keeps_history_aligned_across_calls(self) -> None:
        y_true, y_pred = self._binary_data()
        monitor = RiskMonitoring(risk="accuracy", threshold=1.0, warn=False)

        monitor.update_online_risk(y_true, y_pred)
        first_history = monitor.online_risk_lower_bound_sequence_history.copy()
        monitor.update_online_risk(y_true, y_pred)

        assert monitor.online_risk_sequence_history.size == 2 * y_true.size
        assert (
            monitor.online_risk_lower_bound_sequence_history.size
            == monitor.online_risk_sequence_history.size
        )
        np.testing.assert_array_equal(
            first_history,
            monitor.online_risk_lower_bound_sequence_history[: first_history.size],
        )

    def test_update_online_risk_ignores_empty_effective_sample_batch(self) -> None:
        y_true: NDArray[np.int_] = np.array([1, 0, 1, 0])
        y_pred: NDArray[np.int_] = np.array([0, 0, 0, 0])
        monitor = RiskMonitoring(risk="precision", threshold=0.5, warn=False)

        returned = monitor.update_online_risk(y_true, y_pred)

        assert returned is monitor
        assert monitor.online_risk_sequence_history.size == 0
        assert monitor.online_risk_lower_bound_sequence_history.size == 0
        assert monitor.online_risk_lower_bound_latest is None

    def test_summary_can_be_called(self) -> None:
        monitor = RiskMonitoring(risk="accuracy")
        monitor.summary()
