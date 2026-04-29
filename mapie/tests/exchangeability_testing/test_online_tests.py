import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from mapie.exchangeability_testing.exchangeability import (
    FixedDatasetExchangeabilityTest,
    OnlineExchangeabilityTest,
)
import mapie.exchangeability_testing.exchangeability as et_module
import mapie.exchangeability_testing.martingales as omt_module
from mapie.exchangeability_testing.martingales import OnlineMartingaleTest


def test_fixed_dataset_exchangeability_validation_errors():
    """Test fixed-dataset wrapper validation on method names."""
    with pytest.raises(ValueError, match=r"Invalid method_names type"):
        FixedDatasetExchangeabilityTest(method_names=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"Invalid method name: not_a_method"):
        FixedDatasetExchangeabilityTest(method_names="not_a_method")  # type: ignore[arg-type]


def test_fixed_dataset_exchangeability_accepts_list_of_method_names():
    """Test fixed-dataset wrapper accepts a list of method names."""
    wrapper = FixedDatasetExchangeabilityTest(
        method_names=["pvalue_permutation", "jumper_martingale"]
    )

    assert wrapper.method_names == ["pvalue_permutation", "jumper_martingale"]


def test_online_exchangeability_validation_errors():
    """Test online wrapper validation on method names."""
    with pytest.raises(ValueError, match=r"Invalid method_names type"):
        OnlineExchangeabilityTest(method_names=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"Invalid method name: not_a_method"):
        OnlineExchangeabilityTest(method_names="not_a_method")  # type: ignore[arg-type]


def test_online_exchangeability_accepts_list_of_method_names():
    """Test online wrapper accepts a list of method names."""
    wrapper = OnlineExchangeabilityTest(
        method_names=["plugin_martingale", "jumper_martingale"]
    )

    assert wrapper.method_names == ["plugin_martingale", "jumper_martingale"]


def test_init_validation_errors():
    """Test that invalid initialization parameters raise ValueError."""
    with pytest.raises(ValueError, match=r"test_level must lie in \(0, 1\)"):
        OnlineMartingaleTest(test_level=1.0)

    with pytest.raises(ValueError, match=r".*test_method must be one of.*"):
        OnlineMartingaleTest(test_method="invalid")

    with pytest.raises(ValueError, match=r"jump_size must lie in \(0, 1\)"):
        OnlineMartingaleTest(jump_size=-0.1)


def test_fixed_dataset_exchangeability_injects_martingale_test_method():
    """Test fixed-dataset wrapper injects test_method for martingales."""
    plugin_test = FixedDatasetExchangeabilityTest(method_names="plugin_martingale")
    jumper_test = FixedDatasetExchangeabilityTest(method_names="jumper_martingale")
    assert plugin_test.test_methods[0].test_method == "plugin_martingale"
    assert jumper_test.test_methods[0].test_method == "jumper_martingale"


def test_fixed_dataset_exchangeability_method_params_override_injected_default():
    """Test explicit method_params take precedence over injected test_method."""
    wrapper = FixedDatasetExchangeabilityTest(
        method_names="plugin_martingale",
        method_params={"plugin_martingale": {"test_method": "jumper_martingale"}},
    )
    assert wrapper.test_methods[0].test_method == "jumper_martingale"


def test_fixed_dataset_exchangeability_injects_sequential_mc_strategy():
    """Test fixed-dataset wrapper injects strategy for SMC methods."""
    bin_test = FixedDatasetExchangeabilityTest(method_names="permutation_binomial")
    mix_test = FixedDatasetExchangeabilityTest(
        method_names="permutation_binomial_mixture"
    )
    agg_test = FixedDatasetExchangeabilityTest(method_names="permutation_aggressive")

    assert bin_test.test_methods[0].strategy == "binomial"
    assert mix_test.test_methods[0].strategy == "binomial_mixture"
    assert agg_test.test_methods[0].strategy == "aggressive"


def test_fixed_dataset_exchangeability_override_injected_smc_strategy():
    """Test explicit method_params override injected SMC strategy."""
    wrapper = FixedDatasetExchangeabilityTest(
        method_names="permutation_binomial",
        method_params={"permutation_binomial": {"strategy": "aggressive"}},
    )
    assert wrapper.test_methods[0].strategy == "aggressive"


def test_fixed_dataset_exchangeability_is_exchangeable_returns_by_method():
    """Test fixed-dataset wrapper exposes each method exchangeability decision."""
    wrapper = FixedDatasetExchangeabilityTest(method_names="all")
    wrapper.test_methods = [
        MagicMock(is_exchangeable=True),
        MagicMock(is_exchangeable=False),
        MagicMock(is_exchangeable=None),
        MagicMock(is_exchangeable=True),
        MagicMock(is_exchangeable=False),
        MagicMock(is_exchangeable=None),
    ]

    assert wrapper.is_exchangeable == {
        "pvalue_permutation": True,
        "permutation_binomial": False,
        "permutation_binomial_mixture": None,
        "permutation_aggressive": True,
        "plugin_martingale": False,
        "jumper_martingale": None,
    }


def test_fixed_dataset_exchangeability_run_calls_update_when_available():
    """Test fixed-dataset wrapper prefers update when both APIs exist."""
    wrapper = FixedDatasetExchangeabilityTest(method_names="pvalue_permutation")
    X = np.array([[1.0], [2.0]])
    y = np.array([0.0, 1.0])
    update_result = object()
    run_result = object()
    test_method = MagicMock()
    test_method.update = MagicMock(return_value=update_result)
    test_method.run = MagicMock(return_value=run_result)
    wrapper.test_methods = [test_method]

    results = wrapper.run(X, y)

    test_method.update.assert_called_once_with(X, y)
    test_method.run.assert_not_called()
    assert results == {"pvalue_permutation": update_result}


def test_fixed_dataset_exchangeability_run_calls_run_when_update_missing():
    """Test fixed-dataset wrapper falls back to run when needed."""
    wrapper = FixedDatasetExchangeabilityTest(method_names="pvalue_permutation")
    X = np.array([[1.0], [2.0]])
    y = np.array([0.0, 1.0])
    run_result = object()

    class RunOnlyMethod:
        def __init__(self):
            self.run = MagicMock(return_value=run_result)
            self.is_exchangeable = None

    test_method = RunOnlyMethod()
    wrapper.test_methods = [test_method]

    results = wrapper.run(X, y)

    test_method.run.assert_called_once_with(X, y)
    assert results == {"pvalue_permutation": run_result}


def test_fixed_dataset_exchangeability_run_raises_when_no_supported_api():
    """Test fixed-dataset wrapper errors on invalid test method API."""
    wrapper = FixedDatasetExchangeabilityTest(method_names="pvalue_permutation")
    wrapper.test_methods = [object()]

    with pytest.raises(AttributeError, match=r"must define either 'update' or 'run'"):
        wrapper.run(np.array([[1.0]]), np.array([1.0]))


def test_online_exchangeability_injects_martingale_test_method():
    """Test online wrapper injects test_method for martingales."""
    plugin_test = OnlineExchangeabilityTest(method_names="plugin_martingale")
    jumper_test = OnlineExchangeabilityTest(method_names="jumper_martingale")

    assert plugin_test.test_methods[0].test_method == "plugin_martingale"
    assert jumper_test.test_methods[0].test_method == "jumper_martingale"


def test_online_exchangeability_method_params_override_injected_default():
    """Test explicit method_params take precedence over injected test_method."""
    wrapper = OnlineExchangeabilityTest(
        method_names="plugin_martingale",
        method_params={"plugin_martingale": {"test_method": "jumper_martingale"}},
    )

    assert wrapper.test_methods[0].test_method == "jumper_martingale"


def test_online_exchangeability_does_not_inject_test_method_for_other_classes(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test _init_test_method path when class is not OnlineMartingaleTest."""

    class DummyOnlineMethod:
        def __init__(self, test_level, warn, extra_flag=False):  # noqa: ANN001
            self.test_level = test_level
            self.warn = warn
            self.extra_flag = extra_flag

    monkeypatch.setitem(
        et_module.online_test_method_choice_map,
        "dummy_online",
        DummyOnlineMethod,
    )

    wrapper = OnlineExchangeabilityTest(
        method_names="dummy_online",  # type: ignore[arg-type]
        method_params={"dummy_online": {"extra_flag": True}},
    )
    method = wrapper.test_methods[0]
    assert isinstance(method, DummyOnlineMethod)
    assert method.extra_flag is True


def test_online_exchangeability_is_exchangeable_returns_by_method():
    """Test online wrapper exposes each method exchangeability decision."""
    wrapper = OnlineExchangeabilityTest(method_names="all")
    wrapper.test_methods[0].pvalue_history = [0.1]
    wrapper.test_methods[0].martingale_value_history = [1.0]
    wrapper.test_methods[0].current_martingale_value = 1.0
    wrapper.test_methods[0].burn_in = 1
    wrapper.test_methods[1].pvalue_history = [0.1]
    wrapper.test_methods[1].martingale_value_history = [100.0]
    wrapper.test_methods[1].current_martingale_value = 100.0
    wrapper.test_methods[1].burn_in = 1

    assert wrapper.is_exchangeable == {
        "plugin_martingale": True,
        "jumper_martingale": False,
    }


def test_online_exchangeability_update_calls_test_method_update():
    """Test online wrapper forwards update arguments to each test method."""
    wrapper = OnlineExchangeabilityTest(method_names="all")
    X = np.array([[1.0], [2.0]])
    y = np.array([0.0, 1.0])

    first_result = object()
    second_result = object()
    wrapper.test_methods[0].update = MagicMock(return_value=first_result)
    wrapper.test_methods[1].update = MagicMock(return_value=second_result)

    results = wrapper.update(X, y)

    wrapper.test_methods[0].update.assert_called_once_with(X, y)
    wrapper.test_methods[1].update.assert_called_once_with(X, y)
    assert results == {
        "plugin_martingale": first_result,
        "jumper_martingale": second_result,
    }


def test_online_exchangeability_update_raises_when_update_missing():
    """Test online wrapper errors when a method does not define update."""
    wrapper = OnlineExchangeabilityTest(method_names="plugin_martingale")
    wrapper.test_methods = [object()]

    with pytest.raises(AttributeError, match=r"must define an 'update' method"):
        wrapper.update(np.array([[1.0]]), np.array([1.0]))


def test_reject_threshold_computation():
    """Test that reject_threshold is computed correctly from test_level."""
    omt = OnlineMartingaleTest(test_level=0.1)
    assert omt.reject_threshold == pytest.approx(10.0)

    omt2 = OnlineMartingaleTest(test_level=0.05)
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
    omt = OnlineMartingaleTest(random_state=1234)
    rng = np.random.default_rng(1234)
    rng.uniform()
    expected = float(rng.uniform())

    actual = omt.compute_p_value(
        current_conformity_score=0.5, conformity_score_history=np.asarray([])
    )

    assert actual == pytest.approx(expected)


def test_compute_p_value_with_history_and_ties():
    """Test p-value computation with history and tied scores."""
    omt = OnlineMartingaleTest(random_state=1234)
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
    omt = OnlineMartingaleTest(burn_in=1)
    omt.pvalue_history = [0.1]  # 1 pvalue
    omt.martingale_value_history = [21.0]  # One value above threshold (20.0)
    omt.current_martingale_value = 1.0  # Current value below threshold

    assert omt.is_exchangeable is False


def test_estimate_pvalues_density_returns_uniform_for_empty_history():
    """Test that density estimate returns uniform for empty p-value history."""
    omt = OnlineMartingaleTest(random_state=0)
    assert omt._estimate_pvalues_density(0.5) == 1.0


def test_estimate_pvalues_density_invalid_pvalue():
    """Test that invalid p-value raises ValueError."""
    omt = OnlineMartingaleTest(random_state=0)
    omt.pvalue_history = [0.1, 0.2]

    with pytest.raises(ValueError, match=r"pvalue must lie in \[0, 1\]"):
        omt._estimate_pvalues_density(-0.1)


def test_estimate_pvalues_density_returns_uniform_when_normalization_invalid(
    monkeypatch,
):
    """Test that density estimate returns uniform when normalization is invalid."""
    omt = OnlineMartingaleTest(random_state=0)
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
    omt = OnlineMartingaleTest(random_state=0)
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
    omt = OnlineMartingaleTest(random_state=0)
    result = omt.update_simple_jumper_martingale(0.3)

    assert result == pytest.approx(omt.current_martingale_value)
    assert omt.martingale_value_history[-1] == pytest.approx(result)
    assert len(omt.martingale_value_history) == 1
    assert omt._jumper_wealth_by_expert.shape == (3,)


def test_update_simple_jumper_martingale_invalid_pvalue():
    """Test that invalid p-value raises ValueError in jumper martingale update."""
    omt = OnlineMartingaleTest(random_state=0)

    with pytest.raises(ValueError, match=r"pvalue must lie in \[0, 1\]"):
        omt.update_simple_jumper_martingale(1.5)


def test_update_plugin_martingale_uses_density_estimate():
    """Test that plugin martingale update uses density estimate."""
    omt = OnlineMartingaleTest(
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
        test_method="plugin_martingale",
        test_level=0.5,
        burn_in=1,
        warn=True,
        random_state=0,
    )
    omt.current_martingale_value = 100.0
    omt.pvalue_history = [0.1, 0.2, 0.3]
    omt.conformity_score_history = [0.1, 0.2, 0.3]
    omt._compute_non_conformity_scores = lambda X, y: np.asarray([0.5])  # type: ignore[method-assign]

    with pytest.warns(
        UserWarning,
        match=r".*The online martingale test has rejected exchangeability.*",
    ):
        omt.update(np.array([1.0]), np.array([[1.0]]))

    assert omt._warning_already_raised is True


def test_update_unsupported_method_raises():
    """Test that update raises ValueError for unsupported test method."""
    omt = OnlineMartingaleTest(random_state=0)
    omt.test_method = "best_martingale"
    omt._compute_non_conformity_scores = lambda X, y: np.asarray([0.5])  # type: ignore[method-assign]

    with pytest.raises(ValueError, match=r".*Unsupported test method.*"):
        omt.update(np.array([1.0]), np.array([[1.0]]))


def test_update_appends_scores_and_pvalues():
    """Test that update appends scores and p-values to history."""
    omt = OnlineMartingaleTest(burn_in=1, random_state=1234)
    omt._compute_non_conformity_scores = lambda X, y: np.asarray([0.1, 0.2])  # type: ignore[method-assign]
    omt.update(np.array([1.0, 2.0]), np.array([[10.0], [20.0]]))

    assert len(omt.conformity_score_history) == 2
    assert len(omt.pvalue_history) == 2
    assert omt.current_martingale_value == pytest.approx(1.0)


def test_summary_without_values():
    """Test that summary returns None values when no martingale values exist."""
    omt = OnlineMartingaleTest()
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
    omt = OnlineMartingaleTest(test_level=0.05, burn_in=1)
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
    omt = OnlineMartingaleTest(test_level=0.1, burn_in=1)
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
    omt = OnlineMartingaleTest(test_level=0.1, burn_in=1)
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
    omt = OnlineMartingaleTest(test_level=0.1, burn_in=1)
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
        test_method="plugin_martingale",
        test_level=0.5,
        burn_in=1,
        warn=False,
        random_state=0,
    )
    omt.current_martingale_value = 100.0
    omt.pvalue_history = [0.1, 0.2, 0.3]
    omt.conformity_score_history = [0.1, 0.2, 0.3]
    omt._compute_non_conformity_scores = lambda X, y: np.asarray([0.5])  # type: ignore[method-assign]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        omt.update(np.array([1.0]), np.array([[1.0]]))
        user_warnings = [
            warning for warning in w if issubclass(warning.category, UserWarning)
        ]
        assert len(user_warnings) == 0


def test_is_exchangeable_with_exactly_one_value_above_threshold():
    """Test that exactly one value above threshold returns False."""
    omt = OnlineMartingaleTest(test_level=0.05, burn_in=1)
    omt.pvalue_history = [0.1]  # Above threshold (20.0)
    omt.martingale_value_history = [21.0]  # One value above threshold
    omt.current_martingale_value = 1.0  # Current value below threshold

    assert omt.is_exchangeable is False


def test_prepare_estimator_resets_and_clears_state_on_copy():
    """Test that estimator state is reset on a deep-copied estimator."""

    class DummyInnerEstimator:
        def __init__(self):
            self.conformity_scores_ = np.array([1.0])
            self.quantiles_ = np.array([0.5])

    class DummyEstimator:
        def __init__(self):
            self._is_conformalized = True
            self._predict_params = {"alpha": 0.1}
            self.conformity_scores_ = np.array([1.0])
            self.quantiles_ = np.array([0.5])
            self._mapie_regressor = DummyInnerEstimator()
            self._mapie_classifier = DummyInnerEstimator()

    omt = OnlineMartingaleTest()
    estimator = DummyEstimator()

    prepared = omt._prepare_estimator(estimator)

    assert prepared is not estimator
    assert prepared._is_conformalized is False
    assert prepared._predict_params == {}
    assert not hasattr(prepared, "conformity_scores_")
    assert not hasattr(prepared, "quantiles_")
    assert not hasattr(prepared._mapie_regressor, "conformity_scores_")
    assert not hasattr(prepared._mapie_regressor, "quantiles_")
    assert not hasattr(prepared._mapie_classifier, "conformity_scores_")
    assert not hasattr(prepared._mapie_classifier, "quantiles_")

    # Ensure the original object is not modified.
    assert estimator._is_conformalized is True
    assert estimator._predict_params == {"alpha": 0.1}
    assert hasattr(estimator, "conformity_scores_")
    assert hasattr(estimator, "quantiles_")


def test_prepare_estimator_raises_on_forbidden_estimators(monkeypatch):
    """Test forbidden estimators raise a ValueError."""

    class ForbiddenEstimator:
        pass

    monkeypatch.setattr(omt_module, "CrossConformalClassifier", ForbiddenEstimator)
    monkeypatch.setattr(omt_module, "CrossConformalRegressor", ForbiddenEstimator)
    monkeypatch.setattr(
        omt_module,
        "JackknifeAfterBootstrapRegressor",
        ForbiddenEstimator,
    )

    omt = OnlineMartingaleTest()
    with pytest.raises(ValueError, match=r"supported in permutation tests"):
        omt._prepare_estimator(ForbiddenEstimator())


def test_prepare_estimator_with_partial_inner_estimators_and_missing_attrs():
    """Test _prepare_estimator when inner estimators/attrs are only partially present."""

    class DummyInnerEstimator:
        def __init__(self):
            # Keep only one attribute to trigger both hasattr True/False branches.
            self.conformity_scores_ = np.array([1.0])

    class DummyEstimator:
        def __init__(self):
            self._is_conformalized = True
            self._predict_params = {"alpha": 0.1}
            # Keep top-level attribute absent and present combinations.
            self.quantiles_ = np.array([0.5])
            # Provide only one inner estimator to exercise missing-branch path.
            self._mapie_regressor = DummyInnerEstimator()

    omt = OnlineMartingaleTest()
    estimator = DummyEstimator()

    prepared = omt._prepare_estimator(estimator)

    assert prepared is not estimator
    assert prepared._is_conformalized is False
    assert prepared._predict_params == {}
    assert not hasattr(prepared, "quantiles_")
    assert not hasattr(prepared, "conformity_scores_")
    assert hasattr(prepared, "_mapie_regressor")
    assert not hasattr(prepared._mapie_regressor, "conformity_scores_")
    # quantiles_ was never present in inner estimator: ensure branch with missing attr is covered.
    assert not hasattr(prepared._mapie_regressor, "quantiles_")
    # _mapie_classifier does not exist: ensure missing inner-estimator branch is exercised.
    assert not hasattr(prepared, "_mapie_classifier")


def test_prepare_estimator_without_optional_state_attrs():
    """Test _prepare_estimator when optional top-level attrs are missing."""

    class DummyEstimator:
        # Intentionally do not define _is_conformalized or _predict_params.
        pass

    omt = OnlineMartingaleTest()
    estimator = DummyEstimator()

    prepared = omt._prepare_estimator(estimator)

    assert prepared is not estimator
    assert not hasattr(prepared, "_is_conformalized")
    assert not hasattr(prepared, "_predict_params")


def test_infer_task_from_estimator_and_target_type():
    """Test task inference from estimator attributes and from y type."""

    class DummyEstimator:
        pass

    omt = OnlineMartingaleTest()

    cls_estimator = DummyEstimator()
    cls_estimator._mapie_classifier = object()
    omt.mapie_estimator = cls_estimator
    assert omt._infer_task(np.array([0, 1])) == "classification"

    reg_estimator = DummyEstimator()
    reg_estimator._mapie_regressor = object()
    omt.mapie_estimator = reg_estimator
    assert omt._infer_task(np.array([0.1, 0.2])) == "regression"

    omt.mapie_estimator = DummyEstimator()
    with pytest.raises(ValueError, match=r"Unable to infer the task"):
        omt._infer_task(np.array([0, 1]))

    omt.mapie_estimator = None
    assert omt._infer_task(np.array([0, 1, 0])) == "classification"
    assert omt._infer_task(np.array([0.1, 0.2, 0.3])) == "regression"

    with pytest.raises(ValueError, match=r"Unknown type of target"):
        omt._infer_task(np.array([[1, 0], [0, 1]]))


def test_compute_non_conformity_scores_classification_branch(monkeypatch):
    """Test classification path, including estimator initialization and fitting."""

    class DummyMapieClassifier:
        def __init__(self):
            self._is_fitted = False
            self._mapie_classifier = type("ScoreHolder", (), {})()
            self._mapie_classifier.conformity_scores_ = np.array([0.3, 0.7])
            self.fit_calls = 0
            self.conformalize_calls = 0

        def fit(self, X, y):
            self.fit_calls += 1
            self._is_fitted = True
            self._fit_X = np.asarray(X)
            self._fit_y = np.asarray(y)

        def conformalize(self, X, y):
            self.conformalize_calls += 1
            self._conformalize_X = np.asarray(X)
            self._conformalize_y = np.asarray(y)

    estimator = DummyMapieClassifier()

    def fake_train_test_split(X, y, test_size, shuffle):
        assert test_size == pytest.approx(0.7)
        assert shuffle is False
        return X[:1], X[1:], y[:1], y[1:]

    monkeypatch.setattr(
        omt_module,
        "SplitConformalClassifier",
        lambda prefit=False: estimator,
    )
    monkeypatch.setattr(omt_module, "train_test_split", fake_train_test_split)

    omt = OnlineMartingaleTest(task=None)
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 0])

    scores = omt._compute_non_conformity_scores(X, y)

    assert np.array_equal(scores, np.array([0.3, 0.7]))
    assert omt.task == "classification"
    assert estimator.fit_calls == 1
    assert estimator.conformalize_calls == 1
    assert np.array_equal(estimator._fit_X, np.array([[0.0]]))
    assert np.array_equal(estimator._fit_y, np.array([0]))
    assert np.array_equal(estimator._conformalize_X, np.array([[1.0], [2.0]]))
    assert np.array_equal(estimator._conformalize_y, np.array([1, 0]))


def test_compute_non_conformity_scores_regression_branch(monkeypatch):
    """Test regression path and returned score extraction from regressor."""

    class DummyMapieRegressor:
        def __init__(self):
            self._is_fitted = True
            self._mapie_regressor = type("ScoreHolder", (), {})()
            self._mapie_regressor.conformity_scores_ = np.array([1.1, 2.2])
            self.conformalize_calls = 0

        def conformalize(self, X, y):
            self.conformalize_calls += 1

    estimator = DummyMapieRegressor()

    monkeypatch.setattr(
        omt_module,
        "SplitConformalRegressor",
        lambda prefit=False: estimator,
    )

    omt = OnlineMartingaleTest(task="regression")

    scores = omt._compute_non_conformity_scores(
        np.array([[0.0], [1.0]]),
        np.array([0.2, 0.4]),
    )

    assert np.array_equal(scores, np.array([1.1, 2.2]))
    assert estimator.conformalize_calls == 1


def test_compute_non_conformity_scores_uses_provided_estimator_without_creation():
    """Test branch where a provided mapie_estimator is reused as-is."""

    class DummyProvidedRegressor:
        def __init__(self):
            self._is_fitted = True
            self._mapie_regressor = type("ScoreHolder", (), {})()
            self.conformalize_calls = 0

        def conformalize(self, X, y):
            self.conformalize_calls += 1
            self._mapie_regressor.conformity_scores_ = np.array([0.9, 1.1])

    provided = DummyProvidedRegressor()
    omt = OnlineMartingaleTest(mapie_estimator=provided, task="regression")

    scores = omt._compute_non_conformity_scores(
        np.array([[0.0], [1.0]]),
        np.array([0.2, 0.4]),
    )

    assert omt.mapie_estimator is not provided
    assert np.array_equal(scores, np.array([0.9, 1.1]))
    assert omt.mapie_estimator.conformalize_calls == 1


def test_compute_non_conformity_scores_can_reuse_estimator_across_updates():
    """Test repeated conformalization calls reuse one estimator safely."""

    class DummyProvidedRegressor:
        def __init__(self):
            self._is_fitted = True
            self._mapie_regressor = type("ScoreHolder", (), {})()
            self.conformalize_calls = 0

        def conformalize(self, X, y):
            self.conformalize_calls += 1
            self._mapie_regressor.conformity_scores_ = np.asarray(y, dtype=float)

    omt = OnlineMartingaleTest(
        mapie_estimator=DummyProvidedRegressor(),
        task="regression",
    )

    first_scores = omt._compute_non_conformity_scores(
        np.array([[0.0], [1.0]]),
        np.array([0.2, 0.4]),
    )
    second_scores = omt._compute_non_conformity_scores(
        np.array([[2.0], [3.0]]),
        np.array([0.6, 0.8]),
    )

    assert np.array_equal(first_scores, np.array([0.2, 0.4]))
    assert np.array_equal(second_scores, np.array([0.6, 0.8]))
    assert omt.mapie_estimator.conformalize_calls == 2


def test_compute_non_conformity_scores_warns_when_estimator_not_fitted(monkeypatch):
    """Test warning is raised when estimator is not fitted."""

    class DummyProvidedRegressor:
        def __init__(self):
            self._is_fitted = False
            self._mapie_regressor = type("ScoreHolder", (), {})()
            self._mapie_regressor.conformity_scores_ = np.array([0.4, 0.6])
            self.fit_calls = 0

        def fit(self, X, y):
            self.fit_calls += 1
            self._is_fitted = True

        def conformalize(self, X, y):
            self._mapie_regressor.conformity_scores_ = np.array([0.4, 0.6])

    def fake_train_test_split(X, y, test_size, shuffle):
        assert test_size == pytest.approx(0.7)
        assert shuffle is False
        return X[:1], X[1:], y[:1], y[1:]

    monkeypatch.setattr(omt_module, "train_test_split", fake_train_test_split)

    omt = OnlineMartingaleTest(
        mapie_estimator=DummyProvidedRegressor(),
        task="regression",
    )

    with pytest.warns(
        UserWarning,
        match=r"The provided MAPIE estimator is not fitted\.",
    ):
        scores = omt._compute_non_conformity_scores(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([0.1, 0.2, 0.3]),
        )

    assert np.array_equal(scores, np.array([0.4, 0.6]))
    assert omt.mapie_estimator.fit_calls == 1


def test_compute_non_conformity_scores_raises_on_unknown_task():
    """Test unknown task raises ValueError when estimator must be created."""
    omt = OnlineMartingaleTest(task="not-a-task")

    with pytest.raises(ValueError, match=r"Unknown task type"):
        omt._compute_non_conformity_scores(np.array([[1.0]]), np.array([1.0]))


def test_update_raises_on_mismatched_number_of_rows():
    """Test update validates that X and y have matching row counts."""
    omt = OnlineMartingaleTest()

    with pytest.raises(ValueError, match=r"X and y must have the same number of rows"):
        omt.update(np.array([[1.0], [2.0]]), np.array([1.0]))


def test_is_exchangeable_with_two_values_above_threshold():
    """Test that two or more values above threshold returns False."""
    omt = OnlineMartingaleTest(test_level=0.05, burn_in=1)
    omt.pvalue_history = [0.1, 0.2]  # At least 2 pvalues
    omt.martingale_value_history = [21.0, 22.0]  # Two values above threshold (20.0)
    omt.current_martingale_value = 1.0  # Current value below threshold doesn't matter

    assert omt.is_exchangeable is False


def test_summary_martingale_statistics_completeness():
    """Test that all martingale statistics are returned with values."""
    omt = OnlineMartingaleTest(test_level=0.05, burn_in=1)
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
    omt = OnlineMartingaleTest(random_state=42)
    history = np.array([0.5, 1.0, 1.5, 2.0])

    pvalue = omt.compute_p_value(
        current_conformity_score=1.2, conformity_score_history=history
    )

    assert 0.0 <= pvalue <= 1.0
    assert pvalue != 0.0
    assert pvalue != 1.0


def test_initiate_estimator_classification():
    """Test that _initiate_estimator creates SplitConformalClassifier for classification."""
    omt = OnlineMartingaleTest(task="classification")
    result = omt._initiate_estimator()

    from mapie.classification import SplitConformalClassifier

    assert isinstance(omt.mapie_estimator, SplitConformalClassifier)
    assert result is omt


def test_initiate_estimator_regression():
    """Test that _initiate_estimator creates SplitConformalRegressor for regression."""
    omt = OnlineMartingaleTest(task="regression")
    result = omt._initiate_estimator()

    from mapie.regression import SplitConformalRegressor

    assert isinstance(omt.mapie_estimator, SplitConformalRegressor)
    assert result is omt


def test_initiate_estimator_unknown_task():
    """Test that _initiate_estimator raises ValueError for unknown task type."""
    omt = OnlineMartingaleTest(task="unknown_task")

    with pytest.raises(ValueError, match=r"Unknown task type"):
        omt._initiate_estimator()
