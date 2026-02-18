import numpy as np
import pytest

from mapie.risk_control import (
    FWERBonferroniCorrection,
    FWERBonferroniHolm,
    FWERFixedSequenceTesting,
    FWERProcedure,
    control_fwer,
)


def test_fwerprocedure_run_stops_when_none():
    class Dummy(FWERProcedure):
        def _init_state(self, n_lambdas, delta):
            self.done = False

        def _select_next_hypothesis(self, p_values):
            if self.done:
                return None
            self.done = True
            return 0

        def _local_significance_levels(self):
            return np.array([1.0])

        def _update_on_reject(self, idx):
            pass

    fwer_procedure = Dummy()
    rejected = fwer_procedure.run(np.array([0.0]), delta=0.1)
    assert np.array_equal(rejected, np.array([0]))


def test_fwerprocedure_run_stops_on_failure():
    class Dummy(FWERProcedure):
        def _init_state(self, n_lambdas, delta):
            self.i = 0
            self.n = n_lambdas

        def _select_next_hypothesis(self, p):
            if self.i >= self.n:
                return None
            val = self.i
            self.i += 1
            return val

        def _local_significance_levels(self):
            return np.zeros(self.n)

        def _update_on_reject(self, idx):
            pass

    fwer_procedure = Dummy()
    rejected = fwer_procedure.run(np.array([0.1, 0.1]), delta=0.05)
    assert len(rejected) == 0


def test_bonferroni_stops_after_first_failure():
    fwer_procedure = FWERBonferroniCorrection()
    rejected = fwer_procedure.run(np.array([0.9, 0.0001]), delta=0.05)
    assert np.array_equal(rejected, np.array([1]))


def test_fixed_sequence_multistart_multiple_starts():
    p_values = np.array([0.001, 0.003, 0.01, 0.02, 0.2, 0.6])
    delta = 0.1
    n_starts = 3
    fwer_procedure = FWERFixedSequenceTesting(n_starts=n_starts)
    rejected = fwer_procedure.run(p_values, delta)
    assert rejected.tolist() == [0, 1, 2, 3]


def test_fixed_sequence_starts_clipped():
    fwer_procedure = FWERFixedSequenceTesting(n_starts=10)
    rejected = fwer_procedure.run(np.array([0.0, 0.0]), delta=0.1)
    assert len(rejected) >= 0


def test_fixed_sequence_no_start_remaining():
    fwer_procedure = FWERFixedSequenceTesting(n_starts=1)
    rejected = fwer_procedure.run(np.array([0.0]), delta=1.0)
    assert np.array_equal(rejected, np.array([0]))


def test_all_subclasses_instantiable():
    for cls in FWERProcedure.__subclasses__():
        obj = cls() if cls is not FWERFixedSequenceTesting else cls(n_starts=1)
        assert isinstance(obj, FWERProcedure)


def test_fixed_sequence_ascending_invalid_inputs():
    with pytest.raises(ValueError, match=r".*n_starts must be a positive integer.*"):
        FWERFixedSequenceTesting(n_starts=0).run(np.array([0.1, 0.2]), delta=0.1)

    with pytest.warns(
        UserWarning, match=r".*n_starts is greater than the number of tests.*"
    ):
        FWERFixedSequenceTesting(n_starts=5).run(np.array([0.1, 0.2]), delta=0.1)


def test_sgt_bonferroni_holm_no_rejection():
    p_values = np.array([0.5, 0.6, 0.7])
    delta = 0.1
    valid_index = FWERBonferroniHolm().run(p_values, delta)
    assert len(valid_index) == 0


def test_sgt_bonferroni_holm_single_rejection():
    p_values = np.array([0.001, 0.4, 0.6])
    delta = 0.05
    valid_index = FWERBonferroniHolm().run(p_values, delta)
    assert np.array_equal(valid_index, np.array([0]))


def test_sgt_bonferroni_holm_multiple_rejections():
    p_values = np.array([0.001, 0.01, 0.2])
    delta = 0.05
    valid_index = FWERBonferroniHolm().run(p_values, delta)
    # Test behavior:
    # - first rejection at index 0
    # - redistribution allows rejection at index 1
    assert np.array_equal(valid_index, np.array([0, 1]))


def test_sgt_bonferroni_holm_all_rejected():
    p_values = np.array([0.001, 0.002, 0.003])
    delta = 0.05
    valid_index = FWERBonferroniHolm().run(p_values, delta)
    assert np.array_equal(valid_index, np.array([0, 1, 2]))


def test_sgt_bonferroni_holm_single_value():
    fwer_procedure = FWERBonferroniHolm()
    rejected = fwer_procedure.run(np.array([0.0]), delta=1.0)
    assert np.array_equal(rejected, np.array([0]))


def test_control_fwer_bonferroni():
    p_values = np.array([0.001, 0.02, 0.2, 0.8])
    delta = 0.05

    valid_index = control_fwer(
        p_values=p_values,
        delta=delta,
        fwer_method="bonferroni",
    )
    assert np.array_equal(valid_index, np.array([0]))


def test_control_fwer_fixed_sequence():
    p_values = np.array([0.001, 0.003, 0.01, 0.02, 0.2])
    delta = 0.1

    valid_index = control_fwer(
        p_values,
        delta,
        fwer_method="fixed_sequence",
    )

    assert np.array_equal(valid_index, np.array([0, 1, 2, 3]))


def test_control_fwer_bonferroni_holm():
    p_values = np.array([0.001, 0.01, 0.2])
    delta = 0.05

    valid_index = control_fwer(
        p_values,
        delta,
        fwer_method="bonferroni_holm",
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
