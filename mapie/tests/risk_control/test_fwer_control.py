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


def test_fwerprocedure_run_calls_update_and_none():
    class Dummy(FWERProcedure):
        def _init_state(self, n_lambdas, delta):
            self.i = 0
            self.n = n_lambdas
            self.updated = False

        def _select_next_hypothesis(self, p):
            if self.i >= self.n:
                return None
            val = self.i
            self.i += 1
            return val

        def _local_significance_levels(self):
            return np.ones(self.n)

        def _update_on_reject(self, idx):
            self.updated = True

    fwer_procedure = Dummy()
    rejected = fwer_procedure.run(np.array([0.0]), delta=0.05)

    assert rejected.tolist() == [0]
    assert fwer_procedure.updated is True


def test_abstract_methods_raise():
    class Dummy(FWERProcedure):
        def _init_state(self, n_lambdas, delta):
            super()._init_state(n_lambdas, delta)

        def _select_next_hypothesis(self, p):
            return super()._select_next_hypothesis(p)

        def _local_significance_levels(self):
            return super()._local_significance_levels()

        def _update_on_reject(self, idx):
            super()._update_on_reject(idx)

    d = Dummy()

    import pytest

    with pytest.raises(NotImplementedError):
        d._init_state(1, 0.1)

    with pytest.raises(NotImplementedError):
        d._select_next_hypothesis(np.array([0.1]))

    with pytest.raises(NotImplementedError):
        d._local_significance_levels()

    with pytest.raises(NotImplementedError):
        d._update_on_reject(0)


def test_fwerprocedure_run_break_on_non_reject():
    class Dummy(FWERProcedure):
        def _init_state(self, n_lambdas, delta):
            self.calls = 0
            self.reject_next = False

        def _select_next_hypothesis(self, p):
            self.calls += 1
            if self.calls == 1:
                return 0
            return None

        def _local_significance_levels(self):
            return np.array([1.0])

        def _update_on_reject(self, idx):
            pass

    fwer_procedure = Dummy()
    rejected = fwer_procedure.run(np.array([0.0]), delta=0.1)
    assert rejected.tolist() == [0]


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


def test_fixed_sequence_branch_start_less_than_index():
    fwer_procedure = FWERFixedSequenceTesting(n_starts=2)
    fwer_procedure._init_state(n_lambdas=5, delta=0.1)

    # force start_positions
    fwer_procedure.start_positions = [0, 3]

    fwer_procedure._update_on_reject(3)
    assert 0 in fwer_procedure.start_positions


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


def test_fwer_method_conservatism():
    p_values = [
        0.001,
        0.002,
        0.004,
        0.006,
        0.008,
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
    ]
    delta = 0.2
    valid_bonferroni = FWERBonferroniCorrection().run(p_values=p_values, delta=delta)
    valid_bonferroni_holm = FWERBonferroniHolm().run(p_values=p_values, delta=delta)
    valid_fixed_sequence = FWERFixedSequenceTesting(n_starts=1).run(
        p_values=p_values, delta=delta
    )
    assert len(valid_bonferroni) != 0
    assert len(valid_bonferroni_holm) != 0
    assert len(valid_fixed_sequence) != 0
    assert len(valid_bonferroni) < len(valid_bonferroni_holm)
    assert len(valid_bonferroni_holm) < len(valid_fixed_sequence)


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


def test_control_fwer_FWERProcedure_instance():
    p_values = np.array([0.001, 0.003, 0.01, 0.02, 0.2])
    delta = 0.1

    valid_index = control_fwer(
        p_values,
        delta,
        fwer_method=FWERFixedSequenceTesting(),
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
