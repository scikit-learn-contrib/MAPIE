import numpy as np

from mapie.risk_control import fwer_control


def test_fwer_control_bonferroni():
    p_values = np.array([0.001, 0.02, 0.2, 0.8])
    delta = 0.05

    valid_index = fwer_control(
        p_values=p_values,
        delta=delta,
        fwer_graph="bonferroni",
    )
    assert np.array_equal(valid_index, np.array([0]))
