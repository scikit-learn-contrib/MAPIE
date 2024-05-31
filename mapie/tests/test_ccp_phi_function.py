from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from mapie.regression import PhiFunction

random_state = 1
np.random.seed(random_state)

X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
z = X[:, -2:]

PHI = [
    PhiFunction([lambda X: np.ones((len(X), 1))]),
    PhiFunction([lambda X: X]),
    PhiFunction([lambda X: X, lambda z: z]),
    PhiFunction([lambda X: X, lambda y_pred: y_pred]),
    PhiFunction([
        PhiFunction([lambda X: X, lambda y_pred: y_pred]),
        lambda z: z
    ]),
]

# n_out without marginal_guarantee
N_OUT_RAW = [1, 10, 12, 11, 13, 20, 20, 3, 40]

PHI_FUNCTIONS = [
    [lambda X: np.ones((len(X), 1))],
    [lambda X: X],
    [lambda X: X, lambda z: z],
    [lambda X: X, lambda y_pred: y_pred],
    [PhiFunction([lambda X: X, lambda y_pred: y_pred]), lambda z: z],
]


# ======== PhiFunction =========
def test_phi_initialized() -> None:
    """Test that initialization does not crash."""
    PhiFunction()


def test_phi_default_parameters() -> None:
    """
    Test default values of input parameters of PhiFunction.
      - ``marginal_guarantee`` should be ``True``
      -  ``functions``should be a list with a unique callable element
    """
    phi = PhiFunction()
    assert phi.marginal_guarantee
    assert isinstance(phi.functions, list)
    assert len(phi.functions) == 0


@pytest.mark.parametrize("phi, n_out_raw", zip(PHI, N_OUT_RAW))
def test_phi_n_attributes(phi: PhiFunction, n_out_raw: int) -> None:
    """
    Test that the n_in and n_out attributes are corrects
    """
    phi(X=X, y_pred=y, z=z)
    assert phi.n_in == 10
    assert phi.n_out == n_out_raw + int(phi.marginal_guarantee)


@pytest.mark.parametrize("phi_function_1, n_out_raw_1, m_g_1",
                         zip(PHI_FUNCTIONS, N_OUT_RAW, [True, False]))
@pytest.mark.parametrize("phi_function_2, n_out_raw_2, m_g_2",
                         zip(PHI_FUNCTIONS, N_OUT_RAW, [True, False]))
@pytest.mark.parametrize("m_g_0", [True, False])
def test_phi_compound_and_guarantee(
    phi_function_1: PhiFunction, n_out_raw_1: int, m_g_1: bool,
    phi_function_2: PhiFunction, n_out_raw_2: int, m_g_2: bool,
    m_g_0: bool,
) -> None:
    """
    Test that when phi is defined using a compound of other PhiFunctions,
    the column of ones, added of marginal_guarantee, is added only once
    """
    phi_1 = PhiFunction(phi_function_1, marginal_guarantee=m_g_1)
    phi_2 = PhiFunction(phi_function_2, marginal_guarantee=m_g_2)
    phi_0 = PhiFunction([phi_1, phi_2], marginal_guarantee=m_g_0)
    phi_0(X=X, y_pred=y, z=z)

    assert phi_0.n_out == n_out_raw_1 + n_out_raw_2 + int(any(
        [m_g_0, m_g_1, m_g_2]
    ))


def test_phi_functions_warning() -> None:
    """
    Test that creating a PhiFunction object with functions which have
    optional arguments different from 'X', 'y_pred' or 'z' raise a warning.
    """
    with pytest.warns(UserWarning,
                      match="WARNING: Unknown optional arguments."):
        PhiFunction([lambda X, d=d: X**d for d in range(4)])


def test_phi_functions_error() -> None:
    """
    Test that creating a PhiFunction object with functions which have
    required arguments different from 'X', 'y_pred' or 'z' raise an error.
    """
    with pytest.raises(ValueError, match="Forbidden required argument."):
        PhiFunction([lambda X, other: X + other, lambda X, other: X - other])


def test_phi_functions_empty() -> None:
    """
    Test that creating a PhiFunction object with functions which have
    required arguments different from 'X', 'y_pred' or 'z' raise an error.
    """
    with pytest.raises(ValueError):
        PhiFunction([], marginal_guarantee=False)
