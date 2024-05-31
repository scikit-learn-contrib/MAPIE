from __future__ import annotations

from typing import Any, List, Dict

import numpy as np
import pytest
from sklearn.datasets import make_regression
from mapie.regression import (PhiFunction, PolynomialPhiFunction,
                              GaussianPhiFunction)

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
    PolynomialPhiFunction(2, "X", marginal_guarantee=True),
    PolynomialPhiFunction([1, 2], "X", marginal_guarantee=True),
    PolynomialPhiFunction([1, 4, 5], "y_pred", marginal_guarantee=False),
    GaussianPhiFunction(4, X=X)
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

GAUSS_NEED_CALIB_SETTINGS: List[Dict[str, Any]] = [
    {
        "points": 10,
        "sigma": 1,
        "X": None,
    },
    {
        "points": 10,
        "sigma": None,
        "X": None,
    },
    {
        "points": np.ones((2, X.shape[1])),
        "sigma": None,
        "X": None,
    },
]

GAUSS_NO_NEED_CALIB_SETTINGS: List[Dict[str, Any]] = [
    {
        "points": 10,
        "sigma": 1,
        "X": X,
    },
    {
        "points": 10,
        "sigma": None,
        "X": X,
    },
    {
        "points": np.ones((2, X.shape[1])),
        "sigma": None,
        "X": X,
    },
    {
        "points": np.ones((2, X.shape[1])),
        "sigma": np.ones(X.shape[1]),
        "X": X,
    },
    {
        "points": (np.ones((2, X.shape[1])), [1, 2]),
        "sigma": None,
        "X": None,
    },
    {
        "points": (np.ones((2, X.shape[1])), np.ones((2, X.shape[1]))),
        "sigma": None,
        "X": None,
    },
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


# ======== PolynomialPhiFunction =========
def test_poly_phi_init() -> None:
    """Test that initialization does not crash."""
    PolynomialPhiFunction()


@pytest.mark.parametrize("degree", [2, [0, 1, 3]])
@pytest.mark.parametrize("variable", ["X", "y_pred", "z"])
@pytest.mark.parametrize("marginal_guarantee", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
def test_poly_phi_init_other(
    degree: Any, variable: Any, marginal_guarantee: bool, normalized: bool
) -> None:
    """Test that initialization does not crash."""
    PolynomialPhiFunction(degree, variable, marginal_guarantee, normalized)


@pytest.mark.parametrize("var", ["other", 1, np.ones((10, 1))])
def test_invalid_variable_value(var: Any) -> None:
    """
    Test that invalid variable value raise error
    """
    with pytest.raises(ValueError):
        PolynomialPhiFunction(variable=var)


# ======== GaussianPhiFunction =========
def test_gauss_phi_init() -> None:
    """Test that initialization does not crash."""
    GaussianPhiFunction()


@pytest.mark.parametrize("points", [3, [[10, 20], [2, 39], [2, 3]],
                                    ([[1], [2], [3]], [1, 2, 3])])
@pytest.mark.parametrize("sigma", [None, 1, [1, 2]])
@pytest.mark.parametrize("random_sigma", [True, False])
@pytest.mark.parametrize("X", [None, np.ones((30, 2))])
@pytest.mark.parametrize("marginal_guarantee", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
def test_poly_gauss_init_other(
    points: Any, sigma: Any, random_sigma: Any, X: Any,
    marginal_guarantee: bool, normalized: bool
) -> None:
    """Test that initialization does not crash."""
    GaussianPhiFunction(points, sigma, random_sigma, X,
                        marginal_guarantee, normalized)


@pytest.mark.parametrize("points", [np.ones((10)),
                                    np.ones((10, 2, 2)),
                                    (np.ones((10, 3)), np.ones((10, 2))),
                                    (np.ones((10, 3)), np.ones((8, 1))),
                                    (np.ones((10, 3)), np.ones(7))])
def test_invalid_gauss_points(points: Any) -> None:
    """
    Test that invalid ``GaussianPhiFunction`` ``points``argument values raise an
    error
    """
    with pytest.raises(ValueError):
        GaussianPhiFunction(points)


@pytest.mark.parametrize("sigma", ["1",
                                   np.ones((10, 2)),
                                   np.ones((8, 1)),
                                   np.ones(8)])
def test_invalid_gauss_sigma(sigma: Any) -> None:
    """
    Test that invalid ``GaussianPhiFunction`` ``sigma``argument values raise an
    error
    """
    with pytest.raises(ValueError):
        GaussianPhiFunction(3, sigma, X=X)


@pytest.mark.parametrize("ind", range(len(GAUSS_NEED_CALIB_SETTINGS)))
def test_gauss_need_calib(ind: int) -> None:
    """
    Test that ``GaussianPhiFunction`` arguments that require later completion
    have ``_need_x_calib`` = ``True`` 
    """
    phi = GaussianPhiFunction(**GAUSS_NEED_CALIB_SETTINGS[ind])
    assert phi._need_x_calib


@pytest.mark.parametrize("ind", range(len(GAUSS_NO_NEED_CALIB_SETTINGS)))
def test_gauss_no_need_calib(ind: int) -> None:
    """
    Test that ``GaussianPhiFunction`` arguments that don't require later
    completion have ``_need_x_calib`` = ``False`` 
    """
    phi = GaussianPhiFunction(**GAUSS_NO_NEED_CALIB_SETTINGS[ind])
    assert not phi._need_x_calib


@pytest.mark.parametrize("ind", range(len(GAUSS_NEED_CALIB_SETTINGS)))
def test_chained_check_need_calib(ind: int) -> None:
    """
    Test that a PhiFunction object _check_need_calib call the _check_need_calib
    method of children PhiFunction objects
    """
    child_phi = GaussianPhiFunction(**GAUSS_NEED_CALIB_SETTINGS[ind])
    assert child_phi._need_x_calib

    phi = PhiFunction([child_phi, lambda X: X, lambda X: np.ones(len(X))])
    assert not phi._need_x_calib

    phi._check_need_calib(X)
    assert not child_phi._need_x_calib
