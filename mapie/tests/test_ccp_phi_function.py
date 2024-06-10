from __future__ import annotations

from typing import Any, List, Dict

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.datasets import make_regression
from mapie.phi_function import (CustomPhiFunction, GaussianPhiFunction,
                                PolynomialPhiFunction, PhiFunction)

random_state = 1
np.random.seed(random_state)

X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
z = X[:, -2:]

PHI = [
    CustomPhiFunction([lambda X: np.ones((len(X), 1))]),
    CustomPhiFunction(None, bias=True),
    CustomPhiFunction([lambda X: X]),
    CustomPhiFunction([lambda X: X, lambda z: z]),
    CustomPhiFunction([lambda X: X, lambda y_pred: y_pred]),
    PolynomialPhiFunction(2, "X", bias=True),
    PolynomialPhiFunction([1, 2], "X", bias=True),
    PolynomialPhiFunction([1, 4, 5], "y_pred", bias=False),
    PolynomialPhiFunction([0, 1, 4, 5], "y_pred", bias=False),
    PolynomialPhiFunction([0, 1, 3], "z", bias=False),
    GaussianPhiFunction(4),
    CustomPhiFunction([lambda X: X, PolynomialPhiFunction(2)]),
    CustomPhiFunction([lambda X: X, GaussianPhiFunction(2)]),
    CustomPhiFunction([
        lambda X: X, PolynomialPhiFunction([1, 2], bias=False)
    ]),
]

# n_out without bias
N_OUT = [1, 1, 10, 12, 11, 21, 21, 3, 4, 5, 4, 31, 12, 30]

GAUSS_NEED_FIT_SETTINGS: List[Dict[str, Any]] = [
    {
        "points": 10,
        "sigma": 1,
    },
    {
        "points": 10,
        "sigma": None,
    },
    {
        "points": 10,
        "sigma": None,
        "random_sigma": True,
    },
    {
        "points": 10,
        "sigma": None,
        "random_sigma": False,
    },
    {
        "points": np.ones((2, X.shape[1])),
        "sigma": None,
    },
]

GAUSS_NO_NEED_FIT_SETTINGS: List[Dict[str, Any]] = [
    {
        "points": np.ones((2, X.shape[1])),
        "sigma": np.ones(X.shape[1]),
    },
    {
        "points": (np.ones((2, X.shape[1])), [1, 2]),
        "sigma": None,
    },
    {
        "points": (np.ones((2, X.shape[1])), np.ones((2, X.shape[1]))),
        "sigma": None,
    },
]


# ======== CustomPhiFunction =========
@pytest.mark.parametrize("functions", [
    lambda X: X, [lambda X: X],
    [lambda X: X, PolynomialPhiFunction(2)],
    [lambda X: X, PolynomialPhiFunction(2), GaussianPhiFunction(2)],
])
def test_custom_phi_functions(functions: Any) -> None:
    """Test that initialization does not crash."""
    phi = CustomPhiFunction(functions)
    phi.fit(X)
    phi.transform(X)


@pytest.mark.parametrize("phi, n_out_raw", zip(PHI, N_OUT))
def test_phi_n_attributes(phi: PhiFunction, n_out_raw: int) -> None:
    """
    Test that the n_in and n_out attributes are corrects
    """
    phi.fit(X, y_pred=y, z=z)
    phi.transform(X, y_pred=y, z=z)
    assert phi.n_in == 10
    assert phi.n_out == n_out_raw


def test_phi_functions_warning() -> None:
    """
    Test that creating a PhiFunction object with functions which have
    optional arguments different from 'X', 'y_pred' or 'z' raise a warning.
    """
    with pytest.warns(UserWarning,
                      match="WARNING: Unknown optional arguments."):
        phi = CustomPhiFunction([lambda X, d=d: X**d for d in range(4)])
        phi.fit(X)
        phi.transform(X)


@pytest.mark.parametrize("functions", [
    [lambda X, other: X + other, lambda X, other: X - other],
    [lambda X, other: X + other]
])
def test_phi_functions_error(functions: Any) -> None:
    """
    Test that creating a PhiFunction object with functions which have
    required arguments different from 'X', 'y_pred' or 'z' raise an error.
    """
    for f in functions:     # For coverage
        f(np.ones((10, 1)), np.ones((10, 1)))
    with pytest.raises(ValueError, match=r"Forbidden required argument."):
        phi = CustomPhiFunction(functions)
        phi.fit(X)


def test_phi_functions_empty() -> None:
    """
    Test that creating a PhiFunction object with functions which have
    required arguments different from 'X', 'y_pred' or 'z' raise an error.
    """
    with pytest.raises(ValueError):
        phi = CustomPhiFunction([], bias=False)
        phi.fit(X)


# ======== PolynomialPhiFunction =========
def test_poly_phi_init() -> None:
    """Test that initialization does not crash."""
    phi = PolynomialPhiFunction()
    phi.fit(X)


@pytest.mark.parametrize("degree", [2, [0, 1, 3]])
@pytest.mark.parametrize("variable", ["X", "y_pred", "z"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
def test_poly_phi_init_other(
    degree: Any, variable: Any, bias: bool, normalized: bool
) -> None:
    """Test that initialization does not crash."""
    PolynomialPhiFunction(degree, variable, bias, normalized)


@pytest.mark.parametrize("var", ["other", 1, np.ones((10, 1))])
def test_invalid_variable_value(var: Any) -> None:
    """
    Test that invalid variable value raise error
    """
    with pytest.raises(ValueError):
        phi = PolynomialPhiFunction(variable=var)
        phi.fit(X)


# ======== GaussianPhiFunction =========
def test_gauss_phi_init() -> None:
    """Test that initialization does not crash."""
    GaussianPhiFunction()


@pytest.mark.parametrize("points", [3, [[10, 20], [2, 39], [2, 3]],
                                    ([[1], [2], [3]], [1, 2, 3])])
@pytest.mark.parametrize("sigma", [None, 1, [1, 2]])
@pytest.mark.parametrize("random_sigma", [True, False])
@pytest.mark.parametrize("X", [None, np.ones((30, 2))])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("normalized", [True, False])
def test_poly_gauss_init_other(
    points: Any, sigma: Any, random_sigma: Any, X: Any,
    bias: bool, normalized: bool
) -> None:
    """Test that initialization does not crash."""
    GaussianPhiFunction(points, sigma, random_sigma,
                        bias, normalized)


@pytest.mark.parametrize("points", [np.ones((10)),
                                    np.ones((10, 2, 2))])
def test_invalid_gauss_points(points: Any) -> None:
    """
    Test that invalid ``GaussianPhiFunction`` ``points``argument values raise
    an error
    """
    with pytest.raises(ValueError, match="Invalid `points` argument."):
        phi = GaussianPhiFunction(points)
        phi.fit(X)


def test_invalid_gauss_points_2() -> None:
    """
    Test that invalid ``GaussianPhiFunction`` ``points``argument values raise
    an error
    """
    with pytest.raises(ValueError, match="There should have as many points"):
        phi = GaussianPhiFunction(points=(np.ones((10, 3)), np.ones((8, 3))))
        phi.fit(X)


def test_invalid_gauss_points_3() -> None:
    """
    Test that invalid ``GaussianPhiFunction`` ``points``argument values raise
    an error
    """
    with pytest.raises(ValueError, match="The standard deviation 2D array"):
        phi = GaussianPhiFunction(points=(np.ones((10, 3)), np.ones((10, 2))))
        phi.fit(X)


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
        phi = GaussianPhiFunction(3, sigma)
        phi.fit(X)
        phi.transform(X)


@pytest.mark.parametrize("ind", range(len(GAUSS_NEED_FIT_SETTINGS)))
def test_gauss_need_calib(ind: int) -> None:
    """
    Test that ``GaussianPhiFunction`` arguments that require later completion
    have ``_need_x_calib`` = ``True``
    """
    phi = GaussianPhiFunction(**GAUSS_NEED_FIT_SETTINGS[ind])
    phi.fit(X)
    check_is_fitted(phi, phi.fit_attributes)


@pytest.mark.parametrize("ind", range(len(GAUSS_NO_NEED_FIT_SETTINGS)))
def test_gauss_no_need_calib(ind: int) -> None:
    """
    Test that ``GaussianPhiFunction`` arguments that don't require later
    completion have ``_need_x_calib`` = ``False``
    """
    phi = GaussianPhiFunction(**GAUSS_NO_NEED_FIT_SETTINGS[ind])
    phi.fit(X)
    check_is_fitted(phi, phi.fit_attributes)
