"""
==========================================================
Estimate the prediction intervals of 1D homoscedastic data
==========================================================

:class:`mapie.regression.MapieRegressor` is used to estimate
the prediction intervals of 1D homoscedastic data using
different strategies.
"""
from typing import Tuple
from typing_extensions import TypedDict

import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from mapie.regression import MapieRegressor


def f(x: np.ndarray) -> np.ndarray:
    """Polynomial function used to generate one-dimensional data"""
    return np.array(5*x + 5*x**4 - 9*x**2)


def get_homoscedastic_data(
    n_train: int = 200,
    n_true: int = 200,
    sigma: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate one-dimensional data from a given function,
    number of training and test samples and a given standard
    deviation for the noise.
    The training data data is generated from an exponential distribution.

    Parameters
    ----------
    n_train : int, optional
        Number of training samples, by default  200.
    n_true : int, optional
        Number of test samples, by default 1000.
    sigma : float, optional
        Standard deviation of noise, by default 0.1

    Returns
    -------
    Tuple[Any, Any, np.ndarray, Any, float]
        Generated training and test data.
        [0]: X_train
        [1]: y_train
        [2]: X_true
        [3]: y_true
        [4]: y_true_sigma
    """
    np.random.seed(59)
    q95 = scipy.stats.norm.ppf(0.95)
    X_train = np.linspace(0, 1, n_train)
    X_true = np.linspace(0, 1, n_true)
    y_train = f(X_train) + np.random.normal(0, sigma, n_train)
    y_true = f(X_true)
    y_true_sigma = q95*sigma
    return X_train, y_train, X_true, y_true, y_true_sigma


def plot_1d_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_sigma: float,
    y_pred: np.ndarray,
    y_pred_low: np.ndarray,
    y_pred_up: np.ndarray,
    ax: plt.Axes,
    title: str
) -> None:
    """
    Generate a figure showing the training data and estimated
    prediction intervals on test data.

    Parameters
    ----------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data.
    y_test : np.ndarray
        True function values on test data.
    y_test_sigma : float
        True standard deviation.
    y_pred : np.ndarray
        Predictions on test data.
    y_pred_low : np.ndarray
        Predicted lower bounds on test data.
    y_pred_up : np.ndarray
        Predicted upper bounds on test data.
    ax : plt.Axes
        Axis to plot.
    title : str
        Title of the figure.
    """
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.scatter(X_train, y_train, color="red", alpha=0.3, label="training")
    ax.plot(X_test, y_test, color="gray", label="True confidence intervals")
    ax.plot(X_test, y_test - y_test_sigma, color="gray", ls="--")
    ax.plot(X_test, y_test + y_test_sigma, color="gray", ls="--")
    ax.plot(X_test, y_pred, label="Prediction intervals")
    ax.fill_between(X_test, y_pred_low, y_pred_up, alpha=0.3)
    ax.set_title(title)
    ax.legend()


X_train, y_train, X_test, y_test, y_test_sigma = get_homoscedastic_data()

polyn_model = Pipeline([
    ("poly", PolynomialFeatures(degree=4)),
    ("linear", LinearRegression(fit_intercept=False))
])

Params = TypedDict("Params", {"method": str, "cv": int})
STRATEGIES = {
    "jackknife": Params(method="base", cv=-1),
    "jackknife_plus": Params(method="plus", cv=-1),
    "jackknife_minmax": Params(method="minmax", cv=-1),
    "cv": Params(method="base", cv=10),
    "cv_plus": Params(method="plus", cv=10),
    "cv_minmax": Params(method="minmax", cv=10),
}
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(3*6, 12))
axs = [ax1, ax2, ax3, ax4, ax5, ax6]
for i, (strategy, params) in enumerate(STRATEGIES.items()):
    mapie = MapieRegressor(
        polyn_model,
        ensemble=True,
        n_jobs=-1,
        **params
    )
    mapie.fit(X_train.reshape(-1, 1), y_train)
    y_pred, y_pis = mapie.predict(X_test.reshape(-1, 1), alpha=0.05,)
    plot_1d_data(
        X_train,
        y_train,
        X_test,
        y_test,
        y_test_sigma,
        y_pred,
        y_pis[:, 0, 0],
        y_pis[:, 1, 0],
        axs[i],
        strategy
    )
plt.show()
