"""
==========================================================
Estimate the prediction intervals of 1D homoscedastic data
==========================================================

:class:`mapie.MapieRegressor` is used to estimate
the prediction intervals of 1D homoscedastic data using
different methods.
"""
from typing import Tuple, Optional, Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

from mapie import MapieRegressor


def f(x: np.ndarray) -> np.ndarray:
    """Polynomial function used to generate one-dimensional data"""
    return 5*x + 5*x ** 4 - 9*x**2


def get_homoscedastic_data(
    n_samples: str = 200,
    n_test: int = 1000,
    sigma: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate one-dimensional data from a given function, number of training and test samples
    and a given standard deviation for the noise.

    Parameters
    ----------
        n_samples (str, optional): [description]. Defaults to 200.
        n_test (int, optional): [description]. Defaults to 1000.
        sigma (float, optional): [description]. Defaults to 0.1.

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]: [description]
    """
    np.random.seed(0)
    q90 = 1.8
    x_mesh = np.linspace(0.001, 1.2, 2000, endpoint=False)
    y_mesh = f(x_mesh)
    y_mesh_sig = q90*sigma
    x_train = np.random.exponential(0.4, n_samples)
    y_train = f(x_train) + np.random.normal(0, sigma, n_samples)
    x_test = np.random.exponential(0.4, n_test)
    y_test = f(x_test) + np.random.normal(0, sigma, n_test)
    return x_mesh, y_mesh, y_mesh_sig, x_train, y_train, x_test, y_test, sigma


def fit_and_predict(
    funct: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str,
    alpha: float,
    n_splits: int,
    return_pred: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pireg = MapieRegressor(
        funct,
        method=method,
        alpha=alpha,
        n_splits=n_splits,
        return_pred=return_pred
    )
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    pireg.fit(X_train, y_train)
    y_preds = pireg.predict(X_test)
    y_pred, y_pred_low, y_pred_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]
    return y_pred, y_pred_low, y_pred_up


def plot_1d_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_mesh: np.ndarray,
    y_mesh: np.ndarray,
    y_sigma: np.ndarray = None,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    y_pred: np.ndarray = None,
    y_pred_low: np.ndarray = None,
    y_pred_up: np.ndarray = None,
    ax: Optional[Any] = None,
    title: Optional[str] = None
) -> None:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(X_train, y_train, color='red', alpha=0.3, label='training')
    ax.set_ylim([y_mesh.min()*1.2, y_mesh.max()*1.2])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1.1])
    ax.plot(X_mesh, y_mesh, color='gray', label='True confidence intervals')
    if y_sigma is not None:
        ax.plot(X_mesh, y_mesh-y_sigma, color='gray', ls='--')
        ax.plot(X_mesh, y_mesh+y_sigma, color='gray', ls='--')
    if X_test is not None:
        order = np.argsort(X_test)
    if y_test is not None:
        ax.scatter(X_test[order], y_test[order], color='blue', alpha=0.3, label='test')
    if y_pred is not None:
        order = np.argsort(X_test)
        ax.plot(X_test[order], y_pred[order], label='Prediction intervals')
    if y_pred_low is not None and y_pred_up is not None:
        ax.fill_between(X_test[order], y_pred_low[order], y_pred_up[order], alpha=0.3)
    if title is not None:
        ax.set_title(title)
    ax.legend()


(
    X_mesh,
    y_mesh,
    y_mesh_sig,
    X_train,
    y_train,
    X_test,
    y_test,
    sigma
) = get_homoscedastic_data(n_samples=200, n_test=200, sigma=0.1)

polyn_model = Pipeline(
    [
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression(fit_intercept=False))
    ]
)

methods = ['jackknife', 'jackknife_plus', 'jackknife_minmax', 'cv', 'cv_plus', 'cv_minmax']
preds, lows, ups = [], [], []
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(3*6, 12))
axs = [ax1, ax2, ax3, ax4, ax5, ax6]
for i, method in enumerate(methods):
    y_pred, y_pred_low, y_pred_up = fit_and_predict(
        polyn_model, X_train, y_train, X_test, y_test,
        method=method, alpha=0.1, n_splits=10, return_pred='ensemble'
    )
    preds.append(y_pred), lows.append(y_pred_low), preds.append(y_pred_up)
    plot_1d_data(
        X_train,
        y_train,
        X_mesh,
        y_mesh,
        y_mesh_sig,
        X_test=X_test,
        y_test=None,
        y_pred=y_pred,
        y_pred_low=y_pred_low,
        y_pred_up=y_pred_up,
        ax=axs[i], title=method
    )
