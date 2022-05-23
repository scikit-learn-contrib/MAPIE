"""
==========================================================
Estimate the prediction intervals of 1D homoscedastic data
==========================================================

:class:`mapie.regression.MapieRegressor` is used to estimate
the prediction intervals of 1D homoscedastic data using
different strategies.
"""
from typing import Tuple, Union

from typing_extensions import TypedDict
import scipy
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mapie.metrics import regression_coverage_score


from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor

from mapie._typing import NDArray

data_n = 1.5


def f(x: NDArray) -> NDArray:
    """Polynomial function used to generate one-dimensional data"""
    # return np.array(5 + (3 * x))
    return np.array(5 * x + 5 * x ** 4 - 9 * x ** 2)


def get_homoscedastic_data(
    n_train: int = int(200*data_n),
    n_true: int = int(200*data_n),
    sigma: float = 0.1
) -> Tuple[NDArray, NDArray, NDArray, NDArray, float]:
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
    Tuple[NDArray, NDArray, NDArray, NDArray, float]
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
    y_train = np.array(
        [
            (f(x) + (np.random.normal(0, sigma))*x) for x in X_train
        ]
    )
    y_true = f(X_true)
    y_true_sigma = q95 * sigma
    return X_train, y_train, X_true, y_true, y_true_sigma


def plot_1d_data(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    y_test_sigma: float,
    y_pred: NDArray,
    y_pred_low: NDArray,
    y_pred_up: NDArray,
    ax: plt.Axes,
    title: str,
) -> None:
    """
    Generate a figure showing the training data and estimated
    prediction intervals on test data.

    Parameters
    ----------
    X_train : NDArray
        Training data.
    y_train : NDArray
        Training labels.
    X_test : NDArray
        Test data.
    y_test : NDArray
        True function values on test data.
    y_test_sigma : float
        True standard deviation.
    y_pred : NDArray
        Predictions on test data.
    y_pred_low : NDArray
        Predicted lower bounds on test data.
    y_pred_up : NDArray
        Predicted upper bounds on test data.
    ax : plt.Axes
        Axis to plot.
    title : str
        Title of the figure.
    """
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim([np.min(X_test), np.max(X_test)])
    ax.set_ylim([np.min(y_test), np.max(y_test)])
    ax.scatter(X_train, y_train, color="red", alpha=0.3, label="training")
    ax.plot(X_test, y_test, color="gray", label="True confidence intervals")
    ax.plot(X_test, y_test - y_test_sigma, color="gray", ls="--")
    ax.plot(X_test, y_test + y_test_sigma, color="gray", ls="--")
    ax.plot(X_test, y_pred, label="Prediction intervals")
    ax.fill_between(X_test, y_pred_low, y_pred_up, alpha=0.3)
    ax.set_title(title)
    ax.legend()


X_train, y_train, X_test, y_test, y_test_sigma = get_homoscedastic_data()

polyn_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=4)),
        ("linear", LinearRegression(fit_intercept=False)),
    ]
)

strategies = []
width = []
coverage = []

Params = TypedDict("Params", {"method": str, "cv": Union[int, str]})
STRATEGIES = {
    "jackknife": Params(method="base", cv=-1),
    "jackknife_plus": Params(method="plus", cv=-1),
    "jackknife_minmax": Params(method="minmax", cv=-1),
    # "cv": Params(method="base", cv=10),
    "cv_plus": Params(method="plus", cv=10),
    "cv_minmax": Params(method="minmax", cv=10),
    "quantile": Params(method="quantile", cv="split"),
}
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
    2, 3, figsize=(3 * 6, 12)
)
axs = [ax1, ax2, ax3, ax4, ax5, ax6]
for i, (strategy, params) in enumerate(STRATEGIES.items()):
    if strategy == "quantile":
        mapie = MapieQuantileRegressor(
            estimator=GradientBoostingRegressor(loss="quantile"),
            alpha=0.1,
            **params)
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train,
            y_train,
            test_size=0.5,
            random_state=1
        )
        mapie.fit(
            X_train.reshape(-1, 1),
            y_train,
            X_calib.reshape(-1, 1),
            y_calib
        )
        y_pred, y_pis = mapie.predict(
            X_test.reshape(-1, 1)
        )
    else:
        mapie = MapieRegressor(
            GradientBoostingRegressor(), n_jobs=-1, **params
        )
        mapie.fit(X_train.reshape(-1, 1), y_train)
        y_pred, y_pis = mapie.predict(
            X_test.reshape(-1, 1),
            alpha=0.05,
        )

    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    strategies.append(strategy)
    width.append((y_pred_up - y_pred_low).mean())
    coverage.append(regression_coverage_score(y_test, y_pred_low, y_pred_up))
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
        strategy,
    )


# data = pd.DataFrame(
#     list(zip(strategies, coverage, width)),
#     columns=['strategy', 'coverage', 'width']
# )
# print(data)

plt.show()
