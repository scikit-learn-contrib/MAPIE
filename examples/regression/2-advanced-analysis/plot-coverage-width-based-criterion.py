"""
===============================================================================
Focus on intervals width
===============================================================================


This example uses :class:`~mapie.regression.CrossConformalRegressor`,
:class:`~mapie.regression.ConformalizedQuantileRegressor` and
:class:`~mapie.regression.JackknifeAfterBootstrapRegressor`.
:class:`~mapie.metrics` is used to estimate the coverage width
based criterion of 1D homoscedastic data using different strategies.
The coverage width based criterion is computed with the function
:func:`~mapie.metrics.coverage_width_based`
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
    coverage_width_based,
)
from mapie.regression import (
    CrossConformalRegressor,
    ConformalizedQuantileRegressor,
    JackknifeAfterBootstrapRegressor,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


RANDOM_STATE = 1
##############################################################################
# Estimating the aleatoric uncertainty of heteroscedastic noisy data
# ---------------------------------------------------------------------
#
# Let's define again the ``x * sin(x)`` function and another simple
# function that generates one-dimensional data with normal noise uniformely
# in a given interval.


def x_sinx(x):
    """One-dimensional x*sin(x) function."""
    return x * np.sin(x)


def get_1d_data_with_heteroscedastic_noise(funct, min_x, max_x, n_samples, noise):
    """
    Generate 1D noisy data uniformely from the given function
    and standard deviation for the noise.
    """
    np.random.seed(59)
    X_train = np.linspace(min_x, max_x, n_samples)
    np.random.shuffle(X_train)
    X_test = np.linspace(min_x, max_x, n_samples * 5)
    y_train = funct(X_train) + (np.random.normal(0, noise, len(X_train)) * X_train)
    y_test = funct(X_test) + (np.random.normal(0, noise, len(X_test)) * X_test)
    y_mesh = funct(X_test)
    return (X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh)


##############################################################################
# We first generate noisy one-dimensional data uniformely on an interval.
# Here, the noise is considered as *heteroscedastic*, since it will increase
# linearly with `x`.

min_x, max_x, n_samples, noise = 0, 5, 300, 0.5
(X_train_conformalize, y_train_conformalize, X_test, y_test, y_mesh) = (
    get_1d_data_with_heteroscedastic_noise(x_sinx, min_x, max_x, n_samples, noise)
)

##############################################################################
# Let's visualize our noisy function. As x increases, the data becomes more
# noisy.

plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_train_conformalize, y_train_conformalize, color="C0")
plt.plot(X_test, y_mesh, color="C1")
plt.show()

##############################################################################
# As mentioned previously, we fit our training data with a simple
# polynomial function. Here, we choose a degree equal to 10 so the function
# is able to perfectly fit ``x * sin(x)``.

degree_polyn = 10
polyn_model = Pipeline(
    [("poly", PolynomialFeatures(degree=degree_polyn)), ("linear", LinearRegression())]
)
polyn_model_quant = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=degree_polyn)),
        (
            "linear",
            QuantileRegressor(
                solver="highs",
                alpha=0,
            ),
        ),
    ]
)

##############################################################################
# We then estimate the prediction intervals for all the strategies very easily
# with a `fit` and `predict` process. The prediction interval's lower and
# upper bounds are then saved in a DataFrame. Here, we set an alpha value of
# 0.05 in order to obtain a 95% confidence for our prediction intervals.

STRATEGIES = {
    "cv": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="base", cv=10),
    },
    "cv_plus": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="plus", cv=10),
    },
    "cv_minmax": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="minmax", cv=10),
    },
    "jackknife": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="base", cv=-1),
    },
    "jackknife_plus": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="plus", cv=-1),
    },
    "jackknife_minmax": {
        "class": CrossConformalRegressor,
        "init_params": dict(method="minmax", cv=-1),
    },
    "jackknife_plus_ab": {
        "class": JackknifeAfterBootstrapRegressor,
        "init_params": dict(method="plus", resampling=50),
    },
    "jackknife_minmax_ab": {
        "class": JackknifeAfterBootstrapRegressor,
        "init_params": dict(method="minmax", resampling=50),
    },
    "conformalized_quantile_regression": {
        "class": ConformalizedQuantileRegressor,
        "init_params": dict(),
    },
}


y_pred, y_pis = {}, {}
for strategy_name, strategy_params in STRATEGIES.items():
    init_params = strategy_params["init_params"]
    class_ = strategy_params["class"]
    if strategy_name == "conformalized_quantile_regression":
        X_train, X_conformalize, y_train, y_conformalize = train_test_split(
            X_train_conformalize,
            y_train_conformalize,
            test_size=0.3,
            random_state=RANDOM_STATE,
        )
        mapie = class_(polyn_model_quant, confidence_level=0.95, **init_params)
        mapie.fit(X_train, y_train)
        mapie.conformalize(X_conformalize, y_conformalize)
        y_pred[strategy_name], y_pis[strategy_name] = mapie.predict_interval(X_test)
    else:
        mapie = class_(
            polyn_model, confidence_level=0.95, random_state=RANDOM_STATE, **init_params
        )
        mapie.fit_conformalize(X_train_conformalize, y_train_conformalize)
        y_pred[strategy_name], y_pis[strategy_name] = mapie.predict_interval(X_test)

##############################################################################
# Once again, let’s compare the target confidence intervals with prediction
# intervals obtained with the Jackknife+, Jackknife-minmax, CV+, CV-minmax,
# Jackknife+-after-Boostrap, and CQR strategies.


def plot_1d_data(
    X_train,
    y_train,
    X_test,
    y_test,
    y_sigma,
    y_pred,
    y_pred_low,
    y_pred_up,
    ax=None,
    title=None,
):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.fill_between(X_test, y_pred_low, y_pred_up, alpha=0.3)
    ax.scatter(X_train, y_train, color="red", alpha=0.3, label="Training data")
    ax.plot(X_test, y_test, color="gray", label="True confidence intervals")
    ax.plot(X_test, y_test - y_sigma, color="gray", ls="--")
    ax.plot(X_test, y_test + y_sigma, color="gray", ls="--")
    ax.plot(X_test, y_pred, color="blue", alpha=0.5, label="Prediction intervals")
    if title is not None:
        ax.set_title(title)
    ax.legend()


strategies = [
    "jackknife_plus",
    "jackknife_minmax",
    "cv_plus",
    "cv_minmax",
    "jackknife_plus_ab",
    "conformalized_quantile_regression",
]
n_figs = len(strategies)
fig, axs = plt.subplots(3, 2, figsize=(9, 13))
coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]
for strategy, coord in zip(strategies, coords):
    plot_1d_data(
        X_train.ravel(),
        y_train.ravel(),
        X_test.ravel(),
        y_mesh.ravel(),
        (1.96 * noise * X_test).ravel(),
        y_pred[strategy].ravel(),
        y_pis[strategy][:, 0, 0].ravel(),
        y_pis[strategy][:, 1, 0].ravel(),
        ax=coord,
        title=strategy,
    )
plt.show()


##############################################################################
# Let’s now conclude by summarizing the *effective* coverage, namely the
# fraction of test
# points whose true values lie within the prediction intervals, given by
# the different strategies.

coverage_score = {}
width_mean_score = {}
cwc_score = {}

for strategy in STRATEGIES:
    coverage_score[strategy] = regression_coverage_score(y_test, y_pis[strategy])[0]
    width_mean_score[strategy] = regression_mean_width_score(y_pis[strategy])[0]
    cwc_score[strategy] = coverage_width_based(
        y_test,
        y_pis[strategy][:, 0, 0],
        y_pis[strategy][:, 1, 0],
        eta=0.001,
        confidence_level=0.95,
    )

results = pd.DataFrame(
    [
        [coverage_score[strategy], width_mean_score[strategy], cwc_score[strategy]]
        for strategy in STRATEGIES
    ],
    index=STRATEGIES,
    columns=["Coverage", "Width average", "Coverage Width-based Criterion"],
).round(2)

print(results)


##############################################################################
# All the strategies have the wanted coverage, however, we notice that the CQR
# strategy has much lower interval width than all the other methods, therefore,
# with heteroscedastic noise, CQR would be the preferred method.
