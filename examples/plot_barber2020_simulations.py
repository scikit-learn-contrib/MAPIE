"""
============================================================
Reproducing the simulations from Foygel-Barber et al. (2020)
============================================================

:class:`mapie.MapieRegressor` is used to investigate
the coverage level and the prediction interval width as function
of the dimension using simulated data points as introduced in
Foygel-Barber et al. (2020).
"""
from typing import List, Dict, Any

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from mapie import MapieRegressor


def PIs_vs_dimensions(
    methods: List[str],
    alpha: float,
    n_trial: int,
    dimensions: List[int]
) -> Dict[int, Dict[str, Any]]:
    """
    Compute the prediction intervals for a linear regression problem.
    Function adapted from Foygel-Barber et al. (2020).

    It generates several times linear data with random noise whose signal-to-noise
    is equal to 10 and for several given dimensions, given by the dimensions list.

    MAPIE is used to estimate the width means and the coverage levels of the
    prediction intervals estimated by all the available methods as function of
    the dataset dimension. provided by MAPIE using a LinearRegression model.

    This simulation is carried out to emphasize the instability of the prediction
    intervals estimated by the Jackknife method when the dataset dimension is
    equal to the number of training samples (here 100).

    Parameters
    ----------
    methods : List[str]
        List of methods for estimating prediction intervals.
    alpha : float
        1 - (target coverage level).
    n_trial : int
        Number of trials for each dimension for estimating prediction intervals.
        For each trial, a new random noise is generated.
    dimensions : List[int]
        List of dimension values of input data.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Prediction interval widths and coverages for each method, trial,
        and dimension value.
    """
    n = 100
    n_test = 100
    SNR = 10

    results: Dict[int, Dict[str, Any]] = {}
    for dimension in dimensions:
        for trial in range(n_trial):
            beta = np.random.normal(size=dimension)
            beta_norm = np.sqrt((beta**2).sum())
            beta = beta/beta_norm * np.sqrt(SNR)
            X_train = np.random.normal(size=(n, dimension))
            noise = np.random.normal(size=n)
            noise_test = np.random.normal(size=n_test)
            y_train = X_train.dot(beta) + noise
            X_test = np.random.normal(size=(n_test, dimension))
            y_test = X_test.dot(beta) + noise_test

            preds: Dict[str, Dict[str, np.ndarray]] = {}
            for method in methods:
                mapie = MapieRegressor(
                    LinearRegression(),
                    alpha=alpha,
                    method=method,
                    n_splits=10,
                    shuffle=False,
                    return_pred="ensemble"
                )
                mapie.fit(X_train, y_train)
                y_preds = mapie.predict(X_test)
                y_pred_low, y_pred_up = y_preds[:, 1], y_preds[:, 2]
                preds[method] = {"lower": y_pred_low, "upper": y_pred_up}

            for method in methods:
                coverage = (
                    (preds[method]["lower"] <= y_test) &
                    (preds[method]["upper"] >= y_test)
                ).mean()
                width_mean = (
                    preds[method]["upper"] - preds[method]["lower"]
                ).mean()
                results[len(results)] = {
                    "trial": trial,
                    "dimension": dimension,
                    "method": method,
                    "coverage": coverage,
                    "width_mean": width_mean
                }
    return results


def plot_simulation_results(
    results: Dict[int, Dict[str, Any]],
    methods: List[str],
    title: str
) -> None:
    """
    Show the prediction interval coverages and widths as function of dimension values
    for selected methods with standard deviation given by different trials.

    Parameters
    ----------
    results : Dict[int, Dict[str, Any]]
        Prediction interval widths and coverages for each method, trial,
        and dimension value.
    methods : List[str]
        List of methods to show.
    title : str
        Title of the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.rcParams.update({"font.size": 14})
    if title is not None:
        plt.suptitle(title)
    for method in methods:
        coverage_mean, coverage_SE = [], []
        width_mean, width_SE = [], []
        for dimension in dimensions:
            coverage_mean.append(np.array([
                results[key]["coverage"] for key in results
                if (
                    (results[key]["method"] == method) &
                    (results[key]["dimension"] == dimension)
                )
            ]).mean())
            coverage_SE.append(np.array([
                results[key]["coverage"] for key in results
                if (
                    (results[key]["method"] == method) &
                    (results[key]["dimension"] == dimension)
                )
            ]).std(ddof=np.sqrt(ntrial)))
            width_mean.append(np.array([
                results[key]["width_mean"] for key in results
                if (
                    (results[key]["method"] == method) &
                    (results[key]["dimension"] == dimension)
                )
            ]).mean())
            width_SE.append(np.array([
                results[key]["width_mean"] for key in results
                if (
                    (results[key]["method"] == method) &
                    (results[key]["dimension"] == dimension)
                )
            ]).std(ddof=np.sqrt(ntrial)))
        coverage_mean_np = np.stack(coverage_mean)
        coverage_SE_np = np.stack(coverage_SE)
        width_mean_np = np.stack(width_mean)
        width_SE_np = np.stack(width_SE)
        ax1.plot(dimensions, coverage_mean_np, label=method)
        ax1.fill_between(
            dimensions,
            coverage_mean_np - coverage_SE_np,
            coverage_mean_np + coverage_SE_np,
            alpha=0.25
        )
        ax2.plot(dimensions, width_mean_np, label=method)
        ax2.fill_between(
            dimensions,
            width_mean - width_SE_np,
            width_mean + width_SE_np,
            alpha=0.25
        )
    ax1.axhline(1-alpha, linestyle="dashed", c="k")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Dimension d")
    ax1.set_ylabel("Coverage")
    ax1.legend()
    ax2.set_ylim(0, 20)
    ax2.set_xlabel("Dimension d")
    ax2.set_ylabel("Interval width")
    ax2.legend()


methods = [
    "naive", "jackknife", "jackknife_plus", "jackknife_minmax", "cv", "cv_plus", "cv_minmax"
]
alpha = 0.1
ntrial = 10
dimensions = np.arange(5, 205, 5)
results = PIs_vs_dimensions(methods, alpha, ntrial, dimensions)
plot_simulation_results(results, methods, title="Coverages and interval widths")
