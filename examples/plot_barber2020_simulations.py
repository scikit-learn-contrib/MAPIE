"""
============================================================
Reproducing the simulations from Foygel-Barber et al. (2020)
============================================================

:class:`mapie.MapieRegressor` is used to investigate
the coverage level and the prediction interval width as function
of the dimension using simulated data points as introduced in
Foygel-Barber et al. (2020).

This simulation generates several times linear data with random noise
whose signal-to-noise is equal to 10 and for several given dimensions.

Here we use MAPIE, with a LinearRegression base model, to estimate the width
means and the coverage levels of the prediction intervals estimated by all the
available methods as function of the dataset dimension.

We then show the prediction interval coverages and widths as function of the
dimension values for selected methods with standard error given by the different trials.

This simulation is carried out to emphasize the instability of the prediction
intervals estimated by the Jackknife method when the dataset dimension is
equal to the number of training samples (here 100).
"""
from typing import List, Dict

import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from mapie import MapieRegressor


def PIs_vs_dimensions(
    methods: List[str],
    alpha: float,
    n_trial: int,
    dimensions: List[int]
) -> Dict[int, Dict[int, Dict[str, Dict[str, float]]]]:
    """
    Compute the prediction intervals for a linear regression problem.
    Function adapted from Foygel-Barber et al. (2020).

    It generates several times linear data with random noise whose signal-to-noise
    is equal to 10 and for several given dimensions, given by the dimensions list.

    Here we use MAPIE, with a LinearRegression base model, to estimate the width
    means and the coverage levels of the prediction intervals estimated by all the
    available methods as function of the dataset dimension.

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
    Dict[int, Dict[int, Dict[str, Dict[str, float]]]]
        Prediction interval widths and coverages for each method, trial,
        and dimension value.
    """
    n_train = 100
    n_test = 100
    SNR = 10
    results: Dict[int, Dict[int, Dict[str, Dict[str, float]]]] = {}
    for dimension in dimensions:
        results_trial: Dict[int, Dict[str, Dict[str, float]]] = {}
        for trial in range(n_trial):
            beta = np.random.normal(size=dimension)
            beta_norm = np.sqrt((beta**2).sum())
            beta = beta/beta_norm*np.sqrt(SNR)
            X_train = np.random.normal(size=(n_train, dimension))
            noise_train = np.random.normal(size=n_train)
            noise_test = np.random.normal(size=n_test)
            y_train = X_train.dot(beta) + noise_train
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

            results_method: Dict[str, Dict[str, float]] = {}
            for method, pred in preds.items():
                coverage = (
                    (pred["lower"] <= y_test) &
                    (pred["upper"] >= y_test)
                ).mean()
                width_mean = (pred["upper"] - pred["lower"]).mean()
                results_method[method] = {
                    "coverage": coverage,
                    "width_mean": width_mean
                }
            results_trial[trial] = results_method
        results[dimension] = results_trial
    return results


def plot_simulation_results(
    results: Dict[int, Dict[int, Dict[str, Dict[str, float]]]],
    methods: List[str],
    title: str
) -> None:
    """
    Show the prediction interval coverages and widths as function of dimension values
    for selected methods with standard error given by different trials.

    Parameters
    ----------
    results : Dict[str, Dict[int, Dict[str, Dict[str, float]]]]
        Prediction interval widths and coverages for each method, trial,
        and dimension value.
    methods : List[str]
        List of methods to show.
    title : str
        Title of the plot.
    """
    dimensions = list(results.keys())
    trials = list(results[dimensions[0]].keys())
    methods = list(results[dimensions[0]][trials[0]].keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.rcParams.update({"font.size": 14})
    if title is not None:
        plt.suptitle(title)
        for method in methods:
            coverage_mean = np.zeros([len(dimensions)])
            coverage_SE = np.zeros([len(dimensions)])
            width_mean = np.zeros([len(dimensions)])
            width_SE = np.zeros([len(dimensions)])
            for idim, dimension in enumerate(dimensions):
                coverage_mean[idim] = np.stack(
                    [results[dimension][trial][method]["coverage"] for trial in trials]
                ).mean()
                coverage_SE[idim] = np.stack(
                    [results[dimension][trial][method]["coverage"] for trial in trials]
                ).std()/np.sqrt(ntrial)
                width_mean[idim] = np.stack(
                    [results[dimension][trial][method]["width_mean"] for trial in trials]
                ).mean()
                width_SE[idim] = np.stack(
                    [results[dimension][trial][method]["width_mean"] for trial in trials]
                ).std()/np.sqrt(ntrial)
            ax1.plot(dimensions, coverage_mean, label=method)
            ax1.fill_between(
                dimensions,
                coverage_mean - coverage_SE,
                coverage_mean + coverage_SE,
                alpha=0.25
            )
            ax2.plot(dimensions, width_mean, label=method)
            ax2.fill_between(
                dimensions,
                width_mean - width_SE,
                width_mean + width_SE,
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
