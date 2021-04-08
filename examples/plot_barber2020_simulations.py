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
) -> Dict[Any, Any]:
    """
    Compute the prediction intervals estimated by a defined list of methods
    using a linear model on noisy linear data generated several times
    for several dimensions.
    Function adapted from Foygel-Barber et al. (2020).

    Parameters:
    -----------
    methods (List[str]):
        List of methods for estimating prediction intervals.
    alpha (float):
        1 - (target coverage level).
    n_trial (int):
        Number of trials for each dimension for estimating prediction intervals.
        For each trial, a new random noise is generated.
    dimensions (List[int]):
        List of dimension values of input data.

    Returns:
    --------
    Dict:
        Prediction interval widths and coverages for each method, trial,
        and dimension value.
    """
    n = 100
    n_test = 100
    SNR = 10

    results: Dict[int, List[Any]] = {}
    for dimension in dimensions:
        for i_trial in range(n_trial):
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
                preds_low, preds_up = y_preds[:, 1], y_preds[:, 2]
                preds[method] = {"lower": preds_low, "upper": preds_up}

            for method in methods:
                coverage = (
                    (preds[method]['lower'] <= y_test) & (preds[method]['upper'] >= y_test)
                ).mean()
                width = (preds[method]['upper'] - preds[method]['lower']).mean()
                results[len(results)] = [i_trial, dimension, method, coverage, width]
    return results


def plot_simulation_results(
    results: Dict[np.ndarray, np.ndarray],
    methods: List[str],
    title: str
) -> None:
    """
    Show the prediction interval coverages and widths as function of dimension values
    for selected methods with standard deviation given by different trials.

    Parameters:
    -----------
    results (Dict):
        Prediction interval widths and coverages for each method, trial,
        and dimension value.
    methods (List[str]):
        List of methods to show.
    title (str):
        Title of the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.rcParams.update({'font.size': 14})
    if title is not None:
        plt.suptitle(title)
    for method in methods:
        coverage_mean = []
        coverage_SE = []
        for dim_val in dimensions:
            coverage_mean.append(np.array([
                results[key][3] for key in results
                if (results[key][2] == method) & (results[key][1] == dim_val)
            ]).mean())
            coverage_SE.append(np.array([
                results[key][3] for key in results
                if (results[key][2] == method) & (results[key][1] == dim_val)
            ]).std()/np.sqrt(ntrial))
        coverage_mean_np = np.array(coverage_mean)
        coverage_SE_np = np.array(coverage_SE)
        ax1.plot(dimensions, coverage_mean_np, label=method)
        ax1.fill_between(
            dimensions,
            coverage_mean_np-coverage_SE_np,
            coverage_mean_np+coverage_SE_np,
            alpha=0.25
        )
    ax1.axhline(1-alpha, linestyle='dashed', c='k')
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel('Dimension d')
    ax1.set_ylabel('Coverage')
    ax1.legend()
    for method in methods:
        width_mean = []
        width_SE = []
        for dim_val in dimensions:
            width_mean.append(np.array([
                results[key][-1] for key in results
                if (results[key][2] == method) & (results[key][1] == dim_val)
            ]).mean())
            width_SE.append(np.array([
                results[key][-1] for key in results
                if (results[key][2] == method) & (results[key][1] == dim_val)
            ]).std()/np.sqrt(ntrial))
        width_mean_np = np.array(width_mean)
        width_SE_np = np.array(width_SE)
        ax2.plot(dimensions, width_mean_np, label=method)
        ax2.fill_between(
            dimensions, width_mean-width_SE_np, width_mean+width_SE_np, alpha=0.25
        )
    ax2.set_ylim(0, 20)
    ax2.set_xlabel('Dimension d')
    ax2.set_ylabel('Interval width')
    ax2.legend()


methods = [
    'naive', 'jackknife', 'jackknife_plus', 'jackknife_minmax', 'cv', 'cv_plus', 'cv_minmax'
]
alpha = 0.1
ntrial = 10
dimensions = np.arange(5, 205, 5)
results = PIs_vs_dimensions(methods, alpha, ntrial, dimensions)
plot_simulation_results(results, methods, title='Coverages and interval widths')
