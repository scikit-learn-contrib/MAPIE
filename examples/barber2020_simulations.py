"""
============================================================
Reproducing the simulations from Foygel-Barber et al. (2020)
============================================================

class:`predinterv.PredictionInterval` is used to investigate
the coverage level and prediction interval width as function
of the dimension using simulated data points.
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from predinterv import PredictionInterval

def PIs_vs_dim_vals(method_names, alpha, ntrial, dim_vals):
    n = 100
    n1 = 100
    SNR = 10

    columns = ['itrial', 'd', 'method', 'coverage', 'width']
    results = pd.DataFrame(columns=columns)
    results_2 = pd.DataFrame(columns=columns)
    for d in dim_vals:
        print(f'{d}/{dim_vals[-1]}', end='\r')
        for itrial in range(ntrial):
            beta = np.random.normal(size=d)
            beta = beta/np.sqrt((beta**2).sum()) * np.sqrt(SNR)
            X = np.random.normal(size=(n, d))
            Y = X.dot(beta) + np.random.normal(size=n)
            min_Y = Y.min() - 0.1 * (Y.max()-Y.min())
            max_Y = Y.max() + 0.1 * (Y.max()-Y.min())
            X1 = np.random.normal(size=(n1, d))
            Y1 = X1.dot(beta) + np.random.normal(size=n1)

            preds_df = {}
            for im, method in enumerate(method_names):
                predinterv = PredictionInterval(LinearRegression(), alpha=alpha, method=method, n_splits=10, shuffle=False, return_pred="ensemble")
                predinterv.fit(X, Y)
                y_preds = predinterv.predict(X1)
                y_pred, preds_low, preds_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]
                preds_df[method] = pd.DataFrame(np.stack([preds_low, preds_up], axis=1), columns=['lower', 'upper'])
            preds_df = pd.concat(preds_df, axis=1)

            for method in method_names:
                coverage = ((preds_df[method]['lower'] <= Y1) & (preds_df[method]['upper'] >= Y1)).mean()
                width = (preds_df[method]['upper'] - preds_df[method]['lower']).mean()
                results.loc[len(results)] = [itrial, d, method, coverage, width]
    return results

def plot_simulation_results(results, method_names, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plt.rcParams.update({'font.size': 14})
    if title is not None:
        plt.suptitle(title)
    for method in method_names:
        coverage_mean = results[results['method']==method].groupby('d')['coverage'].mean()
        coverage_SE = results[results['method']==method].groupby('d')['coverage'].std()/np.sqrt(ntrial)
        ax1.plot(dim_vals,coverage_mean,label=method)
        ax1.fill_between(dim_vals,coverage_mean-coverage_SE,coverage_mean+coverage_SE,alpha = 0.25)
    ax1.axhline(1-alpha,linestyle='dashed',c='k')
    ax1.set_ylim(0.,1.0)
    ax1.set_xlabel('Dimension d')
    ax1.set_ylabel('Coverage')
    ax1.legend()
    for method in method_names:
        width_mean = results[results['method']==method].groupby('d')['width'].mean()
        width_SE = results[results['method']==method].groupby('d')['width'].std()/np.sqrt(ntrial)
        ax2.plot(dim_vals,width_mean,label=method)
        ax2.fill_between(dim_vals,width_mean-width_SE,width_mean+width_SE,alpha = 0.25)
    ax2.set_ylim(0,results['width'].max()*1.2)
    ax2.set_xlabel('Dimension d')
    ax2.set_ylabel('Interval width')
    ax2.legend()

methods = ['naive', 'jackknife', 'jackknife_plus', 'jackknife_minmax' , 'cv', 'cv_plus', 'cv_minmax']
alpha = 0.1
ntrial = 1
dim_vals = np.arange(5, 205, 5)
results = PIs_vs_dim_vals(methods, alpha, ntrial, dim_vals)
plot_simulation_results(results, methods, title='Coverages and interval widths.')