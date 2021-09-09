######################
Quick Start with MAPIE
######################

This package allows you to easily estimate uncertainties in both regression and classification settings.
In regression settings, MAPIE provides prediction intervals on single-output data.
In classification settings, MAPIE provides prediction sets on multi-class data.
In any case, MAPIE is compatible with any scikit-learn-compatible estimator.

Estimate your prediction intervals
==================================

1. Download and install the module
----------------------------------

Install via ``pip``:

.. code:: python

    pip install mapie

or via `conda`:

.. code:: sh

    $ conda install -c conda-forge mapie

To install directly from the github repository :

.. code:: python

    pip install git+https://github.com/simai-ml/MAPIE


2. Run MapieRegressor
---------------------

Let us start with a basic regression problem. 
Here, we generate one-dimensional noisy data with normal distribution
that we fit with a linear model.

.. code:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression

    regressor = LinearRegression()
    X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)

Since MAPIE is compliant with the standard scikit-learn API, we follow the standard
sequential ``fit`` and ``predict`` process  like any scikit-learn regressor.
We set two values for alpha to estimate prediction intervals at approximately one
and two standard deviations from the mean.

.. code:: python

    from mapie.regression import MapieRegressor
    alpha = [0.05, 0.32]
    mapie = MapieRegressor(regressor)
    mapie.fit(X, y)
    y_pred, y_pis = mapie.predict(X, alpha=alpha)


3. Show the results
-------------------

MAPIE returns a ``np.ndarray`` of shape (n_samples, 3, len(alpha)) giving the predictions,
as well as the lower and upper bounds of the prediction intervals for the target quantile
for each desired alpha value.
The estimated prediction intervals can then be plotted as follows. 

.. code:: python
    
    from matplotlib import pyplot as plt
    from mapie.metrics import coverage_score

    coverage_scores = [
        coverage_score(y, y_pis[:, 0, i], y_pis[:, 1, i])
        for i, _ in enumerate(alpha)
    ]

    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X, y_preds[:, 0, 0], color="C1")
    order = np.argsort(X[:, 0])
    plt.plot(X[order], y_pis[order][:, 0, 1], color="C1", ls="--")
    plt.plot(X[order], y_pis[order][:, 1, 1], color="C1", ls="--")
    plt.fill_between(
        X[order].ravel(),
        y_pis[order][:, 0, 0].ravel(),
        y_pis[order][:, 1, 0].ravel(),
        alpha=0.2
    )
    plt.title(
        f"Target and effective coverages for "
        f"alpha={alpha[0]:.2f}: ({1-alpha[0]:.3f}, {coverage_scores[0]:.3f})\n"
        f"Target and effective coverages for "
        f"alpha={alpha[1]:.2f}: ({1-alpha[1]:.3f}, {coverage_scores[1]:.3f})"
    )
    plt.show()


.. image:: images/quickstart_1.png
    :width: 400
    :align: center

The title of the plot compares the target coverages with the effective coverages.
The target coverage, or the confidence interval, is the fraction of true labels lying in the
prediction intervals that we aim to obtain for a given dataset.
It is given by the alpha parameter defined in ``MapieRegressor``, here equal to ``0.05`` and ``0.32``,
thus giving target coverages of 0.95 and 0.68.
The effective coverage is the actual fraction of true labels lying in the prediction intervals.
