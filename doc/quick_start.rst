#####################################
Quick Start with MAPIE
#####################################

This package allows you to easily estimate prediction intervals using your
favourite sklearn-compatible regressor on single-output data.

Estimate your prediction intervals
===================================================

1. Download and install the module
----------------------------------

Install via `pip`:

.. code:: python

    pip install mapie

To install directly from the github repository :

.. code:: python

    pip install git+https://github.com/simai-ml/MAPIE


2. Run MapieRegressor
---------------------

Let us start with a basic regression problem. 
Here, we generate one-dimensional noisy data that we fit with a linear model.

.. code:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression

    regressor = LinearRegression()
    X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)

Since MAPIE is compliant with the standard scikit-learn API, we follow the standard
sequential ``fit`` and ``predict`` process  like any scikit-learn regressor.

.. code:: python

    from mapie.estimators import MapieRegressor
    mapie = MapieRegressor(regressor, method="jackknife_plus")
    mapie.fit(X, y)
    y_preds = mapie.predict(X)


3. Show the results
-------------------

MAPIE returns a ``np.ndarray`` of shape (n_samples, 3) giving the predictions,
as well as the lower and upper bounds of the prediction intervals for the target quantile.
The estimated prediction intervals can then be plotted as follows. 

.. code:: python
    
    from matplotlib import pyplot as plt
    from mapie.metrics import coverage_score
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X, y_preds[:, 0], color='C1')
    order = np.argsort(X[:, 0])
    plt.fill_between(X[order].ravel(), y_preds[:, 1][order], y_preds[:, 2][order], alpha=0.3)
    plt.title(
        f"Target coverage = 0.9; Effective coverage = {coverage_score(y, y_preds[:, 1], y_preds[:, 2])}"
    )
    plt.show()


.. image:: doc/images/quickstart_1.png
    :width: 400
    :align: center

The title of the plot compares the target coverage with the effective coverage.
The target coverage, or the confidence interval, is the fraction of true labels lying in the
prediction intervals that we aim to obtain for a given dataset.
It is given by the alpha parameter defined in `MapieRegressor`, here equal to the default value of
0.1 thus giving a target coverage of 0.9.
The effective coverage is the actual fraction of true labels lying in the prediction intervals.
