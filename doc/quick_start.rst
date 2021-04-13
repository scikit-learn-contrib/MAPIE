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

    from mapie import MapieRegressor
    mapie = MapieRegressor(regressor)
    mapie.fit(X, y)
    X_pi = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_preds = mapie.predict(X_pi)


3. Show the results
-------------------

MAPIE returns a ``np.ndarray`` of shape (3, n_sample) giving the predictions,
as well as the lower and upper bounds of the prediction intervals for the target quantile.
The estimated prediction interval can be easily plotted as follows.

.. code:: python
    
    from matplotlib import pyplot as plt
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X_pi, y_preds[:, 0], color='C1')
    plt.fill_between(X_pi.ravel(), y_preds[:, 1], y_preds[:, 2], alpha=0.3)
    plt.show()


.. image:: images/quickstart_1.png
    :width: 400
    :align: center