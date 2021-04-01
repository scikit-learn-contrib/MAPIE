#####################################
Quick Start with MAPIE
#####################################

This package allows you to easily estimate prediction intervals using your favourite sklearn-compatible regressor.

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


2. Run PredictionInterval
-------------------------

Before calling MAPIE, we first define a sklearn-compatible regressor as well as training and test sets.
MAPIE is compliant with the standard scikit-learn API.

.. code:: python

    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    regressor = LinearRegression()
    X, y = make_regression(n_samples=500, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


Like any scikit-learn regressor, MAPIE follows the following sequential ``fit`` and ``predict`` process. 

.. code:: python

    from mapie import PredictionInterval
    mapie = PredictionInterval(regressor)
    mapie.fit(X_train, y_train)
    y_preds = mapie.predict(X_test, y_test)

3. Show the results
-------------------

MAPIE returns a ``np.ndarray`` of shape (3, n_sample) giving the predictions,
as well as the lower and upper bounds of the prediction intervals for the target quantile.
The estimated prediction interval can be easily plotted as follows.

.. code:: python

    from matplotlib import pyplot as plt
    plt.plot(X_test, y_preds[0, :])
    plt.fill_between(X_test.ravel(), y_preds[1, :], y_preds[2, :])