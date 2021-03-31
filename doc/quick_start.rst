#####################################
Quick Start with MAPIE
#####################################

This package allows you to easily estimate prediction intervals using your favourite sklearn-compatible regressor.

Estimate your prediction intervals
===================================================

1. Download and install the module
----------------------------------

(TBD) Install via `pip`:

.. code:: python

    pip install mapie


2. Run PredictionInterval
-------------------------

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