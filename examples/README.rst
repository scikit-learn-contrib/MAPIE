.. _general_examples:

General examples
================

Plotting MAPIE prediction intervals with a toy dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An example plot of MapieRegressor used in the Quickstart.

Estimate the prediction intervals of 1D homoscedastic data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MapieRegressor is used to estimate the prediction intervals of 1D homoscedastic data.

Reproducing the simulations from Foygel-Barber et al. (2020)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`mapie.estimators.MapieRegressor` is used to investigate
the coverage level and the prediction interval width as function
of the dimension using simulated data points as introduced in
Foygel-Barber et al. (2020).

Nested cross-validation for estimating prediction intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example compares non-nested and nested cross-validation strategies for
estimating prediction intervals with MapieRegressor.