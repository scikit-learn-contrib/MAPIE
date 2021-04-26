.. _general_examples:

General examples
================

plot_toy_model
^^^^^^^^^^^^^^
An example plot of :class:`mapie.estimators.MapieRegressor` used
in the Quickstart.

plot_homoscedastic_1d_data
^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`mapie.estimators.MapieRegressor` is used to estimate
the prediction intervals of 1D homoscedastic data using
different methods.

plot_barber2020_simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
:class:`mapie.estimators.MapieRegressor` is used to investigate
the coverage level and the prediction interval width as function
of the dimension using simulated data points as introduced in
Foygel-Barber et al. (2020).

plot_nested-cv
^^^^^^^^^^^^^^
This example compares non-nested and nested cross-validation strategies for
estimating prediction intervals with :class:`mapie.estimators.MapieRegressor`.