#########
MAPIE API
#########

.. currentmodule:: mapie

Regression
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   regression.MapieRegressor
   quantile_regression.MapieQuantileRegressor
   time_series_regression.MapieTimeSeriesRegressor

Classification
==============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   classification.MapieClassifier

Metrics
=======

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   metrics.regression_coverage_score
   metrics.classification_coverage_score
   metrics.regression_mean_width_score
   metrics.classification_mean_width_score
   metrics.expected_calibration_error
   metrics.top_label_ece

Conformity scores
=================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   conformity_scores.AbsoluteConformityScore
   conformity_scores.GammaConformityScore


Resampling
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   subsample.BlockBootstrap
   subsample.Subsample
