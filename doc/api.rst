#########
MAPIE API
#########

----

Regression
==========

Conformalizers
--------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.regression.SplitConformalRegressor
   mapie.regression.CrossConformalRegressor
   mapie.regression.JackknifeAfterBootstrapRegressor
   mapie.regression.ConformalizedQuantileRegressor
   mapie.regression.TimeSeriesRegressor

Metrics
-------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie.metrics.regression.regression_coverage_score
   mapie.metrics.regression.regression_mean_width_score
   mapie.metrics.regression.regression_ssc
   mapie.metrics.regression.regression_ssc_score
   mapie.metrics.regression.hsic
   mapie.metrics.regression.coverage_width_based
   mapie.metrics.regression.regression_mwi_score

Conformity Scores
-----------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.conformity_scores.BaseRegressionScore
   mapie.conformity_scores.AbsoluteConformityScore
   mapie.conformity_scores.GammaConformityScore
   mapie.conformity_scores.ResidualNormalisedScore

Resampling
----------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.subsample.BlockBootstrap
   mapie.subsample.Subsample

----

Classification
==============

Conformalizers
--------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.classification.SplitConformalClassifier
   mapie.classification.CrossConformalClassifier

Metrics
-------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie.metrics.classification.classification_coverage_score
   mapie.metrics.classification.classification_mean_width_score
   mapie.metrics.classification.classification_ssc
   mapie.metrics.classification.classification_ssc_score

Conformity Scores
-----------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.conformity_scores.BaseClassificationScore
   mapie.conformity_scores.NaiveConformityScore
   mapie.conformity_scores.LACConformityScore
   mapie.conformity_scores.APSConformityScore
   mapie.conformity_scores.RAPSConformityScore
   mapie.conformity_scores.TopKConformityScore

----

Risk Control
============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.risk_control.MultiLabelClassificationController
   mapie.risk_control.BinaryClassificationController
   mapie.risk_control.SemanticSegmentationController
   mapie.risk_control.BinaryClassificationRisk

----

Calibration
===========

Conformalizer
-------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.calibration.TopLabelCalibrator

Metrics
-------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie.metrics.calibration.expected_calibration_error
   mapie.metrics.calibration.top_label_ece
   mapie.metrics.calibration.cumulative_differences
   mapie.metrics.calibration.kolmogorov_smirnov_cdf
   mapie.metrics.calibration.kolmogorov_smirnov_p_value
   mapie.metrics.calibration.kolmogorov_smirnov_statistic
   mapie.metrics.calibration.kuiper_cdf
   mapie.metrics.calibration.kuiper_p_value
   mapie.metrics.calibration.kuiper_statistic
   mapie.metrics.calibration.length_scale
   mapie.metrics.calibration.spiegelhalter_p_value
   mapie.metrics.calibration.spiegelhalter_statistic

----

Utils
=====

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie.utils.train_conformalize_test_split
