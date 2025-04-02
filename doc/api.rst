#########
MAPIE API
#########

Regression V1 (from mapie_v1)
=============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie_v1.regression.SplitConformalRegressor
   mapie_v1.regression.CrossConformalRegressor
   mapie_v1.regression.JackknifeAfterBootstrapRegressor
   mapie_v1.regression.ConformalizedQuantileRegressor

Regression (from mapie)
=======================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.regression.MapieRegressor
   mapie.regression.MapieQuantileRegressor
   mapie.regression.MapieTimeSeriesRegressor

Classification
==============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.classification.MapieClassifier

Multi-Label Classification
==========================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.multi_label_classification.MapieMultiLabelClassifier

Calibration
===========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.calibration.MapieCalibrator

Calibration Metrics
======================================

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

Classification Metrics
========================================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie.metrics.classification.classification_coverage_score
   mapie.metrics.classification.classification_coverage_score_v2
   mapie.metrics.classification.classification_mean_width_score
   mapie.metrics.classification.classification_ssc
   mapie.metrics.classification.classification_ssc_score

Regression Metrics
====================================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie.metrics.regression.regression_coverage_score
   mapie.metrics.regression.regression_coverage_score_v2
   mapie.metrics.regression.regression_mean_width_score
   mapie.metrics.regression.regression_ssc
   mapie.metrics.regression.regression_ssc_score
   mapie.metrics.regression.hsic
   mapie.metrics.regression.coverage_width_based
   mapie.metrics.regression.regression_mwi_score

Utils (from mapie_v1)
==============================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   mapie_v1.utils.train_conformalize_test_split

Conformity Scores (Regression)
==============================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.conformity_scores.BaseRegressionScore
   mapie.conformity_scores.AbsoluteConformityScore
   mapie.conformity_scores.GammaConformityScore
   mapie.conformity_scores.ResidualNormalisedScore

Conformity Scores (Classification)
==================================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.conformity_scores.BaseClassificationScore
   mapie.conformity_scores.NaiveConformityScore
   mapie.conformity_scores.LACConformityScore
   mapie.conformity_scores.APSConformityScore
   mapie.conformity_scores.RAPSConformityScore
   mapie.conformity_scores.TopKConformityScore

Resampling
==========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.subsample.BlockBootstrap
   mapie.subsample.Subsample

Mondrian
========

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mapie.mondrian.MondrianCP
