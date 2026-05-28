# API Reference

Complete API documentation for MAPIE v1.

---

## [Regression](regression.md)

### Conformalizers

| Class | Description |
|---|---|
| [`SplitConformalRegressor`](regression.md#mapie.regression.SplitConformalRegressor) | Split conformal prediction for regression |
| [`CrossConformalRegressor`](regression.md#mapie.regression.CrossConformalRegressor) | Cross-conformal prediction for regression |
| [`JackknifeAfterBootstrapRegressor`](regression.md#mapie.regression.JackknifeAfterBootstrapRegressor) | Jackknife+-after-bootstrap for regression |
| [`ConformalizedQuantileRegressor`](regression.md#mapie.regression.ConformalizedQuantileRegressor) | Conformalized quantile regression |
| [`TimeSeriesRegressor`](regression.md#mapie.regression.TimeSeriesRegressor) | Conformal prediction for time series |

### Metrics

| Function | Description |
|---|---|
| [`regression_coverage_score`](metrics.md#mapie.metrics.regression.regression_coverage_score) | Fraction of outcomes within prediction intervals |
| [`regression_mean_width_score`](metrics.md#mapie.metrics.regression.regression_mean_width_score) | Average width of prediction intervals |
| [`regression_ssc`](metrics.md#mapie.metrics.regression.regression_ssc) | Size-stratified coverage for regression |
| [`regression_ssc_score`](metrics.md#mapie.metrics.regression.regression_ssc_score) | Size-stratified coverage score |
| [`hsic`](metrics.md#mapie.metrics.regression.hsic) | Hilbert-Schmidt Independence Criterion |
| [`coverage_width_based`](metrics.md#mapie.metrics.regression.coverage_width_based) | Coverage width-based criterion |
| [`regression_mwi_score`](metrics.md#mapie.metrics.regression.regression_mwi_score) | Mean Winkler Interval score |

### Conformity Scores

| Class | Description |
|---|---|
| [`BaseRegressionScore`](conformity-scores.md#mapie.conformity_scores.BaseRegressionScore) | Base class for regression conformity scores |
| [`AbsoluteConformityScore`](conformity-scores.md#mapie.conformity_scores.AbsoluteConformityScore) | Absolute residual conformity score |
| [`GammaConformityScore`](conformity-scores.md#mapie.conformity_scores.GammaConformityScore) | Gamma (normalized) conformity score |
| [`ResidualNormalisedScore`](conformity-scores.md#mapie.conformity_scores.ResidualNormalisedScore) | Residual normalized conformity score |

### Resampling

| Class | Description |
|---|---|
| [`BlockBootstrap`](utils.md#mapie.subsample.BlockBootstrap) | Block bootstrap for time series |
| [`Subsample`](utils.md#mapie.subsample.Subsample) | Subsample for jackknife-after-bootstrap |

---

## [Classification](classification.md)

### Conformalizers

| Class | Description |
|---|---|
| [`SplitConformalClassifier`](classification.md#mapie.classification.SplitConformalClassifier) | Split conformal prediction for classification |
| [`CrossConformalClassifier`](classification.md#mapie.classification.CrossConformalClassifier) | Cross-conformal prediction for classification |

### Metrics

| Function | Description |
|---|---|
| [`classification_coverage_score`](metrics.md#mapie.metrics.classification.classification_coverage_score) | Fraction of true labels in prediction sets |
| [`classification_mean_width_score`](metrics.md#mapie.metrics.classification.classification_mean_width_score) | Average size of prediction sets |
| [`classification_ssc`](metrics.md#mapie.metrics.classification.classification_ssc) | Size-stratified coverage for classification |
| [`classification_ssc_score`](metrics.md#mapie.metrics.classification.classification_ssc_score) | Size-stratified coverage score |

### Conformity Scores

| Class | Description |
|---|---|
| [`BaseClassificationScore`](conformity-scores.md#mapie.conformity_scores.BaseClassificationScore) | Base class for classification scores |
| [`NaiveConformityScore`](conformity-scores.md#mapie.conformity_scores.NaiveConformityScore) | Naive conformity score |
| [`LACConformityScore`](conformity-scores.md#mapie.conformity_scores.LACConformityScore) | Least Ambiguous Classifier score |
| [`APSConformityScore`](conformity-scores.md#mapie.conformity_scores.APSConformityScore) | Adaptive Prediction Sets score |
| [`RAPSConformityScore`](conformity-scores.md#mapie.conformity_scores.RAPSConformityScore) | Regularized APS score |
| [`TopKConformityScore`](conformity-scores.md#mapie.conformity_scores.TopKConformityScore) | Top-K conformity score |

---

## [Risk Control](risk-control.md)

| Class | Description |
|---|---|
| [`MultiLabelClassificationController`](risk-control.md#mapie.risk_control.MultiLabelClassificationController) | Risk control for multi-label classification |
| [`BinaryClassificationController`](risk-control.md#mapie.risk_control.BinaryClassificationController) | Risk control for binary classification |
| [`SemanticSegmentationController`](risk-control.md#mapie.risk_control.SemanticSegmentationController) | Risk control for semantic segmentation |
| [`BinaryRisk`](risk-control.md#mapie.risk_control.BinaryRisk) | Binary classification risk utilities |
| [`BinaryClassificationRisk`](risk-control.md#mapie.risk_control.BinaryClassificationRisk) | Deprecated alias for `BinaryRisk` |

---

## [Calibration](calibration.md)

### Calibrators

| Class | Description |
|---|---|
| [`TopLabelCalibrator`](calibration.md#mapie.calibration.TopLabelCalibrator) | Top-label calibration for multi-class |
| [`VennAbersCalibrator`](calibration.md#mapie.calibration.VennAbersCalibrator) | Venn-Abers calibration |

### Metrics

| Function | Description |
|---|---|
| [`expected_calibration_error`](metrics.md#mapie.metrics.calibration.expected_calibration_error) | ECE metric |
| [`top_label_ece`](metrics.md#mapie.metrics.calibration.top_label_ece) | Top-label ECE for multi-class |
| [`kolmogorov_smirnov_statistic`](metrics.md#mapie.metrics.calibration.kolmogorov_smirnov_statistic) | KS statistic |
| [`kolmogorov_smirnov_p_value`](metrics.md#mapie.metrics.calibration.kolmogorov_smirnov_p_value) | KS p-value |
| [`kuiper_statistic`](metrics.md#mapie.metrics.calibration.kuiper_statistic) | Kuiper statistic |
| [`kuiper_p_value`](metrics.md#mapie.metrics.calibration.kuiper_p_value) | Kuiper p-value |
| [`spiegelhalter_statistic`](metrics.md#mapie.metrics.calibration.spiegelhalter_statistic) | Spiegelhalter statistic |
| [`spiegelhalter_p_value`](metrics.md#mapie.metrics.calibration.spiegelhalter_p_value) | Spiegelhalter p-value |

---

## [Utilities](utils.md)

| Item | Description |
|---|---|
| [`train_conformalize_test_split`](utils.md#mapie.utils.train_conformalize_test_split) | Split data into train, conformalize, and test sets |
| [`Subsample`](utils.md#mapie.subsample.Subsample) | Subsample for jackknife-after-bootstrap |
| [`BlockBootstrap`](utils.md#mapie.subsample.BlockBootstrap) | Block bootstrap for time series |
