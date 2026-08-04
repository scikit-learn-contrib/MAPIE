# Choosing the Right Algorithm

Start with the guarantee or diagnostic that your application needs. The table
below directs you to the appropriate MAPIE area before you choose a specific
class.

| Goal | MAPIE area | Output |
|---|---|---|
| Quantify uncertainty around a regression prediction | Conformal prediction | Prediction interval |
| Return several plausible classes | Conformal prediction | Prediction set |
| Make probability scores reflect observed frequencies | Calibration | Calibrated probabilities |
| Meet a target precision, recall, or custom decision metric | Risk control | Controlled decision rule |
| Check assumptions or detect deployment shifts | Exchangeability testing | Test or monitoring decision |

## Choosing a Conformal Predictor

Most conformal prediction methods require the data used to compute conformity
scores and future observations to be exchangeable. First determine whether a
separate, representative conformalization set is available and whether the base
model is already fitted.

| Situation | Recommended starting point |
|---|---|
| Regression with a separate conformalization set | `SplitConformalRegressor` |
| Classification with a separate conformalization set | `SplitConformalClassifier` |
| Small dataset where a held-out set would be costly | `CrossConformalRegressor` or `CrossConformalClassifier` |
| Regression with bootstrap resampling | `JackknifeAfterBootstrapRegressor` |
| Quantile-regression model | `ConformalizedQuantileRegressor` |
| Ordered observations with gradually available labels | `TimeSeriesRegressor` |
| Coverage required across user-defined feature groups | `ConditionalSplitConformalRegressor` or `ConditionalSplitConformalClassifier` |

### Split or Cross Conformal?

- **Split conformal** is the simplest and fastest option. Fit the model on one
  subset and compute conformity scores on another. Set `prefit=True` when the
  supplied model is already fitted; otherwise use `prefit=False` and call
  `fit` before `conformalize`.
- **Cross conformal** computes out-of-fold conformity scores with
  `fit_conformalize`. It uses limited data more efficiently but fits several
  models and is therefore more computationally expensive.

There is no universal dataset-size cutoff between the two. The decision depends
on model-training cost, the amount of representative data available, and the
precision needed when estimating a coverage quantile. See the
[conformalization-set guide](split-cross-conformal.md) for the complete
workflows.

### Regression or Classification?

- Choose a **regression** conformalizer when the target is numerical and the
  desired output is an interval.
- Choose a **classification** conformalizer when the target is categorical and
  the desired output is a set of labels. The conformity score (`"lac"`,
  `"aps"`, `"raps"`, or `"top_k"`) controls how those sets are constructed.

Conformal prediction provides marginal coverage by default. If the guarantee
must hold across selected subgroups or feature-defined functions, review the
[conditional-guarantees documentation](../theory/conditional-guarantees.md)
and its additional assumptions.

## Choosing a Risk Controller

Use risk control when the required guarantee is about a decision metric rather
than prediction-set or interval coverage. All controllers receive a prediction
function from an already-fitted model and use separate labeled data in their
`calibrate` method.

| Task | Controller | Supported starting points |
|---|---|---|
| Binary classification | `BinaryClassificationController` | Precision, recall, accuracy, false-positive rate, predicted-positive fraction, multiple risks, or a custom `BinaryRisk` |
| Multi-label classification | `MultiLabelClassificationController` | Recall with CRC or RCPS; precision with LTT |
| Semantic segmentation | `SemanticSegmentationController` | Recall with CRC or RCPS; precision with LTT |

### Binary Classification

`BinaryClassificationController` tests candidate decision parameters using the
Learn Then Test framework. For a standard probabilistic classifier, the
parameter is a probability threshold. It can also control multiple risks at
once or tune multi-dimensional parameters through a custom prediction
function.

Use it when you can state:

1. the metric or risk to control;
2. the minimum performance or maximum risk level (`target_level`);
3. the confidence of the guarantee (`confidence_level`);
4. the candidate decision parameters, if the default threshold grid is not
   suitable.

See the [binary risk-control quick start](quick-start.md#4-risk-control-guaranteed-decision-thresholds)
for a runnable example.

### Multi-label Classification and Semantic Segmentation

Choose the method according to the metric and type of guarantee:

| Goal | Method | Assumption | Guarantee |
|---|---|---|---|
| Control expected recall | CRC | Exchangeable data | Expected-risk control |
| Control recall with specified confidence | RCPS | i.i.d. data | High-probability risk control |
| Control precision with specified confidence | LTT | i.i.d. data | High-probability risk control |

CRC is the default for recall and does not require a `confidence_level`. RCPS
and LTT do require one. The detailed [Risk Control overview](../introductions/risk-control.md)
and [theoretical description](../theory/risk-control.md) explain these
guarantees and assumptions.

!!! warning "Risk control can be infeasible"
    A controller may find no candidate parameter that supports the requested
    target and confidence level. This is a valid outcome, not a software error.
    More representative calibration data or a better predictive model may be
    necessary.

## Check the Data Assumptions

Distribution shifts can invalidate conformal prediction and risk-control
guarantees. Use the [Exchangeability Testing overview](../introductions/exchangeability-testing.md)
to choose between fixed-dataset tests, online martingale tests, and deployed
model risk monitoring. A test can find evidence against exchangeability, but
cannot prove that every possible violation is absent.
