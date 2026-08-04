# Conformal Prediction

Conformal prediction is a framework for quantifying the uncertainty of a
model's predictions. Instead of returning only a point prediction, it uses the
model's past errors to produce an output that expresses which values or labels
remain plausible for a new observation.

MAPIE applies conformal prediction to any scikit-learn-compatible estimator:

- for **regression**, it returns a prediction interval around each prediction;
- for **classification**, it returns a set of plausible classes.

For example, with a confidence level of 90%, conformal prediction aims for the
true value or class to be included in the predicted interval or set for at
least 90% of future observations. This is a **marginal coverage** guarantee: it
holds on average over new observations, not necessarily for every subgroup or
individual observation.

## How It Works

A typical MAPIE workflow has three steps:

1. **Fit** a base estimator on training data.
2. **Conformalize** it by measuring prediction errors, called conformity
   scores, on data that was not used to fit the estimator.
3. **Predict** intervals or sets at the requested confidence level.

The conformalization data can be a separate held-out set, or it can be obtained
through a cross-validation or resampling strategy. The resulting guarantees
require the conformalization observations and future observations to be
exchangeable.

Conformal prediction does not make an inaccurate model more accurate. It adds
an uncertainty layer whose intervals or sets reflect the errors observed during
conformalization. Better base models generally produce more informative,
narrower outputs while preserving the target coverage.

## Explore Conformal Prediction in MAPIE

- [Regression theory](regression.md) explains prediction intervals
  and the split, cross-validation, jackknife, and bootstrap-based methods.
- [Classification theory](classification.md) explains prediction sets
  and the LAC, APS, RAPS, and Top-K methods.
- [Conformity scores](conformity-scores.md) describes how MAPIE
  measures whether a prediction agrees with an observation.
- [Conformalization set](../getting-started/split-cross-conformal.md) compares
  split- and cross-conformal workflows.
- [Metrics](metrics.md) covers coverage, interval width,
  prediction-set size, and conditional coverage diagnostics.
- [Conditional guarantees](conditional-guarantees.md) introduces
  methods that go beyond marginal coverage.
