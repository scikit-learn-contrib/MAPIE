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

## Explore Conformal Prediction in MAPIE

- [Theory](theory.md) presents the foundational elements of conformal prediction.
- [Regression theory](regression.md) explains prediction intervals
  and the split, cross-validation, jackknife, and bootstrap-based methods.
- [Classification](classification.md) explains prediction sets and the
  split- and cross-conformal strategies.
- [Conformity scores](conformity-scores.md) describes how MAPIE
  measures whether a prediction agrees with an observation in regression and
  classification.
- [Conformalization set](split-cross-conformal.md) compares
  split- and cross-conformal workflows.
- [Metrics](metrics.md) covers coverage, interval width,
  prediction-set size, and conditional coverage diagnostics.
- [Conditional guarantees](conditional-guarantees.md) introduces
  methods that go beyond marginal coverage.
