# Conformal Prediction — Theory

Conformal prediction is a framework for quantifying the uncertainty of a
model's predictions. Instead of returning only a point prediction, it uses the
model's past errors to produce an output that expresses which values or labels
remain plausible for a new observation.

MAPIE applies conformal prediction to any scikit-learn-compatible estimator:

- for **regression**, it returns a prediction interval around each prediction;
- for **classification**, it returns a set of plausible classes.

!!! note "Terminology"
    In theoretical parts of the documentation:

    - `alpha` is equivalent to `1 - confidence_level` — it can be seen as a *risk level*.
    - *calibrate* and *calibration* are equivalent to *conformalize* and *conformalization*.

## Mathematical Setting

For a standard i.i.d. case, our data
$(X, Y) = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ has an unknown distribution
$P_{X, Y}$.

Given some target quantile $\alpha$, we aim at constructing a prediction
region $\hat{C}_{n, \alpha}$ such that:

$$
P \{Y_{n+1} \in \hat{C}_{n, \alpha}(X_{n+1}) \} \geq 1 - \alpha
$$

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

The task-specific theory is presented in the following pages:

- [Regression](regression.md) explains prediction intervals and the split,
  cross-validation, jackknife, and bootstrap-based methods.
- [Classification](classification.md) explains prediction sets and the
  split- and cross-conformal strategies.
- [Conformity scores](conformity-scores.md) describes the regression and
  classification scores available in MAPIE.
