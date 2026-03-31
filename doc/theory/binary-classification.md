# Binary Classification — Theoretical Description

The binary classification case relates three approaches for uncertainty quantification:

1. **Calibration** — Transforms scores to probabilities
2. **Confidence Intervals** — Confidence interval for the predictive distribution
3. **Prediction Sets** — Likely predictions with probabilistic guarantee

!!! quote "Gupta et al."
    These three concepts are deeply related for score-based classifiers. The informativeness of prediction sets depends on the quality of calibration.

---

## Calibration

The goal is to transform a non-probability score into a **true probability**:

\[
\Pr(Y = 1 \mid h(X) = q) = q
\]

[:material-arrow-right: Full calibration documentation](calibration.md)

---

## Prediction Sets

Construct **conformal prediction sets** with a marginal coverage guarantee:

\[
P \{Y_{n+1} \in \hat{C}_{n, \alpha}(X_{n+1}) \} \geq 1 - \alpha
\]

[:material-arrow-right: Full classification documentation](classification.md)

---

## Probabilistic Prediction

Confidence intervals for the **predictive distribution** of the model, combining both calibration and prediction set approaches.

---

## Learn More

- [Classification methods (LAC, APS, Top-K)](classification.md)
- [Calibration (Top-Label)](calibration.md)
- [Risk control for binary classification](risk-control.md)
