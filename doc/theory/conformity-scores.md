# Conformity Scores — Theoretical Description

!!! note "Terminology"
    In theoretical parts of the documentation:

    - `alpha` is equivalent to `1 - confidence_level` — it can be seen as a *risk level*.
    - *calibrate* and *calibration* are equivalent to *conformalize* and *conformalization*.

---

The `BaseRegressionScore` class implements various methods to compute conformity scores for regression.

!!! tip "Custom Scores"
    Users can create any conformal scores not already included in MAPIE by inheriting from `BaseRegressionScore`.

## Mathematical Setting

With conformal predictions, we want to transform a **heuristic notion of uncertainty** from a model into a **rigorous one**. The first step is to choose a conformal score.

The only requirement for the score function \(s(X, Y) \in \mathbb{R}\) is that **larger scores encode worse agreement** between \(X\) and \(Y\) [^1].

There are two types of scores:

- **Symmetric**: Two quantiles are computed (right and left side of the distribution).
- **Asymmetric**: A single quantile direction.

---

## 1. Absolute Residual Score

The **absolute residual score** [^1] (`AbsoluteConformityScore`) is the simplest and most commonly used:


\[
s(X, Y) = |Y - \hat{\mu}(X)|
\]


Prediction interval bounds:


\[
[\hat{\mu}(X) - q(s),  \hat{\mu}(X) + q(s)]
\]


where \(q(s)\) is the \((1-\alpha)\) quantile of the conformity scores.

!!! info
    With this score, prediction intervals are **constant** across the whole dataset. This score is **symmetric** by default.

---

## 2. Gamma Score

The **gamma score** [^2] (`GammaConformityScore`) adds **adaptivity** by normalizing residuals by predictions:


\[
s(X, Y) = \frac{|Y - \hat{\mu}(X)|}{\hat{\mu}(X)}
\]


Adaptive prediction intervals:


\[
[\hat{\mu}(X) \cdot (1 - q(s)),  \hat{\mu}(X) \cdot (1 + q(s))]
\]


!!! info
    This score is **asymmetric** by default. It produces intervals proportional to the magnitude of predictions — useful when you expect greater uncertainty for larger predictions.

---

## 3. Residual Normalized Score

The **residual normalized score** [^1] (`ResidualNormalisedScore`) uses an **additional model** \(\hat{\sigma}\) that learns to predict the base model's residuals:


\[
s(X, Y) = \frac{|Y - \hat{\mu}(X)|}{\hat{\sigma}(X)}
\]


where \(\hat{\sigma}\) is trained on \((X, |Y - \hat{\mu}(X)|)\).

Prediction intervals:


\[
[\hat{\mu}(X) - q(s) \cdot \hat{\sigma}(X),  \hat{\mu}(X) + q(s) \cdot \hat{\sigma}(X)]
\]


!!! info
    This score is **symmetric** by default. Due to the additional model, it can only be used with **split methods**.

---

## Key Takeaways


| Score                   | Adaptivity                            | Default Symmetry | Key Property                                           |
| ----------------------- | ------------------------------------- | ---------------- | ------------------------------------------------------ |
| **Absolute Residual**   | Constant intervals                    | Symmetric        | Simplest, default for regression                       |
| **Gamma**               | Adaptive, proportional to predictions | Asymmetric       | Good when uncertainty scales with prediction magnitude |
| **Residual Normalized** | Highly adaptive                       | Symmetric        | Requires additional model, no assumptions on data      |


---

## References

[^1]: Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. (2018). *Distribution-Free Predictive Inference for Regression.* JASA, 113(523), 1094–1111.
[^2]: Cordier, T., Blot, V., Lacombe, L., Morzadec, T., Capitaine, A. & Brunel, N. (2023). *Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library.* PMLR.