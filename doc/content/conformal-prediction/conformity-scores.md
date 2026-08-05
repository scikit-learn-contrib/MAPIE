# Conformity Scores — Theory

The `BaseRegressionScore` and `BaseClassificationScore` classes implement various methods to compute conformity scores for regression and classification.

!!! tip "Custom Scores"
    Users can create any conformal scores not already included in MAPIE by inheriting from `BaseRegressionScore` or `BaseClassificationScore`.

## Mathematical Setting

With conformal predictions, we want to transform a **heuristic notion of uncertainty** from a model into a **rigorous one**. The first step is to choose a conformal score.

The only requirement for the score function $s(X, Y) \in \mathbb{R}$ is that **larger scores encode worse agreement** between $X$ and $Y$ [^1].

There are two types of scores:

- **Symmetric**: Two quantiles are computed (right and left side of the distribution).
- **Asymmetric**: A single quantile direction.

---

## Regression Scores

### 1. Absolute Residual Score

The **absolute residual score** [^1] (`AbsoluteConformityScore`) is the simplest and most commonly used:


$$
s(X, Y) = |Y - \hat{\mu}(X)|
$$


Prediction interval bounds:


$$
[\hat{\mu}(X) - q(s),  \hat{\mu}(X) + q(s)]
$$


where $q(s)$ is the $(1-\alpha)$ quantile of the conformity scores.

!!! info
    With this score, prediction intervals are **constant** across the whole dataset. This score is **symmetric** by default.

---

### 2. Gamma Score

The **gamma score** [^2] (`GammaConformityScore`) adds **adaptivity** by normalizing residuals by predictions:


$$
s(X, Y) = \frac{|Y - \hat{\mu}(X)|}{\hat{\mu}(X)}
$$


Adaptive prediction intervals:


$$
[\hat{\mu}(X) \cdot (1 - q(s)),  \hat{\mu}(X) \cdot (1 + q(s))]
$$


!!! info
    This score is **asymmetric** by default. It produces intervals proportional to the magnitude of predictions — useful when you expect greater uncertainty for larger predictions.

---

### 3. Residual Normalized Score

The **residual normalized score** [^1] (`ResidualNormalisedScore`) uses an **additional model** $\hat{\sigma}$ that learns to predict the base model's residuals:


$$
s(X, Y) = \frac{|Y - \hat{\mu}(X)|}{\hat{\sigma}(X)}
$$


where $\hat{\sigma}$ is trained on $(X, |Y - \hat{\mu}(X)|)$.

Prediction intervals:


$$
[\hat{\mu}(X) - q(s) \cdot \hat{\sigma}(X),  \hat{\mu}(X) + q(s) \cdot \hat{\sigma}(X)]
$$


!!! info
    This score is **symmetric** by default. Due to the additional model, it can only be used with **split methods**.

### Key Takeaways


| Score                   | Adaptivity                            | Default Symmetry | Key Property                                           |
| ----------------------- | ------------------------------------- | ---------------- | ------------------------------------------------------ |
| **Absolute Residual**   | Constant intervals                    | Symmetric        | Simplest, default for regression                       |
| **Gamma**               | Adaptive, proportional to predictions | Asymmetric       | Good when uncertainty scales with prediction magnitude |
| **Residual Normalized** | Highly adaptive                       | Symmetric        | Requires additional model, no assumptions on data      |


---

## Classification Scores

### 1. LAC

In the LAC method [^3], the conformity score is **one minus the score of the true label**:


$$
s_i(X_i, Y_i) = 1 - \hat{\mu}(X_i)_{Y_i}
$$


The quantile $\hat{q}$ is computed as:


$$
\hat{q} = \text{Quantile}\left(s_1, \ldots, s_n ; \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)
$$


The prediction set includes all labels with score higher than the threshold:


$$
\hat{C}(X_{\text{test}}) = \{y : \hat{\mu}(X_{\text{test}})_y \geq 1 - \hat{q}\}
$$


!!! warning
    Although LAC generally results in small prediction sets, it tends to produce **empty sets** when the model is uncertain (e.g., at the border between two classes).

---

### 2. Top-K

Introduced in [^5], the **Top-K** method gives the **same prediction set size** for all observations. The conformity score is the rank of the true label:


$$
s_i(X_i, Y_i) = j \quad \text{where} \quad Y_i = \pi_j \quad \text{and} \quad \hat{\mu}(X_i)_{\pi_1} > \cdots > \hat{\mu}(X_i)_{\pi_n}
$$



$$
\hat{q} = \left\lceil \text{Quantile}\left(s_1, \ldots, s_n ; \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right) \right\rceil
$$



$$
\hat{C}(X_{\text{test}}) = \{\pi_1, \ldots, \pi_{\hat{q}}\}
$$


---

### 3. Adaptive Prediction Sets (APS)

The APS method overcomes LAC's empty set problem by constructing **non-empty** prediction sets. Conformity scores are computed by **summing ranked scores** until reaching the true label:


$$
s_i(X_i, Y_i) = \sum^k_{j=1} \hat{\mu}(X_i)_{\pi_j} \quad \text{where} \quad Y_i = \pi_k
$$


Prediction sets are built similarly:


$$
\hat{C}(X_{\text{test}}) = \{\pi_1, \ldots, \pi_k\} \quad \text{where} \quad k = \inf\left\{k : \sum^k_{j=1} \hat{\mu}(X_{\text{test}})_{\pi_j} \geq \hat{q}\right\}
$$


By default, the label whose cumulative score exceeds the quantile is included. Its incorporation can also be randomized for tighter effective coverage [^4] [^5].

---

### 4. Regularized Adaptive Prediction Sets (RAPS)

RAPS [^5] improves APS by **regularizing** to avoid very large prediction sets:


$$
s_i(X_i, Y_i) = \sum^k_{j=1} \hat{\mu}(X_i)_{\pi_j} + \lambda (k - k_{\text{reg}})^+ \quad \text{where} \quad Y_i = \pi_k
$$


Where:

- $(z)^+$ denotes the positive part of $z$
- $k_{\text{reg}}$ is the optimal set size (determined by the Top-K method on a held-out split)
- $\lambda$ is a regularization parameter (grid search over 0.001, 0.01, 0.1, 0.2, 0.5)

Prediction set construction:


$$
\hat{C}(X_{\text{test}}) = \{\pi_1, \ldots, \pi_k\} \quad \text{where} \quad k = \inf\left\{k : \sum^k_{j=1} \hat{\mu}(X_{\text{test}})_{\pi_j} + \lambda(k - k_{\text{reg}})^+ \geq \hat{q}\right\}
$$


#### Exact Coverage via Randomization

To achieve exact coverage, randomization on the last label can be applied:

1. Define $V_i = \frac{s_i(X_i, Y_i) - \hat{q}_{1-\alpha}}{\hat{\mu}(X_i)_{\pi_k} + \lambda \mathbb{1}(k > k_{\text{reg}})}$.
2. Compare each $V_i$ to $U \sim \text{Unif}(0, 1)$.
3. If $V_i \leq U$, the last included label is removed.

---

## References

[^1]: Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. (2018). *Distribution-Free Predictive Inference for Regression.* JASA, 113(523), 1094–1111.
[^2]: Cordier, T., Blot, V., Lacombe, L., Morzadec, T., Capitaine, A. & Brunel, N. (2023). *Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library.* PMLR.
[^3]: Sadinle, Mauricio, Jing Lei, & Larry Wasserman. "Least Ambiguous Set-Valued Classifiers With Bounded Error Levels." *JASA*, 114:525, 223-234, 2019.
[^4]: Romano, Yaniv, Matteo Sesia and Emmanuel J. Candès. "Classification with Valid and Adaptive Coverage." *NeurIPS* 2020 (spotlight).
[^5]: Angelopoulos, Anastasios N., Stephen Bates, Michael Jordan and Jitendra Malik. "Uncertainty Sets for Image Classifiers using Conformal Prediction." *ICLR* 2021.
