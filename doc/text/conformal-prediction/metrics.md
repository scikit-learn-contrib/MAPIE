# Metrics for Conformal Prediction — Theory

!!! note "Terminology"
    In theoretical parts of the documentation:

    - `alpha` is equivalent to `1 - confidence_level` — it can be seen as a *risk level*.
    - *calibrate* and *calibration* are equivalent to *conformalize* and *conformalization*.

---

This document provides detailed descriptions of various metrics used to evaluate the performance of predictive models, particularly focusing on their ability to estimate uncertainties and calibrate predictions accurately.

## General metrics

### Regression Coverage Score (RCS)

Calculates the **fraction of true outcomes** that fall within the provided prediction intervals:

$$
\text{RCS} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(\hat{y}^{\text{low}}_{i} \leq y_{i} \leq \hat{y}^{\text{up}}_{i})
$$

### Regression Mean Width Score (RMWS)

Assesses the **average width** of the prediction intervals:

$$
\text{RMWS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{\text{up}}_{i} - \hat{y}^{\text{low}}_{i})
$$

### Classification Coverage Score (CCS)

Measures how often the true class labels fall **within the predicted sets**:

$$
\text{CCS} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(y_{i} \in \hat{C}(x_{i}))
$$

### Classification Mean Width Score (CMWS)

Average **size of the prediction sets** across all samples:

$$
\text{CMWS} = \frac{1}{n} \sum_{i=1}^{n} |\hat{C}(x_i)|
$$

### Coverage Width-Based Criterion (CWC)

Balances **empirical coverage and width**, rewarding narrow intervals and penalizing poor coverage [^1]:

$$
\text{CWC} = (1 - \text{Mean Width Score}) \times \exp\left(-\eta \times (\text{Coverage Score} - (1-\alpha))^2\right)
$$

### Mean Winkler Interval Score (MWI)

Combines interval width with a **penalty for non-coverage** [^2]:

$$
\text{MWI Score} = \frac{1}{n} \sum_{i=1}^{n} \left[(\hat{y}^{\text{up}}_{i} - \hat{y}^{\text{low}}_{i}) + \frac{2}{\alpha} \max(0, |y_{i} - \hat{y}^{\text{boundary}}_{i}|)\right]
$$

## Conditional coverage diagnostics

Conditional coverage diagnostics evaluate how prediction sets or intervals behave
across subpopulations, beyond their marginal coverage guarantee.

### Size-Stratified Coverage (SSC)

Evaluates how the size of prediction sets or intervals affects their ability to cover true outcomes [^3]:

**Regression:**

$$
\text{SSC}_{\text{regression}} = \sum_{k=1}^{K} \left( \frac{1}{|I_k|} \sum_{i \in I_k} \mathbf{1}(\hat{y}^{\text{low}}_{i} \leq y_{i} \leq \hat{y}^{\text{up}}_{i}) \right)
$$

**Classification:**

$$
\text{SSC}_{\text{classification}} = \sum_{k=1}^{K} \left( \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{1}(y_{i} \in \hat{C}(x_i)) \right)
$$

---

### Hilbert-Schmidt Independence Criterion (HSIC)

A **non-parametric** measure of independence between interval sizes and coverage indicators [^4]:

$$
\text{HSIC} = \operatorname{trace}(\mathbf{H} \mathbf{K} \mathbf{H} \mathbf{L})
$$

where:

- $\mathbf{K}$, $\mathbf{L}$ are kernel matrices for interval sizes and coverage indicators
- $\mathbf{H} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ is the centering matrix

### Coverage Gap (CovGap and WCovGap) [^5]

Measures how far empirical coverage is from the target coverage inside
predefined groups. For an evaluation set $\{(x_i, y_i, g_i)\}_{i=1}^n$
of size $n$, let $\alpha$ be the target miscoverage level,
$\hat{C}(x_i)$ be the prediction set or interval for sample $i$, $G$ be
the set of observed groups, $I_g = \{i: g_i = g\}$, and $c_i$ be the
binary coverage indicator:

$$
c_i =
\begin{cases}
1, & y_i \in \hat{C}(x_i) \\
0, & \text{otherwise.}
\end{cases}
$$

For regression intervals, $c_i = 1$ means
$\hat{y}^{\text{low}}_i \leq y_i \leq \hat{y}^{\text{up}}_i$, where
$\hat{y}^{\text{low}}_i$ and $\hat{y}^{\text{up}}_i$ are the interval
endpoints. For classification sets, $c_i = 1$ means
$y_i \in \hat{C}(x_i)$.

The empirical coverage of group $g$ is:

$$
\widehat{Cov}_g = \frac{1}{|I_g|} \sum_{i \in I_g} c_i
$$

The unweighted coverage gap averages the absolute group-level deviations from
the target coverage $1-\alpha$:

$$
\text{CovGap} = \frac{1}{|G|} \sum_{g \in G}
\left| \widehat{Cov}_g - (1-\alpha) \right|
$$

The weighted coverage gap weights each group by its empirical sample
proportion:

$$
\text{WCovGap} = \sum_{g \in G} \frac{|I_g|}{n}
\left| \widehat{Cov}_g - (1-\alpha) \right|
$$

CovGap gives small and large groups the same influence, which is useful when
each group is equally important. WCovGap gives more influence to larger groups,
which summarizes the average conditional coverage deviation over samples.

### Worst Slab Coverage (WSC) [^6]

Worst-case slab coverage evaluates conditional coverage over geometric slices of
feature space rather than over predefined groups. Assume that $X \subset \mathbb{R}^d$
and that a predictive set rule $C_\alpha(\cdot)$ is evaluated on a test dataset
$\mathcal{D}_{\text{test}} = \{(X_i, Y_i)\}_{i=1}^n$.

For a direction $v \in \mathbb{R}^d$ and scalars $a < b$, define the slab:

$$
S_{v,a,b} := \{x \in \mathbb{R}^d : a \leq v^\top x \leq b\}.
$$

Let $I_{v,a,b} = \{i : X_i \in S_{v,a,b}\}$. For a mass threshold
$\delta \in (0, 1]$, the empirical WSC in direction $v$ is:

$$
\mathrm{WSC}_n(C_\alpha(\cdot), v)
= \inf_{a < b}
\left\{
\frac{1}{|I_{v,a,b}|}
\sum_{i \in I_{v,a,b}}
\mathbf{1}\{Y_i \in C_\alpha(X_i)\}
\;\middle|\;
\frac{|I_{v,a,b}|}{n} \geq \delta
\right\}.
$$

The constraint $|I_{v,a,b}| / n \geq \delta$ prevents the diagnostic from
selecting slabs that contain too few test samples. Smaller values of
$\delta$ allow more localized diagnostics; larger values force WSC to look at
larger slices of the feature space.

In practice, the metric is computed over a finite set of directions
$V$, typically generated by random sampling, as:

$$
\mathrm{WSC} = \inf_{v \in V}
\mathrm{WSC}_n(C_\alpha(\cdot), v).
$$

MAPIE's `worst_slab_coverage` wrapper first converts regression intervals or
classification prediction sets into the binary indicators
$\mathbf{1}\{Y_i \in C_\alpha(X_i)\}$, then delegates the slab search to
`covmetrics.WSC`. The `n_directions` parameter controls the size of the finite
set $V$, and `delta` is the mass threshold above.

Unlike CovGap, WSC does not compare coverage to the target $1-\alpha$ and
does not require user-defined groups. It returns a coverage value directly:
values well below the target coverage reveal a slab of the feature space where
the prediction sets or intervals under-cover.

### Excess Risk of the Target Coverage (ERT) [^7]

ERT estimates whether the conditional coverage function
$x \mapsto \mathbb{P}\{Y \in \hat{C}(X) \mid X=x\}$ carries useful signal
beyond the constant target coverage $1-\alpha$. It trains a classifier to
predict the binary coverage indicator $c_i$ from the features $x_i$, then
compares the loss of this learned conditional coverage predictor with the loss
of the constant predictor $1-\alpha$:

$$
\widehat{\ell\text{-}\mathrm{ERT}}(h) :=
\frac{1}{m}\sum_{i=1}^m
\left[
\ell(1-\alpha, c_i) - \ell(h(x_i), c_i)
\right].
$$

Here, $h(x_i)$ is the predicted conditional coverage and $\ell$ is a
proper loss. Larger positive values indicate that the features help predict
coverage, which is evidence of conditional coverage variation. Values near zero
indicate little detectable improvement over the target coverage baseline, given
the chosen classifier and loss.

MAPIE's `excess_risk_target_coverage` wrapper converts intervals or prediction
sets into the binary indicators $c_i$, then delegates the cross-validated ERT
estimation to `covmetrics.ERT`. The `model_cls` and `model_kwargs` parameters
control the classifier used to estimate conditional coverage, while `n_splits`
controls the cross-validation estimate.

For more advanced use of ERT, you can directly import ERT from `covmetrics` and
follow the guidelines of the official GitHub
[repository](https://github.com/ElSacho/covmetrics/).

The [conditional-metrics comparison notebook](https://github.com/scikit-learn-contrib/MAPIE/blob/master/notebooks/metrics/conditional_metrics_covmetrics_comparison.ipynb)
computes CovGap, WSC, and ERT on MAPIE regression intervals and classification
prediction sets, then checks the results against `covmetrics` directly.


---

## References

[^1]: Khosravi, A., et al. "Comprehensive Review of Neural Network-Based Prediction Intervals." IEEE Trans. Neural Netw., 2011.
[^2]: Winkler, R. L. "A Decision-Theoretic Approach to Interval Estimation." JASA, 1972.
[^3]: Angelopoulos, A. N., et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR 2021.
[^4]: Gretton, A., et al. "A Kernel Two-Sample Test." JMLR, 2012.
[^5]: Ding, T., Angelopoulos, A., Bates, S., Jordan, M., and Tibshirani, R. J. "Class-conditional conformal prediction with many classes." *NeurIPS*, 2023. [arXiv:2306.09335](https://arxiv.org/abs/2306.09335).
[^6]: Cauchois, M., Gupta, S., and Duchi, J. "Knowing What You Know: Valid and Validated Confidence Sets in Multiclass and Multilabel Prediction." *JMLR*, 2021. [JMLR 22(81)](https://www.jmlr.org/papers/v22/20-753.html).
[^7]: Braun, S., Holzmüller, D., Jordan, M. I., and Bach, F. "Conditional Coverage Diagnostics for Conformal Prediction." *arXiv:2512.11779*, 2025. [arXiv:2512.11779](https://arxiv.org/abs/2512.11779).
