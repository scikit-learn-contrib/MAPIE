# Metrics — Theoretical Description

!!! note "Terminology"
    In theoretical parts of the documentation:

    - `alpha` is equivalent to `1 - confidence_level` — it can be seen as a *risk level*.
    - *calibrate* and *calibration* are equivalent to *conformalize* and *conformalization*.

---

This document provides detailed descriptions of various metrics used to evaluate the performance of predictive models, particularly focusing on their ability to estimate uncertainties and calibrate predictions accurately.

## 1. General Metrics

### Regression Coverage Score (RCS)

Calculates the **fraction of true outcomes** that fall within the provided prediction intervals:

\[
\text{RCS} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(\hat{y}^{\text{low}}_{i} \leq y_{i} \leq \hat{y}^{\text{up}}_{i})
\]

### Regression Mean Width Score (RMWS)

Assesses the **average width** of the prediction intervals:

\[
\text{RMWS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{\text{up}}_{i} - \hat{y}^{\text{low}}_{i})
\]

### Classification Coverage Score (CCS)

Measures how often the true class labels fall **within the predicted sets**:

\[
\text{CCS} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(y_{i} \in \hat{C}(x_{i}))
\]

### Classification Mean Width Score (CMWS)

Average **size of the prediction sets** across all samples:

\[
\text{CMWS} = \frac{1}{n} \sum_{i=1}^{n} |\hat{C}(x_i)|
\]

---

### Size-Stratified Coverage (SSC)

Evaluates how the size of prediction sets or intervals affects their ability to cover true outcomes [^1]:

**Regression:**

\[
\text{SSC}_{\text{regression}} = \sum_{k=1}^{K} \left( \frac{1}{|I_k|} \sum_{i \in I_k} \mathbf{1}(\hat{y}^{\text{low}}_{i} \leq y_{i} \leq \hat{y}^{\text{up}}_{i}) \right)
\]

**Classification:**

\[
\text{SSC}_{\text{classification}} = \sum_{k=1}^{K} \left( \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{1}(y_{i} \in \hat{C}(x_i)) \right)
\]

---

### Hilbert-Schmidt Independence Criterion (HSIC)

A **non-parametric** measure of independence between interval sizes and coverage indicators [^4]:

\[
\text{HSIC} = \operatorname{trace}(\mathbf{H} \mathbf{K} \mathbf{H} \mathbf{L})
\]

where:

- \(\mathbf{K}\), \(\mathbf{L}\) are kernel matrices for interval sizes and coverage indicators
- \(\mathbf{H} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^\top\) is the centering matrix

---

### Coverage Width-Based Criterion (CWC)

Balances **empirical coverage and width**, rewarding narrow intervals and penalizing poor coverage [^6]:

\[
\text{CWC} = (1 - \text{Mean Width Score}) \times \exp\left(-\eta \times (\text{Coverage Score} - (1-\alpha))^2\right)
\]

### Mean Winkler Interval Score (MWI)

Combines interval width with a **penalty for non-coverage** [^8]:

\[
\text{MWI Score} = \frac{1}{n} \sum_{i=1}^{n} \left[(\hat{y}^{\text{up}}_{i} - \hat{y}^{\text{low}}_{i}) + \frac{2}{\alpha} \max(0, |y_{i} - \hat{y}^{\text{boundary}}_{i}|)\right]
\]

---

## 2. Calibration Metrics

### Expected Calibration Error (ECE)

Measures the difference between **predicted confidence levels and actual accuracy**:

\[
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
\]

where:

- \(\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i\) — accuracy within bin \(m\)
- \(\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{f}(x_i)\) — average confidence in bin \(m\)

!!! tip
    The lower the ECE, the better the calibration.

---

### Top-Label ECE

Extends ECE to **multi-class** settings, focusing on calibration of the **most confident prediction** (top-label):

\[
\text{Top-Label ECE} = \frac{1}{L} \sum_{j=1}^L \sum_{i=1}^B \frac{|B_{i,j}|}{n_j} \left| \text{acc}(B_{i,j}) - \text{conf}(B_{i,j}) \right|
\]

where:

- \(L\) = number of unique labels
- \(B_{i,j}\) = indices in bin \(i\) for label \(j\)
- \(n_j\) = total samples for label \(j\)

---

### Cumulative Differences

Calculates the **cumulative differences** between sorted true values and prediction scores [^2]:

\[
\text{Cumulative Differences} = \frac{1}{n} \sum_{i=1}^{n} (y_{\sigma_1(i)} - \hat{y}_{\sigma_2(i)})
\]

---

### Kolmogorov-Smirnov Statistics

Tests whether the **calibration curve** deviates significantly from the ideal diagonal line.

### Kuiper Statistics

Similar to KS but captures **both positive and negative deviations** from perfect calibration.

### Spiegelhalter Statistics

Tests the **overall calibration** of predicted probabilities.

---

## References

[^1]: Angelopoulos, A. N., et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR 2021.
[^2]: Arrieta-Ibarra I, et al. "Metrics of calibration for probabilistic predictions." JMLR 23(1), 2022.
[^4]: Gretton, A., et al. "A Kernel Two-Sample Test." JMLR, 2012.
[^6]: Khosravi, A., et al. "Comprehensive Review of Neural Network-Based Prediction Intervals." IEEE Trans. Neural Netw., 2011.
[^8]: Winkler, R. L. "A Decision-Theoretic Approach to Interval Estimation." JASA, 1972.
