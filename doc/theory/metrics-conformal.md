# Metrics for Confromal Prediction — Theoretical Description

!!! note "Terminology"
    In theoretical parts of the documentation:

    - `alpha` is equivalent to `1 - confidence_level` — it can be seen as a *risk level*.
    - *calibrate* and *calibration* are equivalent to *conformalize* and *conformalization*.

---

This document provides detailed descriptions of various metrics used to evaluate the performance of predictive models, particularly focusing on their ability to estimate uncertainties and calibrate predictions accurately.


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

## References

[^1]: Angelopoulos, A. N., et al. "Uncertainty Sets for Image Classifiers using Conformal Prediction." ICLR 2021.
[^4]: Gretton, A., et al. "A Kernel Two-Sample Test." JMLR, 2012.
[^6]: Khosravi, A., et al. "Comprehensive Review of Neural Network-Based Prediction Intervals." IEEE Trans. Neural Netw., 2011.
[^8]: Winkler, R. L. "A Decision-Theoretic Approach to Interval Estimation." JASA, 1972.
