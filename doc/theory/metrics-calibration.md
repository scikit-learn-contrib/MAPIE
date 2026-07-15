# Metrics for Calibration — Theoretical Description

---

This document provides detailed descriptions of various metrics used to evaluate the performance of predictive models, particularly focusing on their ability to estimate uncertainties and calibrate predictions accurately.


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

[^2]: Arrieta-Ibarra I, et al. "Metrics of calibration for probabilistic predictions." JMLR 23(1), 2022.
