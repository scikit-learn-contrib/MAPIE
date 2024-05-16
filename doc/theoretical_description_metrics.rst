.. title:: Theoretical Description Metrics : contents

.. _theoretical_description_metrics:

=======================
Theoretical Description
=======================

This document provides detailed descriptions of various metrics used to evaluate the performance of predictive models, particularly focusing on their ability to estimate uncertainties and calibrate predictions accurately.


1. General metrics
==================

Regression Coverage Score
-------------------------

The **Regression Coverage Score** calculates the fraction of true outcomes that fall within the provided prediction intervals. 

.. math::

   RCS = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(\hat y^{\text{low}}_{i} \leq y_{i} \leq \hat y^{\text{up}}_{i})

where:

- :math:`n` is the number of samples,
- :math:`y_{i}` is the true value for the :math:`i`-th sample,
- :math:`\hat y^{\text{low}}_{i}` and :math:`\hat y^{\text{up}}_{i}` are the lower and upper bounds of the prediction intervals, respectively.


Regression Mean Width Score
---------------------------

The **Regression Mean Width Score** assesses the average width of the prediction intervals provided by the model.

.. math::

   \text{Mean Width} = \frac{1}{n} \sum_{i=1}^{n} (\hat y^{\text{up}}_{i} - \hat y^{\text{low}}_{i})


Classification Coverage Score
-----------------------------

The **Classification Coverage Score** measures how often the true class labels fall within the predicted sets.

.. math::

   CCS = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(y_{i} \in \hat C(x_{i}))

Here, :math:`\hat C(x_{i})` represents the set of predicted labels that could possibly contain the true label for the :math:`i`-th observation :math:`x_{i}`.


Classification Mean Width Score
-------------------------------

For classification tasks, the **Classification Mean Width Score** calculates the average size of the prediction sets across all samples.

.. math::

   \text{Mean Width} = \frac{1}{n} \sum_{i=1}^{n} |\hat C_{x_i}|

where :math:`|\hat C_{x_i}|` denotes the number of classes included in the prediction set for sample :math:`i`.


Size-Stratified Coverage (SSC)
-------------------------------

**Size-Stratified Coverage (SSC)** evaluates how the size of prediction sets or intervals affects their ability to cover the true outcomes [1]. It's calculated separately for classification and regression:

**Regression:**

.. math::

   \text{SSC}_{\text{regression}} = \sum_{k=1}^{K} \left( \frac{1}{|I_k|} \sum_{i \in I_k} \mathbf{1}(y_{\text{pred, low}}^{(i)} \leq y_{\text{true}}^{(i)} \leq y_{\text{pred, up}}^{(i)}) \right)

**Classification:**

.. math::

   \text{SSC}_{\text{classification}} = \sum_{k=1}^{K} \left( \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{1}(y_{\text{true}}^{(i)} \in \text{Set}_{\text{pred}}^{(i)}) \right)

where:

- :math:`K` is the number of distinct size groups,
- :math:`I_k` and :math:`S_k` are the indices of samples whose prediction intervals or sets belong to the :math:`k`-th size group.


Hilbert-Schmidt Independence Criterion (HSIC)
----------------------------------------------

**Hilbert-Schmidt Independence Criterion (HSIC)** is a non-parametric measure of independence between two variables, applied here to test the independence of interval sizes from their coverage indicators [4].

.. math::

   \text{HSIC} = \operatorname{trace}(\mathbf{H} \mathbf{K} \mathbf{H} \mathbf{L})

where:

- :math:`\mathbf{K}` and :math:`\mathbf{L}` are the kernel matrices representing the interval sizes and coverage indicators, respectively.
- :math:`\mathbf{H}` is the centering matrix, :math:`\mathbf{H} = \mathbf{I} - \frac{1}{n} \mathbf{11}^\top`.

This measure is crucial for determining whether certain sizes of prediction intervals are systematically more or less likely to contain the true values, which can highlight biases in interval-based predictions.


Coverage Width-Based Criterion (CWC)
------------------------------------

The **Coverage Width-Based Criterion (CWC)** evaluates prediction intervals by balancing their empirical coverage and width. It is designed to both reward narrow intervals and penalize those that do not achieve a specified coverage probability [6].

.. math::

   \text{CWC} = (1 - \text{Mean Width Score}) \times \exp\left(-\eta \times (\text{Coverage Score} - (1-\alpha))^2\right)



Regression MWI Score
--------------------

The **MWI (Mean Winkler Interval) Score** evaluates prediction intervals by combining their width with a penalty for intervals that do not contain the observation [8, 10].

.. math::

   \text{MWI Score} = \frac{1}{n} \sum_{i=1}^{n} (\hat y^{\text{up}}_{i} - \hat y^{\text{low}}_{i}) + \frac{2}{\alpha} \sum_{i=1}^{n} \max(0, |y_{i} - \hat y^{\text{boundary}}_{i}|)

where :math:`\hat y^{\text{boundary}}_{i}` is the nearest interval boundary not containing :math:`y_{i}`, and :math:`\alpha` is the significance level.



2. Calibration metrics
======================

Expected Calibration Error (ECE)
--------------------------------

**Expected Calibration Error (ECE)** measures the difference between predicted probabilities of a model and the actual outcomes, across different bins of predicted probabilities [7].

.. math::

   \text{ECE} = \sum_{b=1}^{B} \frac{n_b}{n} | \text{acc}(b) - \text{conf}(b) |

where:

- :math:`B` is the total number of bins,
- :math:`n_b` is the number of samples in bin :math:`b`,
- :math:`\text{acc}(b)` is the accuracy within bin :math:`b`,
- :math:`\text{conf}(b)` is the mean predicted probability in bin :math:`b`.


Top-Label Expected Calibration Error (Top-Label ECE)
----------------------------------------------------

**Top-Label ECE** focuses on the class predicted with the highest confidence for each sample, assessing whether these top-predicted confidences align well with actual outcomes. It is calculated by dividing the confidence score range into bins and comparing the mean confidence against empirical accuracy within these bins [5].

.. math::

   \text{Top-Label ECE} = \sum_{b=1}^{B} \frac{n_b}{n} \left| \text{acc}_b - \text{conf}_b \right|

where:

- :math:`n` is the total number of samples,
- :math:`n_b` is the number of samples in bin :math:`b`,
- :math:`\text{acc}_b` is the empirical accuracy in bin :math:`b`,
- :math:`\text{conf}_b` is the average confidence of the top label in bin :math:`b`.

This metric is especially useful in multi-class classification to ensure that the model is neither overconfident nor underconfident in its predictions.


Cumulative Differences
----------------------

**Cumulative Differences** calculates the cumulative differences between sorted true values and prediction scores, helping to understand how well the prediction scores correspond to the actual outcomes when both are ordered by the score [2].

.. math::

   \text{Cumulative Differences} = \frac{1}{n} \sum_{i=1}^{n} (y_{\sigma_1(i)} - \hat y_{\sigma_2(i)})

where:

- :math:`\sigma_1` is the permutation which sorts all the true values.
- :math:`\sigma_2` is the permutation which sorts all the predicted values.


Kolmogorov-Smirnov Statistic for Calibration
--------------------------------------------

This statistic measures the maximum absolute deviation between the empirical cumulative distribution function (ECDF) of observed outcomes and predicted probabilities [2, 3, 11].

.. math::

   \text{KS Statistic} = \sup_x |F_n(x) - S_n(x)|

where :math:`F_n(x)` is the ECDF of the predicted probabilities and :math:`S_n(x)` is the ECDF of the observed outcomes.


Kuiper's Statistic
------------------

**Kuiper's Statistic** considers both the maximum deviation above and below the mean cumulative difference, making it more sensitive to deviations at the tails of the distribution [2, 3, 11].

.. math::

   \text{Kuiper's Statistic} = \max(F_n(x) - S_n(x)) + \max(S_n(x) - F_n(x))


Spiegelhalter’s Test
--------------------

**Spiegelhalter’s Test** assesses the calibration of binary predictions based on a transformation of the Brier score [9].

.. math::

   \text{Spiegelhalter's Statistic} = \frac{\sum_{i=1}^n (y_i - \hat y_i)(1 - 2\hat y_i)}{\sqrt{\sum_{i=1}^n (1 - 2 \hat y_i)^2 \hat y_i (1 - \hat y_i)}}



References
==========

[1] Angelopoulos, A. N., & Bates, S. (2021).
A gentle introduction to conformal prediction and
distribution-free uncertainty quantification.
arXiv preprint arXiv:2107.07511.

[2] Arrieta-Ibarra I, Gujral P, Tannen J, Tygert M, Xu C.
Metrics of calibration for probabilistic predictions.
The Journal of Machine Learning Research. 2022 Jan 1;23(1):15886-940.

[3] D. A. Darling. A. J. F. Siegert.
The First Passage Problem for a Continuous Markov Process.
Ann. Math. Statist. 24 (4) 624 - 639, December, 1953.

[4] Feldman, S., Bates, S., & Romano, Y. (2021).
Improving conditional coverage via orthogonal quantile regression.
Advances in Neural Information Processing Systems, 34, 2060-2071.

[5] Gupta, Chirag, and Aaditya K. Ramdas.
"Top-label calibration and multiclass-to-binary reductions."
arXiv preprint arXiv:2107.08353 (2021).

[6] Khosravi, Abbas, Saeid Nahavandi, and Doug Creighton.
"Construction of optimal prediction intervals for load forecasting
problems."
IEEE Transactions on Power Systems 25.3 (2010): 1496-1503.

[7] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
"Obtaining well calibrated probabilities using bayesian binning."
Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.

[8] Robert L. Winkler
"A Decision-Theoretic Approach to Interval Estimation",
Journal of the American Statistical Association,
volume 67, pages 187-191 (1972)
(https://doi.org/10.1080/01621459.1972.10481224)

[9] Spiegelhalter DJ.
Probabilistic prediction in patient management and clinical trials.
Statistics in medicine.
1986 Sep;5(5):421-33.

[10] Tilmann Gneiting and Adrian E Raftery
"Strictly Proper Scoring Rules, Prediction, and Estimation",
Journal of the American Statistical Association,
volume 102, pages 359-378 (2007)
(https://doi.org/10.1198/016214506000001437) (Section 6.2)

[11] Tygert M.
Calibration of P-values for calibration and for deviation
of a subpopulation from the full population.
arXiv preprint arXiv:2202.00100.2022 Jan 31.
