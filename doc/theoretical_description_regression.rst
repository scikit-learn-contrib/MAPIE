.. title:: Theoretical Description : contents

.. _theoretical_description_regression:

=======================
Theoretical Description
=======================

The :class:`mapie.regression.MapieRegressor` class uses various
resampling methods based on the jackknife strategy
recently introduced by Foygel-Barber et al. (2020) [1]. 
They allow the user to estimate robust prediction intervals with any kind of
machine learning model for regression purposes on single-output data. 
We give here a brief theoretical description of the methods included in the module.

Before describing the methods, let's briefly present the mathematical setting.
For a regression problem in a standard independent and identically distributed
(i.i.d) case, our training data :math:`(X, Y) = \{(x_1, y_1), \ldots, (x_n, y_n)\}`
has an unknown distribution :math:`P_{X, Y}`. We can assume that :math:`Y = \mu(X)+\epsilon`
where :math:`\mu` is the model function we want to determine and
:math:`\epsilon_i \sim P_{Y \vert X}` is the noise. 
Given some target quantile :math:`\alpha` or associated target coverage level :math:`1-\alpha`,
we aim at constructing a prediction interval :math:`\hat{C}_{n, \alpha}` for a new
feature vector :math:`X_{n+1}` such that 

.. math:: 
    P \{Y_{n+1} \in \hat{C}_{n, \alpha}(X_{n+1}) \} \geq 1 - \alpha

1. The "Naive" method
=====================

The so-called naive method computes the residuals of the training data to estimate the 
typical error obtained on a new test data point. 
The prediction interval is therefore given by the prediction obtained by the 
model trained on the entire training set :math:`\pm` the quantiles of the 
residuals of the same training set:
    
.. math:: \hat{\mu}(X_{n+1}) \pm ((1-\alpha) \textrm{quantile of} |Y_1-\hat{\mu}(X_1)|, ..., |Y_n-\hat{\mu}(X_n)|)

or

.. math:: \hat{C}_{n, \alpha}^{\rm naive}(X_{n+1}) = \hat{\mu}(X_{n+1}) \pm \hat{q}_{n, \alpha}^+{|Y_i-\hat{\mu}(X_i)|}

where :math:`\hat{q}_{n, \alpha}^+` is the :math:`(1-\alpha)` quantile of the distribution.

Since this method estimates the residuals only on the training set, it tends to be too 
optimistic and under-estimates the width of prediction intervals because of a potential overfit. 
As a result, the probability that a new point lies in the interval given by the 
naive method would be lower than the target level :math:`(1-\alpha)`.

The figure below illustrates the Naive method. 

.. image:: images/jackknife_naive.png
   :width: 200
   :align: center

2. The jackknife method
=======================

The *standard* jackknife method is based on the construction of a set of 
*leave-one-out* models. 
Estimating the prediction intervals is carried out in three main steps:

- For each instance *i = 1, ..., n* of the training set, we fit the regression function
  :math:`\hat{\mu}_{-i}` on the entire training set with the :math:`i^{th}` point removed,
  resulting in *n* leave-one-out models.

- The corresponding leave-one-out residual is computed for each :math:`i^{th}` point
  :math:`|Y_i - \hat{\mu}_{-i}(X_i)|`.

- We fit the regression function :math:`\hat{\mu}` on the entire training set and we compute
  the prediction interval using the computed leave-one-out residuals:
  
.. math:: \hat{\mu}(X_{n+1}) \pm ((1-\alpha) \textrm{ quantile of } |Y_1-\hat{\mu}_{-1}(X_1)|, ..., |Y_n-\hat{\mu}_{-n}(X_n)|)

The resulting confidence interval can therefore be summarized as follows

.. math:: \hat{C}_{n, \alpha}^{\rm jackknife}(X_{n+1}) = [ \hat{q}_{n, \alpha}^-\{\hat{\mu}(X_{n+1}) - R_i^{\rm LOO} \}, \hat{q}_{n, \alpha}^+\{\hat{\mu}(X_{n+1}) + R_i^{\rm LOO} \}] 

where

.. math:: R_i^{\rm LOO} = |Y_i - \hat{\mu}_{-i}(X_i)|

is the *leave-one-out* residual.

This method avoids the overfitting problem but can lose its predictive 
cover when :math:`\hat{\mu}` becomes unstable, for example when the 
sample size is closed to the number of features
(as seen in the "Reproducing the simulations from Foygel-Barber et al. (2020)" example). 


3. The jackknife+ method
========================

Unlike the standard jackknife method which estimates a prediction interval centered 
around the prediction of the model trained on the entire dataset, the so-called jackknife+ 
method uses each leave-one-out prediction on the new test point to take the variability of the 
regression function into account.
The resulting confidence interval can therefore be summarized as follows

.. math:: \hat{C}_{n, \alpha}^{\rm jackknife+}(X_{n+1}) = [ \hat{q}_{n, \alpha}^-\{\hat{\mu}_{-i}(X_{n+1}) - R_i^{\rm LOO} \}, \hat{q}_{n, \alpha}^+\{\hat{\mu}_{-i}(X_{n+1}) + R_i^{\rm LOO} \}] 

As described in [1], this method garantees a higher stability 
with a coverage level of :math:`1-2\alpha` for a target coverage level of :math:`1-\alpha`,
without any *a priori* assumption on the distribution of the data :math:`(X, Y)`
nor on the predictive model.

4. The jackknife-minmax method
==============================

The jackknife-minmax method offers a slightly more conservative alternative since it uses 
the minimal and maximal values of the leave-one-out predictions to compute the prediction intervals.
The estimated prediction intervals can be defined as follows

.. math:: 

    \hat{C}_{n, \alpha}^{\rm jackknife-mm}(X_{n+1}) = 
    [\min \hat{\mu}_{-i}(X_{n+1}) - \hat{q}_{n, \alpha}^+\{R_I^{\rm LOO} \}, 
    \max \hat{\mu}_{-i}(X_{n+1}) + \hat{q}_{n, \alpha}^+\{R_I^{\rm LOO} \}] 

As justified by [1], this method garantees a coverage level of 
:math:`1-\alpha` for a target coverage level of :math:`1-\alpha`.

The figure below, adapted from Fig. 1 of [1], illustrates the three jackknife
methods and emphasizes their main differences.

.. image:: images/jackknife_jackknife.png
   :width: 800

However, the jackknife, jackknife+ and jackknife-minmax methods are computationally heavy since 
they require to run as many simulations as the number of training points, which is prohibitive 
for a typical data science use case. 


5. The CV+ method
=================

In order to reduce the computational time, one can adopt a cross-validation approach
instead of a leave-one-out approach, called the CV+ method.

By analogy with the jackknife+ method, estimating the prediction intervals with CV+
is performed in four main steps:

- We split the training set into *K* disjoint subsets :math:`S_1, S_2, ..., S_K` of equal size. 
  
- *K* regression functions :math:`\hat{\mu}_{-S_k}` are fitted on the training set with the 
  corresponding :math:`k^{th}` fold removed.

- The corresponding *out-of-fold* residual is computed for each :math:`i^{th}` point 
  :math:`|Y_i - \hat{\mu}_{-S_{k(i)}}(X_i)|` where *k(i)* is the fold containing *i*.

- Similar to the jackknife+, the regression functions :math:`\hat{\mu}_{-S_{k(i)}}(X_i)` 
  are used to estimate the prediction intervals. 

As for jackknife+, this method garantees a coverage level higher than :math:`1-2\alpha` 
for a target coverage level of :math:`1-\alpha`, without any *a priori* assumption on 
the distribution of the data.
As noted by [1], the jackknife+ can be viewed as a special case of the CV+ 
in which :math:`K = n`. 
In practice, this method results in slightly wider prediction intervals and is therefore 
more conservative, but gives a reasonable compromise for large datasets when the Jacknife+ 
method is unfeasible.


6. The CV and CV-minmax methods
===============================

By analogy with the standard jackknife and jackknife-minmax methods, the CV and CV-minmax approaches
are also included in MAPIE. As for the CV+ method, they rely on out-of-fold regression models that
are used to compute the prediction intervals but using the equations given in the jackknife and
jackknife-minmax sections.  


The figure below, adapted from Fig. 1 of [1], illustrates the three CV
methods and emphasizes their main differences.

.. image:: images/jackknife_cv.png
   :width: 800


7. The jackknife+-after-bootstrap method
========================================

In order to reduce the computational time, and get more robust predictions, 
one can adopt a bootstrap approach instead of a leave-one-out approach, called 
the jackknife+-after-bootstrap method, offered by Kim and al. [2].

By analogy with the CV+ method, estimating the prediction intervals with 
jackknife+-after-bootstrap is performed in four main steps:

- We resample the training set with replacement (boostrap) :math:`K` times,
  and thus we get the (non disjoint) bootstraps :math:`B_{1},..., B_{K}` of equal size.


- :math:`K` regressions functions :math:`\hat{\mu}_{B_{k}}` are then fitted on 
  the bootstraps :math:`(B_{k})`, and the predictions on the complementary sets 
  :math:`(B_k^c)` are computed.


- These predictions are aggregated according to a given aggregation function 
  :math:`{\rm agg}`, typically :math:`{\rm mean}` or :math:`{\rm median}`, and the residuals 
  :math:`|Y_j - {\rm agg}(\hat{\mu}(B_{K(j)}(X_j)))|` are computed for each :math:`X_j`
  (with :math:`K(j)` the boostraps not containing :math:`X_j`).

 
- The sets :math:`\{{\rm agg}(\hat{\mu}_{K(j)}(X_i) + r_j\}` (where :math:`j` indexes  
  the training set) are used to estimate the prediction intervals.


As for jackknife+, this method guarantees a coverage level higher than 
:math:`1 - 2\alpha` for a target coverage level of :math:`1 - \alpha`, without 
any a priori assumption on the distribution of the data. 
In practice, this method results in wider prediction intervals, when the 
uncertainty is higher, than :math:`CV+`, because the models' prediction spread 
is then higher.

Key takeaways
=============

- The jackknife+ method introduced by [1] allows the user to easily obtain theoretically guaranteed
  prediction intervals for any kind of sklearn-compatible Machine Learning regressor.

- Since the typical coverage levels estimated by jackknife+ follow very closely the target coverage levels,
  this method should be used when accurate and robust prediction intervals are required.

- For practical applications where :math:`n` is large and/or the computational time of each 
  *leave-one-out* simulation is high, it is advised to adopt the CV+ method, based on *out-of-fold* 
  simulations, or the jackknife+-after-bootstrap method, instead. 
  Indeed, the methods based on the jackknife resampling approach are very cumbersome because they 
  require to run a high number of simulations, equal to the number of training samples :math:`n`.

- Although the CV+ method results in prediction intervals that are slightly larger than for the 
  jackknife+ method, it offers a good compromise between computational time and accurate predictions.

- The jackknife+-after-bootstrap method results in the same computational efficiency, and
  offers a higher sensitivity to epistemic uncertainty.

- The jackknife-minmax and CV-minmax methods are more conservative since they result in higher
  theoretical and practical coverages due to the larger widths of the prediction intervals.
  It is therefore advised to use them when conservative estimates are needed.

The table below summarizes the key features of each method by focusing on the obtained coverages and the
computational cost. :math:`n`, :math:`n_{\rm test}`, and :math:`K` are the number of training samples,
test samples, and cross-validated folds, respectively.

.. csv-table:: Key features of MAPIE methods (adapted from [1])*.
   :file: images/comp-methods.csv
   :header-rows: 1

.. [*] Here, the training and evaluation costs correspond to the computational time of the MAPIE ``.fit()`` and ``.predict()`` methods.


References
==========

[1] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani.
"Predictive inference with the jackknife+." Ann. Statist., 49(1):486–507, February 2021.

[2] Byol Kim, Chen Xu, and Rina Foygel Barber.
"Predictive Inference Is Free with the Jackknife+-after-Bootstrap."
34th Conference on Neural Information Processing Systems (NeurIPS 2020).