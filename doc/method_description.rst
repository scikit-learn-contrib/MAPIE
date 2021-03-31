.. title:: Theoretical Description : contents

.. _method_description:

=======================
Theoretical Description
=======================

This module uses various resampling methods based on the jackknife strategy
recently introduced by Foygel-Barber et al. (2020). 
They allow the user to estimate robust prediction intervals with any kind of
Machine-Learning model for regression purposes. 
We give here a brief theoretical description of the methods included in the module.
The figure below, adapted from Fig. 1 of Foygel-Barber (2020) illustrates the the methods by emphasing their main differences.

.. image:: images/jackknife_cut.png
   :width: 800

Before describing the methods, let's briefly present the mathematical setting.
For a regression problem in a standard independent and identically distributed
(i.i.d) case, our training data :math:`(X, Y) = \{(x_1, y_1), \ldots, (x_n, y_n)\}`
has an unknown distribution :math:`P_{X, Y}`. We can assume that :math:`Y = = \mu(X)+\epsilon`
where :math:`\mu` is the model function we want to determine and
:math:`\epsilon_i \sim P_{Y \vert X}` is the noise. 
Given some target quantile :math:`\alpha` or associated target coverage level :math:`1-\alpha`,
we aim at constructing a prediction interval :math:`\hat{C}_{n, \alpha}` for a new
feature vector :math:`X_{n+1}` such that 

.. math:: 
    P \{Y_{n+1} \in \hat{C}_{n, \alpha}(X_{n+1}) \} \geq 1 - \alpha

- **"Naive" method**

The so-called naive method computes the residuals of the training data to estimate the 
typical error obtained on a new test data point. 
The prediction interval is therefore given by the prediction obtained by the 
model trained on the entire training set :math:`\pm` the quantiles of the 
residuals of the training set:
    
.. math:: \hat{\mu}(X_{n+1}) \pm ((1-\alpha) \textrm{quantile of} |Y_1-\hat{\mu}(X_1)|, ..., |Y_n-\hat{\mu}(X_n)|)

or

.. math:: \hat{C}_{n, \alpha}^{\rm naive}(X_{n+1}) = \hat{\mu}(X_n+1) \pm \hat{q}_{n, \alpha}^+{|Y_i-\hat{\mu}(X_i)|}

with :math:`\hat{q}_{n, \alpha}^+` is the :math:`(1-\alpha)` quantile of the distribution.

Since this method estimates the residuals only on the training set, it tends to be optimistic and
under-estimates the width of prediction intervals because of a potential overfit. 
As a result, the probability that a new point lies in the interval given by the 
naive method would be lower than the target level :math:`(1-\alpha)`.


- **Jackknife**
  
The *standard* Jackknife method is based on the construction of a set of *leave-one-out* (l-o-o) models. 
Estimating the prediction intervals is carried out in three main steps:

    - For each instance *i = 1, ..., n* of the training set, we fit the regression function
    :math:`\hat{\mu}_{-i}` on the entire training set with the :math:`i^{th}` point removed, resulting in *n* l-o-o models.
    - The corresponding l-o-o residual is computed for each :math:`i^{th}` point :math:`|Y_i - \hat{\mu}_{-i}(X_i)|`.
    - We fit the regression function :math:`\hat{\mu}` on the entire training set and we compute
    the prediction interval using the computed l-o-o residuals. 
  
.. math:: \hat{\mu}(X_{n+1}) \pm ((1-\alpha) \textrm{ quantile of } |Y_1-\hat{\mu}_{-1}(X_1)|, ..., |Y_n-\hat{\mu}_{-n}(X_n)|)

The resulting confidence interval can therefore be summarized as follows

.. math:: \hat{C}_{n, \alpha}^{\rm jackknife}(X_{n+1}) = [ \hat{q}_{n, \alpha}^-\{\hat{\mu}(X_{n+1}) - R_i^{\rm LOO} \}, \hat{q}_{n, \alpha}^+\{\hat{\mu}(X_{n+1}) + R_i^{\rm LOO} \}] 

where

.. math:: R_i^{\rm LOO} = |Y_i - \hat{\mu}_{-i}(X_i)|

is the *l-o-o* residual.

This method avoids the overfitting problem but can loose its predictive 
cover when :math:`\hat{\mu}` becomes unstable, for example when the 
sample size is closed to the number of features (as seen in the example). 


- **Jackknife+**

Unlike the standard Jackknife method which estimates an prediction interval centered 
around the prediction of the model trained on the entire dataset, the so-called Jackknife+ 
method uses each l-o-o prediction on the new test point to take the variability of the 
regression function into account.

.. math:: \hat{C}_{n, \alpha}^{\rm jackknife+}(X_{n+1}) = [ \hat{q}_{n, \alpha}^-\{\hat{\mu}_{-i}(X_{n+1}) - R_I^{\rm LOO} \}, \hat{q}_{n, \alpha}^+\{\hat{\mu}_{-i}(X_{n+1}) + R_I^{\rm LOO} \}] 

As described in Barber et al. (2020), this method garantees a higher stability 
with a coverage level of :math:`1-2\alpha` for a target coverage level of :math:`1-\alpha`.

However, the Jackknife and Jackknife+ methods are computationally heavy since 
they require to run as many simulations as the number of training points, and is prohibitive 
for a typical data science usecase. 


- **CV+**

In order to reduce the computational time, one can adopt a cross-validation approach instead of the leave-one-out approach associated with the jackknife strategy, called the CV+ method.

By analogy with the jackknife+ method, estimating the prediction intervals with CV+ is performed in four main steps:

- We split the training set into *K* disjoint subsets :math:`S_1, S_2, ..., S_k` of equal size. 
- *K* regression functions :math:`\hat{\mu}_{-Sk}` are fitted on the training set with the corresponding :math:`k^{th}` fold removed.
- The corresponding *out-of-fold* residual is computed for each :math:`i^{th}` point :math:`|Y_i - \hat{\mu}_{-Sk(i)}(X_i)|` where *k(i)* is the fold containing *i*.
- Similar to the jackknife+, the regression functions :math:`\hat{\mu}_{-Sk(i)}(X_i)` are used to estimate the prediction intervals. 

As noted by Barber et al. (2020), the Jackknife+ can be viewed as a special case of the CV+ in which :math:`K = n`. 
In practice, this method results in slightly wider prediction intervals and is therefore more conservative, but gives 
a reasonable compromise for large datasets where the Jacknife+ method is unfeasible.