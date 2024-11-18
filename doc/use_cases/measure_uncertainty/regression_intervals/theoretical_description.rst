#######################
Conformal regression : theoretical description
#######################

1. Overview
============

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

All the methods below are described with the absolute residual conformity score for simplicity
but other conformity scores are implemented in MAPIE (see :doc:`theoretical_description_conformity_scores`).

1.1. The "Naive" method
----------------------

The so-called naive method computes the residuals of the training data to estimate the 
typical error obtained on a new test data point. 
The prediction interval is therefore given by the prediction obtained by the 
model trained on the entire training set :math:`\pm` the quantiles of the 
conformity scores of the same training set:
    
.. math:: \hat{\mu}(X_{n+1}) \pm ((1-\alpha) \textrm{quantile of} |Y_1-\hat{\mu}(X_1)|, ..., |Y_n-\hat{\mu}(X_n)|)

or

.. math:: \hat{C}_{n, \alpha}^{\rm naive}(X_{n+1}) = \hat{\mu}(X_{n+1}) \pm \hat{q}_{n, \alpha}^+{|Y_i-\hat{\mu}(X_i)|}

where :math:`\hat{q}_{n, \alpha}^+` is the :math:`(1-\alpha)` quantile of the distribution.

Since this method estimates the conformity scores only on the training set, it tends to be too 
optimistic and underestimates the width of prediction intervals because of a potential overfit. 
As a result, the probability that a new point lies in the interval given by the 
naive method would be lower than the target level :math:`(1-\alpha)`.

The figure below illustrates the naive method. 

.. image:: images/jackknife_naive.png
   :width: 200
   :align: center

1.2. The split method
----------------------

The so-called split method computes the residuals of a calibration dataset to estimate the 
typical error obtained on a new test data point. 
The prediction interval is therefore given by the prediction obtained by the 
model trained on the training set :math:`\pm` the quantiles of the 
conformity scores of the calibration set:
    
.. math:: \hat{\mu}(X_{n+1}) \pm ((1-\alpha) \textrm{quantile of} |Y_1-\hat{\mu}(X_1)|, ..., |Y_n-\hat{\mu}(X_n)|)

or

.. math:: \hat{C}_{n, \alpha}^{\rm split}(X_{n+1}) = \hat{\mu}(X_{n+1}) \pm \hat{q}_{n, \alpha}^+{|Y_i-\hat{\mu}(X_i)|}

where :math:`\hat{q}_{n, \alpha}^+` is the :math:`(1-\alpha)` quantile of the distribution.

Since this method estimates the conformity scores only on a calibration set, one must have enough
observations to split its original dataset into train and calibration as mentioned in [5]. We can
notice that this method is very similar to the naive one, the only difference being that the conformity
scores are not computed on the calibration set. Moreover, this method will always give prediction intervals
with a constant width.
  

1.3. The jackknife method
----------------------

The *standard* jackknife method is based on the construction of a set of 
*leave-one-out* models. 
Estimating the prediction intervals is carried out in three main steps:

- For each instance *i = 1, ..., n* of the training set, we fit the regression function
  :math:`\hat{\mu}_{-i}` on the entire training set with the :math:`i^{th}` point removed,
  resulting in *n* leave-one-out models.

- The corresponding leave-one-out conformity score is computed for each :math:`i^{th}` point
  :math:`|Y_i - \hat{\mu}_{-i}(X_i)|`.

- We fit the regression function :math:`\hat{\mu}` on the entire training set and we compute
  the prediction interval using the computed leave-one-out conformity scores:
  
.. math:: \hat{\mu}(X_{n+1}) \pm ((1-\alpha) \textrm{ quantile of } |Y_1-\hat{\mu}_{-1}(X_1)|, ..., |Y_n-\hat{\mu}_{-n}(X_n)|)

The resulting confidence interval can therefore be summarized as follows

.. math:: \hat{C}_{n, \alpha}^{\rm jackknife}(X_{n+1}) = [ \hat{q}_{n, \alpha}^-\{\hat{\mu}(X_{n+1}) - R_i^{\rm LOO} \}, \hat{q}_{n, \alpha}^+\{\hat{\mu}(X_{n+1}) + R_i^{\rm LOO} \}] 

where

.. math:: R_i^{\rm LOO} = |Y_i - \hat{\mu}_{-i}(X_i)|

is the *leave-one-out* conformity score.

This method avoids the overfitting problem but can lose its predictive 
cover when :math:`\hat{\mu}` becomes unstable, for example, when the 
sample size is close to the number of features
(as seen in the "Reproducing the simulations from Foygel-Barber et al. (2020)" example). 


1.4. The jackknife+ method
----------------------

Unlike the standard jackknife method which estimates a prediction interval centered 
around the prediction of the model trained on the entire dataset, the so-called jackknife+ 
method uses each leave-one-out prediction on the new test point to take the variability of the 
regression function into account.
The resulting confidence interval can therefore be summarized as follows

.. math:: \hat{C}_{n, \alpha}^{\rm jackknife+}(X_{n+1}) = [ \hat{q}_{n, \alpha}^-\{\hat{\mu}_{-i}(X_{n+1}) - R_i^{\rm LOO} \}, \hat{q}_{n, \alpha}^+\{\hat{\mu}_{-i}(X_{n+1}) + R_i^{\rm LOO} \}] 

As described in [1], this method guarantees a higher stability 
with a coverage level of :math:`1-2\alpha` for a target coverage level of :math:`1-\alpha`,
without any *a priori* assumption on the distribution of the data :math:`(X, Y)`
nor on the predictive model.

1.5. The jackknife-minmax method
----------------------

The jackknife-minmax method offers a slightly more conservative alternative since it uses 
the minimal and maximal values of the leave-one-out predictions to compute the prediction intervals.
The estimated prediction intervals can be defined as follows

.. math:: 

    \hat{C}_{n, \alpha}^{\rm jackknife-mm}(X_{n+1}) = 
    [\min \hat{\mu}_{-i}(X_{n+1}) - \hat{q}_{n, \alpha}^+\{R_I^{\rm LOO} \}, 
    \max \hat{\mu}_{-i}(X_{n+1}) + \hat{q}_{n, \alpha}^+\{R_I^{\rm LOO} \}] 

As justified by [1], this method guarantees a coverage level of 
:math:`1-\alpha` for a target coverage level of :math:`1-\alpha`.

The figure below, adapted from Fig. 1 of [1], illustrates the three jackknife
methods and emphasizes their main differences.

.. image:: images/jackknife_jackknife.png
   :width: 800

However, the jackknife, jackknife+ and jackknife-minmax methods are computationally heavy since 
they require to run as many simulations as the number of training points, which is prohibitive 
for a typical data science use case. 


1.6. The CV+ method
----------------------

In order to reduce the computational time, one can adopt a cross-validation approach
instead of a leave-one-out approach, called the CV+ method.

By analogy with the jackknife+ method, estimating the prediction intervals with CV+
is performed in four main steps:

- We split the training set into *K* disjoint subsets :math:`S_1, S_2, ..., S_K` of equal size. 
  
- *K* regression functions :math:`\hat{\mu}_{-S_k}` are fitted on the training set with the 
  corresponding :math:`k^{th}` fold removed.

- The corresponding *out-of-fold* conformity score is computed for each :math:`i^{th}` point 
  :math:`|Y_i - \hat{\mu}_{-S_{k(i)}}(X_i)|` where *k(i)* is the fold containing *i*.

- Similar to the jackknife+, the regression functions :math:`\hat{\mu}_{-S_{k(i)}}(X_i)` 
  are used to estimate the prediction intervals. 

As for jackknife+, this method guarantees a coverage level higher than :math:`1-2\alpha` 
for a target coverage level of :math:`1-\alpha`, without any *a priori* assumption on 
the distribution of the data.
As noted by [1], the jackknife+ can be viewed as a special case of the CV+ 
in which :math:`K = n`. 
In practice, this method results in slightly wider prediction intervals and is therefore 
more conservative, but gives a reasonable compromise for large datasets when the Jacknife+ 
method is unfeasible.


1.7. The CV and CV-minmax methods
----------------------

By analogy with the standard jackknife and jackknife-minmax methods, the CV and CV-minmax approaches
are also included in MAPIE. As for the CV+ method, they rely on out-of-fold regression models that
are used to compute the prediction intervals but using the equations given in the jackknife and
jackknife-minmax sections.  


The figure below, adapted from Fig. 1 of [1], illustrates the three CV
methods and emphasizes their main differences.

.. image:: images/jackknife_cv.png
   :width: 800


1.8. The jackknife+-after-bootstrap method
----------------------

In order to reduce the computational time, and get more robust predictions, 
one can adopt a bootstrap approach instead of a leave-one-out approach, called 
the jackknife+-after-bootstrap method, offered by Kim and al. [2]. Intuitively,
this method uses ensemble methodology to calculate the :math:`i^{\text{th}}`
aggregated prediction and residual by only taking subsets in which the
:math:`i^{\text{th}}` observation is not used to fit the estimator.

By analogy with the CV+ method, estimating the prediction intervals with 
jackknife+-after-bootstrap is performed in four main steps:

- We resample the training set with replacement (bootstrap) :math:`K` times,
  and thus we get the (non-disjoint) bootstraps :math:`B_{1},..., B_{K}` of equal size.


- :math:`K` regressions functions :math:`\hat{\mu}_{B_{k}}` are then fitted on 
  the bootstraps :math:`(B_{k})`, and the predictions on the complementary sets 
  :math:`(B_k^c)` are computed.


- These predictions are aggregated according to a given aggregation function 
  :math:`{\rm agg}`, typically :math:`{\rm mean}` or :math:`{\rm median}`, and the conformity scores 
  :math:`|Y_j - {\rm agg}(\hat{\mu}(B_{K(j)}(X_j)))|` are computed for each :math:`X_j`
  (with :math:`K(j)` the boostraps not containing :math:`X_j`).

 
- The sets :math:`\{\rm agg(\hat{\mu}_{K(j)}(X_i)) + r_j\}` (where :math:`j` indexes  
  the training set) are used to estimate the prediction intervals.


As for jackknife+, this method guarantees a coverage level higher than 
:math:`1 - 2\alpha` for a target coverage level of :math:`1 - \alpha`, without 
any a priori assumption on the distribution of the data. 
In practice, this method results in wider prediction intervals, when the 
uncertainty is higher than :math:`CV+`, because the models' prediction spread 
is then higher.


1.9. The Conformalized Quantile Regression (CQR) Method
----------------------

The conformalized quantile regression (CQR) method allows for better interval widths with
heteroscedastic data. It uses quantile regressors with different quantile values to estimate
the prediction bounds. The residuals of these methods are used to create the guaranteed
coverage value.

Notations and Definitions
-------------------------
- :math:`\mathcal{I}_1` is the set of indices of the data in the training set.
- :math:`\mathcal{I}_2` is the set of indices of the data in the calibration set.
- :math:`\hat{q}_{\alpha_{\text{low}}}`: Lower quantile model trained on :math:`{(X_i, Y_i) : i \in \mathcal{I}_1}`.
- :math:`\hat{q}_{\alpha_{\text{high}}}`: Upper quantile model trained on :math:`{(X_i, Y_i) : i \in \mathcal{I}_1}`.
- :math:`E_i`: Residuals for the i-th sample in the calibration set.
- :math:`E_{\text{low}}`: Residuals from the lower quantile model.
- :math:`E_{\text{high}}`: Residuals from the upper quantile model.
- :math:`Q_{1-\alpha}(E, \mathcal{I}_2)`: The :math:`(1-\alpha)(1+1/|\mathcal{I}_2|)`-th empirical quantile of the set :math:`{E_i : i \in \mathcal{I}_2}`.

Mathematical Formulation
^^^^^^^^^^^^
The prediction interval :math:`\hat{C}_{n, \alpha}^{\text{CQR}}(X_{n+1})` for a new sample :math:`X_{n+1}` is given by:

.. math::

    \hat{C}_{n, \alpha}^{\text{CQR}}(X_{n+1}) = 
    [\hat{q}_{\alpha_{\text{lo}}}(X_{n+1}) - Q_{1-\alpha}(E_{\text{low}}, \mathcal{I}_2),
    \hat{q}_{\alpha_{\text{hi}}}(X_{n+1}) + Q_{1-\alpha}(E_{\text{high}}, \mathcal{I}_2)]

Where:

- :math:`\hat{q}_{\alpha_{\text{lo}}}(X_{n+1})` is the predicted lower quantile for the new sample.
- :math:`\hat{q}_{\alpha_{\text{hi}}}(X_{n+1})` is the predicted upper quantile for the new sample.

Note: In the symmetric method, :math:`E_{\text{low}}` and :math:`E_{\text{high}}` sets are no longer distinct. We consider directly the union set :math:`E_{\text{all}} = E_{\text{low}} \cup E_{\text{high}}` and the empirical quantile is then calculated on all the absolute (positive) residuals.

As justified by the literature, this method offers a theoretical guarantee of the target coverage level :math:`1-\alpha`.


1.10. The ensemble batch prediction intervals (EnbPI) method
----------------------

The coverage guarantee offered by the various resampling methods based on the
jackknife strategy, and implemented in MAPIE, are only valid under the "exchangeability
hypothesis". It means that the probability law of data should not change up to
reordering.
This hypothesis is not relevant in many cases, notably for dynamical times series.
That is why a specific class is needed, namely
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor`.

Its implementation looks like the jackknife+-after-bootstrap method. The
leave-one-out (LOO) estimators are approximated thanks to a few boostraps.
However, the confidence intervals are like those of the jackknife method.

.. math::
  \hat{C}_{n, \alpha}^{\rm EnbPI}(X_{n+1}) = [\hat{\mu}_{agg}(X_{n+1}) + \hat{q}_{n, \beta}\{ R_i^{\rm LOO} \}, \hat{\mu}_{agg}(X_{n+1}) + \hat{q}_{n, (1 - \alpha + \beta)}\{ R_i^{\rm LOO} \}]

where :math:`\hat{\mu}_{agg}(X_{n+1})` is the aggregation of the predictions of
the LOO estimators (mean or median), and
:math:`R_i^{\rm LOO} = |Y_i - \hat{\mu}_{-i}(X_{i})|` 
is the residual of the LOO estimator :math:`\hat{\mu}_{-i}` at :math:`X_{i}` [4].

The residuals are no longer considered in absolute values but in relative
values and the width of the confidence intervals are minimized, up to a given gap
between the quantiles' level, optimizing the parameter :math:`\beta`.

Moreover, the residuals are updated during the prediction, each time new observations 
are available. So that the deterioration of predictions, or the increase of
noise level, can be dynamically taken into account.

Finally, the coverage guarantee is no longer absolute but asymptotic up to two
hypotheses:

1. Errors are short-term independent and identically distributed (i.i.d)

2. Estimation quality: there exists a real sequence :math:`(\delta_T)_{T > 0}`
   that converges to zero such that

.. math::
    \frac{1}{T}\sum_1^T(\hat{\mu}_{-t}(x_t) - \mu(x_t))^2 < \delta_T^2

The coverage level depends on the size of the training set and on 
:math:`(\delta_T)_{T > 0}`.

Be careful: the bigger the training set, the better the covering guarantee
for the point following the training set. However, if the residuals are
updated gradually, but the model is not refitted, the bigger the training set
is, the slower the update of the residuals is effective. Therefore there is a
compromise to make on the number of training samples to fit the model and
update the prediction intervals.

References
----------------------

[1] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani.
"Predictive inference with the jackknife+." Ann. Statist., 49(1):486–507, February 2021.

[2] Byol Kim, Chen Xu, and Rina Foygel Barber.
"Predictive Inference Is Free with the Jackknife+-after-Bootstrap."
34th Conference on Neural Information Processing Systems (NeurIPS 2020).

[3] Yaniv Romano, Evan Patterson, Emmanuel J. Candès.
"Conformalized Quantile Regression." Advances in neural information processing systems 32 (2019).

[4] Chen Xu and Yao Xie. 
"Conformal Prediction Interval for Dynamic Time-Series."
International Conference on Machine Learning (ICML, 2021).

[5] Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J Tibshirani, and Larry Wasserman.
"Distribution-free predictive inference for regression". 
Journal of the American Statistical Association, 113(523):1094–1111, 2018.

2. Conformity scores
=====================

The :class:`mapie.conformity_scores.ConformityScore` class implements various
methods to compute conformity scores for regression.
We give here a brief theoretical description of the scores included in the module.
Note that it is possible for the user to create any conformal scores that are not 
already included in MAPIE by inheriting this class.

Before describing the methods, let's briefly present the mathematical setting.
With conformal predictions, we want to transform a heuristic notion of uncertainty
from a model into a rigorous one, and the first step to do it is to choose a conformal score.
The only requirement for the score function :math:`s(X, Y) \in \mathbb{R}` is
that larger scores should encode worse agreement between :math:`X` and :math:`Y`. [1]

There are two types of scores : the symmetric and asymmetric ones.
The symmetric property defines the way of computing the quantile of the conformity
scores when calculating the interval's bounds. If a score is symmetrical two
quantiles will be computed : one on the right side of the distribution
and the other on the left side.

2.1. The absolute residual score
------------------------------

The absolute residual score (:class:`mapie.conformity_scores.AbsoluteConformityScore`)
is the simplest and most commonly used conformal score, it translates the error
of the model : in regression, it is called the residual.

.. math:: |Y-\hat{\mu}(X)|

The intervals of prediction's bounds are then computed from the following formula :

.. math:: [\hat{\mu}(X) - q(s), \hat{\mu}(X) + q(s)]

Where :math:`q(s)` is the :math:`(1-\alpha)` quantile of the conformity scores.
(see :doc:`theoretical_description_regression` for more details).

With this score, the intervals of predictions will be constant over the whole dataset.
This score is by default symmetric (*see above for definition*).

2.2. The gamma score
------------------

The gamma score [2] (:class:`mapie.conformity_scores.GammaConformityScore`) adds a
notion of adaptivity with the normalization of the residuals by the predictions.

.. math:: \frac{|Y-\hat{\mu}(X)|}{\hat{\mu}(X)}

It computes adaptive intervals : intervals of different size on each example, with
the following formula  :

.. math:: [\hat{\mu}(X) * (1 - q(s)), \hat{\mu}(X) * (1 + q(s))]

Where :math:`q(s)` is the :math:`(1-\alpha)` quantile of the conformity scores.
(see :doc:`theoretical_description_regression` for more details).

This score is by default asymmetric (*see definition above*).

Compared to the absolute residual score, it allows us to see regions with smaller intervals
than others which are interpreted as regions with more certainty than others.
It is important to note that, this conformity score is inversely proportional to the
order of magnitude of the predictions. Therefore, the uncertainty is proportional to
the order of magnitude of the predictions, implying that this score should be used
in use cases where we want greater uncertainty when the prediction is high.

2.3. The residual normalized score
--------------------------------

The residual normalized score [1] (:class:`mapie.conformity_scores.ResidualNormalisedScore`)
is slightly more complex than the previous scores.
The normalization of the residual is now done by the predictions of an additional model
:math:`\hat\sigma` which learns to predict the base model residuals from :math:`X`.
:math:`\hat\sigma` is trained on :math:`(X, |Y-\hat{\mu}(X)|)` and the formula of the score is:

.. math:: \frac{|Y-\hat{\mu}(X)|}{\hat{\sigma}(X)}

This score provides adaptive intervals : intervals of different sizes in each point
with the following formula :

.. math:: [\hat{\mu}(X) - q(s) * \hat{\sigma}(X), \hat{\mu}(X) + q(s) * \hat{\sigma}(X)]

Where :math:`q(s)` is the :math:`(1-\alpha)` quantile of the conformity scores.
(see :doc:`theoretical_description_regression` for more details).

This score is by default symmetric (*see definition above*). Unlike the scores above,
and due to the additional model required this score can only be used with split methods.

Normalization by the learned residuals from :math:`X` adds to the score a knowledge of
:math:`X` and its similarity to the other examples in the dataset.
Compared to the gamma score, the other adaptive score implemented in MAPIE,
it is not proportional to the uncertainty.

References
----------

[1] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-Free 
Predictive Inference for Regression. Journal of the American Statistical Association, 113(523), 1094–1111. 
Available from https://doi.org/10.1080/01621459.2017.1307116

[2] Cordier, T., Blot, V., Lacombe, L., Morzadec, T., Capitaine, A. &amp; Brunel, N.. (2023).
Flexible and Systematic Uncertainty Estimation with Conformal Prediction via the MAPIE library.
Available from https://proceedings.mlr.press/v204/cordier23a.html.