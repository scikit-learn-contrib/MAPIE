##########
User Guide
##########

Introduction
============

1. Code examples
================

2. Algorithms
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

- The conformalized quantile regression method allows for more adaptiveness on the prediction 
  intervals which becomes key when faced with heteroscedastic data.

- If the "exchangeability hypothesis" is not valid, typically for time series,
  use EnbPI, and update the residuals each time new observations are available.

- The absolute residual score is the basic conformity score and gives constant intervals. It is the one used by default by :class:`mapie.regression.MapieRegressor`.
- The gamma conformity score adds a notion of adaptivity by giving intervals of different sizes
  and is proportional to the uncertainty.
- The residual normalized score is a conformity score that requires an additional model
  to learn the residuals of the model from :math:`X`. It gives very adaptive intervals
  without specific assumptions on the data.

The table below summarizes the key features of each method by focusing on the obtained coverages and the
computational cost. :math:`n`, :math:`n_{\rm test}`, and :math:`K` are the number of training samples,
test samples, and cross-validated folds, respectively.

.. csv-table:: Key features of MAPIE methods (adapted from [1])*.
   :file: images/comp-methods.csv
   :header-rows: 1

.. [*] Here, the training and evaluation costs correspond to the computational time of the MAPIE ``.fit()`` and ``.predict()`` methods.

3. Advanced examples
====================