.. title:: Theoretical Description : contents

.. _theoretical_description_conformity_scores:

=============================================
Theoretical Description for Conformity Scores
=============================================

The :class:`mapie.conformity_scores.ConformityScore` class implements various
methods to compute conformity scores for regression.
We give here a brief theoretical description of the scores included in the module.
Note that it is possible for the user to create any conformal scores that are not 
already included in MAPIE by inheriting this class.

Before describing the methods, let's briefly present the mathematical setting.
With conformal predictions we want to transform an heuristic notion of uncertainty
from a model into a rigorous one, and the first step to do it is to choose a conformal score.
The only requirement for the score function :math:`s(X, Y) \in \mathbb{R}` is
that larger scores should encode worse agreement between :math:`X` and :math:`Y`. [1]

There are two types of scores : the symmetric and asymmetric ones.
The symmetric property defines the way of computing the quantile of the conformity
scores when calculating the interval's bounds. If a score is symmetrical two
quantiles will be computed : one on the right side of the distribution
and the other on the left side.

1. The absolute residual score
==============================

The absolute residual score (:class:`mapie.conformity_scores.AbsoluteResidualScore`)
is the simplest and most commonly used conformal score, it translates the error
of the model : in regression it is called the residual.

.. math:: |Y-\hat{\mu}(X)|

With this score the intervals of predictions will be constant over the whole dataset.
This score is by default symmetric (*see above for definition*).

2. The gamma score
==================

The gamma score (:class:`mapie.conformity_scores.GammaConformityScore`) adds a
notion of adaptivity with the normalization of the residuals by the predictions.

.. math:: \frac{|Y-\hat{\mu}(X)|}{\hat{\mu}(X)}

It computes adaptive intervals : intervals of different size on each example.
This score is by default asymmetric (*see definition above*).

Compared to the absolute residual score, it allows to see regions with smaller intervals
than others which are interpreted as regions with more certainty than others.

3. The conformal residual fitting score
=======================================

The conformal residual fitting score (:class:`mapie.conformity_scores.ConformalizedResidualFittingScore`)
(CRF) is slightly more complex than the previous scores.
The normalization of the residual is now done by the predictions of an additional model
:math:`\sigma` which learns to predict the base model residuals from :math:`X`.
:math:`\sigma` is trained on :math:`(X, |Y-\hat{\mu}(X)|)` and the formula of the score is :

.. math:: \frac{|Y-\hat{\mu}(X)|}{\hat{\sigma}(X)}

This score provides adaptive intervals : intervals of different sizes in each point
and is by default symmetric (*see definition above*). Unlike the scores above, and due to
the additionnal model required this score can only be used with split methods.

Normalisation by the learned residuals from :math:`X` adds to the score a knowledge of
:math:`X` and its similarity to the other examples in the dataset. In fact, using this
score results in even more adaptive intervals. Compared to the gamma score, the other adaptive
score implemented in MAPIE, it maintains relevant interval sizes over the entire dataset
even when there are outliers that could perturb the model. With gamma score, if
there are strong outliers, the intervals over the hole datset are sometimes too large
to be useful.
Therefore, the interpretation of the intervals provided by the conformal residual fitting
score can help to detect outliers.


Key takeaways
=============

- The absolute residual score is the basic conformity score and gives constant intervals.
- The gamma conformity score adds a notion of adaptivity by giving intervals of different sizes,
  but it can give absurd results with strong outliers.
- The conformal residual fitting score is a conformity score that requires an additional model
  to learn the residuals of the model from :math:`X`. It gives very adaptive intervals,
  and their sizes can help in detecting outliers.

References
==========

[1] Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal
prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511.