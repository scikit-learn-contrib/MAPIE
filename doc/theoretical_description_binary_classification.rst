.. title:: Theoretical Description : contents

.. _theoretical_description_binay_classification:

=======================
Theoretical Description
=======================

For binary classification in the distribution-free setting, there are
different ways of handling uncertainty quantification: calibration,
probabilistic prediction (consisting in estimating confidence intervals or CIs) 
and set prediction (consisting in estimating prediction sets or PSs). Gupta et al.
have established a tripod of theorems that connect these three notions for
score-based classifier [1]. We will use their notation to present these different concepts.

In MAPIE, we focus specifically on set prediction for multi-class classification
and on calibration for binary classification.

Although set prediction is possible for binary classification, we don't recommend using this setting.
Here is an argument from Gupta et al:

    PSs and CIs are only ‘informative’ if the sets or intervals produced by them are small. To quantify
    this, we measure CIs using their width (denoted as :math:`|C(.)|)`, and PSs using their diameter (defined as
    the width of the convex hull of the PS). For example, in the case of binary classification, the diameter
    of a PS is :math:`1` if the prediction set is :math:`{0,1}`, and :math:`0` otherwise (since :math:`Y\in{0,1}`
    always holds, the set :math:`{0,1}` is ‘uninformative’). A short CI such as :math:`[0.39, 0.41]`
    is more informative than a wider one such as :math:`[0.3, 0.5]`.

In a few words, what you need to remember about these concepts :

* *Calibration* is useful when we want to transform a score (typically given by an ML model)
  that is not a probability into a probability. The algorithms that are used for calibration
  can be interpreted as estimators of the confidence level.
* *Set Prediction* is based on the estimation not of a single prediction,
  but of a range of more or less precise predictions, each with a degree of confidence.
* In contrast, *Probabilistic Prediction* is based on the estimation not of a single probability
  of occurrence of a target or class, but of a range of more or less precise probabilities of occurrence,
  each with a degree of confidence.


1. Set Prediction
-----------------

Definition 1 (CI w.r.t .. :math:`f`) [1].
    Fix a predictor :math:`f:\mathcal{X} \to [0, 1]` and let :math:`(\mathcal{X}, \mathcal{Y} \sim P)`.
    A function :math:`C:[0,1]\to\mathcal{I}` is said to be :math:`(1-\alpha)`-CI with respect to :math:`f` if:

.. math:: 
    P(\mathbb{E}[Y|f(X)]\in C(f(X))) \geq 1 - \alpha

See :class:`~mapie.classification.MapieClassifier`.


2. Probabilistic Prediction
---------------------------

Definition 1 (PS w.r.t .. :math:`f`) [1].
    Fix a predictor :math:`f:\mathcal{X} \to [0, 1]` and let :math:`(\mathcal{X}, \mathcal{Y} \sim P)`.
    A function :math:`S:[0,1]\to\mathcal{L}` is said to be :math:`(1-\alpha)`-PS with respect to :math:`f` if:

.. math:: 
    P(Y\in S(f(X))) \geq 1 - \alpha

In the framework of conformal prediction, the Venn predictor has this property.


3. Calibration
--------------

Definition 3 (Approximate calibration) [1].
    Fix a predictor :math:`f:\mathcal{X} \to [0, 1]` and let :math:`(\mathcal{X}, \mathcal{Y} \sim P)`.
    The predictor :math:`f:\mathcal{X} \to [0, 1]` is :math:`(\epsilon,\alpha)`-calibrated
    for some :math:`\epsilon,\alpha\in[0, 1]` if with probability at least :math:`1-\alpha`:

.. math:: 
    |\mathbb{E}[Y|f(X)] - f(X)| \leq \epsilon

See :class:`~sklearn.calibration.CalibratedClassifierCV` or :class:`~mapie.calibration.MapieCalibrator`.


4. References
-------------

[1] Gupta, Chirag, Aleksandr Podkopaev, and Aaditya Ramdas.
"Distribution-free binary classification: prediction sets, confidence intervals and calibration."
Advances in Neural Information Processing Systems 33 (2020): 3711-3723.
