.. title:: Theoretical Description : contents

.. _theoretical_description_calibration

=======================
Theoretical Description
=======================


One method for multi-class calibration has been implemented in MAPIE so far :
Top-Label Calibration [1].

The goal of binary calibration is to transform a score (typically given by an ML model) that is not a probability into a
probability. The algorithms that are used for calibration can be interpreted as estimators of the confidence level. Hence,
they need independent and dependent variables to be fitted.

The figure below illustrates what we would expect as a result from a calibration, with the scores predicted being closer to the
true probability compared to the original output.

.. image:: images/calibration_basic.png
   :width: 300
   :align: center


Firstly, we introduce binary calibration, we denote the :math:`(h(X), y)` pair as the score and ground truth for the object. Hence, :math:`y`
values are in :math:`{0, 1}`. The model is calibrated if for every output :math:`q \in [0, 1]`, we have:

.. math::
    Pr(Y = 1 \mid h(X) = q) = q

where :math:`h()` is the score predictor.

To apply calibration directly to a multi-class context, Gupta et al. propose a framework, multiclass-to-binary, in order to reduce
a multi-class calibration to multiple binary calibrations (M2B).


1. Top-Label
------------

Top-Label calibration is a calibration technique introduced by Gupta et al. to calibrate the model according to the highest score and
the corresponding class (see [1] Section 2). This framework offers to apply binary calibration techniques to multi-class calibration.

More intuitively, top-label calibration simply performs a binary calibration (such as Platt scaling or isotonic regression) on the
highest score and the corresponding class, whereas confidence calibration only calibrates on the highest score (see [1] Section 2).

Let :math:`c` be the classifier and :math:`h` be the maximum score from the classifier. The couple :math:`(c, h)` is calibrated
according to Top-Label calibration if:

.. math::
    Pr(Y = c(X) \mid h(X), c(X)) = h(X)


2. Metrics for calibration
-------------------------

**Expected calibration error**

The main metric to check if the calibration is correct is the Expected Calibration Error (ECE). It is based on two
components, accuracy and confidence per bin. The number of bins is an hyperparamater :math:`M`, and we refer to a specific bin by
:math:`B_m`.

.. math::
    \text{acc}(B_m) &= \frac{1}{\left| B_m \right|} \sum_{i \in B_m} {y}_i \\
    \text{conf}(B_m) &= \frac{1}{\left| B_m \right|} \sum_{i \in B_m} \hat{f}(x)_i


The ECE is the combination of these two metrics combined together.

.. math::
    \text{ECE} = \sum_{m=1}^M \frac{\left| B_m \right|}{n} \left| acc(B_m) - conf(B_m) \right|

In simple terms, once all the different bins from the confidence scores have been created, we check the mean accuracy of each bin.
The absolute mean difference between the two is the ECE. Hence, the lower the ECE, the better the calibration was performed.


**Top-Label ECE**

In the top-label calibration, we only calculate the ECE for the top-label class. Hence, per top-label class, we condition the calculation
of the accuracy and confidence based on the top label and take the average ECE for each top-label.


3. References
-------------

[1] Gupta, Chirag, and Aaditya K. Ramdas.
"Top-label calibration and multiclass-to-binary reductions."
arXiv preprint arXiv:2107.08353 (2021).
