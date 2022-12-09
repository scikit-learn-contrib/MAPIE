.. title:: Theoretical Description : contents

.. _theoretical_description_calibration

=======================
Theoretical Description
=======================


One method for multi-class calibration has been implemented in MAPIE so far :
Top-Label [1].

The goal of binary calibration is to transform a score (typically the one returned by an ML model) that has no mathematical meaning into a
probability. The algorithms that are used for this type of calibration can be interpreted as estimators of the confidence level. Hence,
they are made of an independent and dependent variable.

The figure below illustrates what we would expect as a result from a calibration procedure, with the scores predicted being closer to the
true accuracy compared to the original output.

.. image:: images/calibration_basic.png
   :width: 300
   :align: center


Firstly, we introduce binary calibration, we denote the :math:`(X, y)` pair as the score and ground truth for the object. Hence, :math:`y`
is made of :math:`{0, 1}`. We define a calibrated output if for every prediction :math:`q \in [0, 1]`:

.. math:: 
    Pr(Y = 1 \mid h(X) = q) = q \quad \text{where} \quad h() \text{is the probabilistic predictor}


To apply calibration directly to a multi-class context, Gupta et al. propose a framework, multiclass-to-binary (M2B), in order to apply
binary calibration concepts to a multi-class problem.


1. Top-Label
------------

Top-Label calibration is a calibration technique introduced by Gupta et al. to calibrate the label with the highest score.
This calibration technique can therefore be directly applied to a multi-class problem.

To be explained in the most intuitive way, top-label calibration simply performs a standard calibration procedure
(such as Platt scaling or Isotonic regression) on the maximum prediction values. This enables us to say that when we choose
label for our classification task, we are sure that specifically for this class the scores are calibrated. 


We denote :math:`c` as the classifier and :math:`h` as the maximum score from the classifier. In this context, a calibrated output
for Top-Label calibration would be:

.. math:: 
    Pr(Y = c(X) \mid h(X), c(X)) = h(X)


2. Metric for calibration
-------------------------

Expected calibration error:

The main metric to check if the calibration has been done correctly is the Expected Calibration Error (ECE). It is made of two
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


Top-Label ECE:

In the top-label scenario, we only calculate the ECE for the top-label. Hence, per top-label, we condition the calculation
of the accuracy and confidence based on the top label and take the average ECE for each top-label.


3. References
-------------

[1] Gupta, Chirag, and Aaditya K. Ramdas.
"Top-label calibration and multiclass-to-binary reductions."
arXiv preprint arXiv:2107.08353 (2021).
