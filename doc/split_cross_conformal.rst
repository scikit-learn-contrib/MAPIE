################################
Split/Cross-Conformal Prediction
################################

**MAPIE** is basically based on two types of techniques:

1. Cross conformal predictions
==============================

- Conformity scores on the whole training set obtained by cross-validation,
- Perturbed models generated during the cross-validation.

**MAPIE** then combines all these elements in a way that provides prediction intervals on new data with strong theoretical guarantees [3-4].

.. image:: https://github.com/simai-ml/MAPIE/raw/master/doc/images/mapie_internals_regression.png
    :width: 300
    :align: center

2. Split conformal predictions
==============================

- Construction of a conformity score
- Calibration of the conformity score on a calibration set not seen by the model during training

**MAPIE** then uses the calibrated conformity scores to estimate sets of labels associated with the desired coverage on new data with strong theoretical guarantees [5-6-7].

.. image:: https://github.com/simai-ml/MAPIE/raw/master/doc/images/mapie_internals_classification.png
    :width: 300
    :align: center
