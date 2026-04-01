######################
Quick Start with MAPIE
######################

This package allows you to easily estimate uncertainties in both regression and classification settings.
In regression settings, **MAPIE** provides prediction intervals on single-output data.
In classification settings, **MAPIE** provides prediction sets on multi-class data.
In any case, **MAPIE** is compatible with any scikit-learn-compatible estimator.


1. Download and install the module
==================================

Install via ``pip``:

.. code:: python

    pip install mapie

or via `conda`:

.. code:: sh

    $ conda install -c conda-forge mapie

To install directly from the github repository :

.. code:: python

    pip install git+https://github.com/scikit-learn-contrib/MAPIE


2. Regression
=====================

Let us start with a basic regression problem.
Here, we generate one-dimensional noisy data that we fit with a MLPRegressor: :ref:`sphx_glr_examples_regression_1-quickstart_plot_toy_model.py`


3. Classification
=======================

Similarly, it's possible to do the same for a basic classification problem: :ref:`sphx_glr_examples_classification_1-quickstart_plot_quickstart_classification.py`


4. Risk Control
=======================

MAPIE implements risk control methods for multilabel classification (in particular, image segmentation) and binary classification: :ref:`sphx_glr_examples_risk_control_1-quickstart_plot_risk_control_binary_classification.py`