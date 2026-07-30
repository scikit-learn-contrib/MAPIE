:orphan:

.. _regression_examples:

All regression examples
========================

Following is a collection of notebooks demonstrating how to use MAPIE.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


1. Quickstart
----------------------

The following examples present the main functionalities of MAPIE through basic quickstart regression problems.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot prediction intervals">

.. only:: html

  .. image:: /examples_regression/1-quickstart/images/thumb/sphx_glr_plot_toy_model_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_1-quickstart_plot_toy_model.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plot prediction intervals</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" This example uses CrossConformalRegressor to estimate prediction intervals associated with Gamma distributed target. The limit of the absolute residual conformity score is illustrated.">

.. only:: html

  .. image:: /examples_regression/1-quickstart/images/thumb/sphx_glr_plot_compare_conformity_scores_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_1-quickstart_plot_compare_conformity_scores.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Data with gamma distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" SplitConformalRegressor and ConformalizedQuantileRegressor are used to conformalize uncertainties for large models for which the cost of cross-validation is too high. Typically, neural networks rely on a single validation set.">

.. only:: html

  .. image:: /examples_regression/1-quickstart/images/thumb/sphx_glr_plot_prefit_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_1-quickstart_plot_prefit.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use a pre-trained model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" CrossConformalRegressor, JackknifeAfterBootstrapRegressor, ConformalizedQuantileRegressor are used to estimate the prediction intervals of 1D heteroscedastic data using different strategies.">

.. only:: html

  .. image:: /examples_regression/1-quickstart/images/thumb/sphx_glr_plot_heteroscedastic_1d_data_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_1-quickstart_plot_heteroscedastic_1d_data.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Data with uneven uncertainty</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" We show here how to use various MAPIE methods on data with homoscedastic data.">

.. only:: html

  .. image:: /examples_regression/1-quickstart/images/thumb/sphx_glr_plot_homoscedastic_1d_data_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_1-quickstart_plot_homoscedastic_1d_data.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Data with constant uncertainty</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Note: in this tutorial, we use the following terms employed in the scientific literature:">

.. only:: html

  .. image:: /examples_regression/1-quickstart/images/thumb/sphx_glr_plot_ts-tutorial_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_1-quickstart_plot_ts-tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tutorial for time series</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


-----

2. Advanced analysis
--------------------

The following examples use MAPIE for discussing more complex MAPIE problems.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Conformal Predictive Distribution">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_conformal_predictive_distribution_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_conformal_predictive_distribution.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Conformal Predictive Distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" An example plot of ConformalizedQuantileRegressor illustrating the impact of the symmetric_correction parameter.">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_cqr_symmetry_difference_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_cqr_symmetry_difference.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The symmetric correction parameter in conformalized quantile regression</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" This example compares non-nested and nested cross-validation strategies when using CrossConformalRegressor.">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_nested-cv_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_nested-cv.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hyperparameters tuning with cross-conformal regression</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" This example uses CrossConformalRegressor, ConformalizedQuantileRegressor and JackknifeAfterBootstrapRegressor to estimate prediction intervals capturing both aleatoric and epistemic uncertainties on a one-dimensional dataset with homoscedastic noise and normal sampling.">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_both_uncertainties_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_both_uncertainties.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Estimating aleatoric and epistemic uncertainties</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Note: in this example, we use the following terms employed in the scientific literature:">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_timeseries_enbpi_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_timeseries_enbpi.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">EnbPI technique for time series</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" This example uses CrossConformalRegressor, ConformalizedQuantileRegressor and JackknifeAfterBootstrapRegressor. metrics is used to estimate the coverage width based criterion of 1D homoscedastic data using different strategies. The coverage width based criterion is computed with the function coverage_width_based">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot-coverage-width-based-criterion_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot-coverage-width-based-criterion.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Focus on intervals width</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" We will use the sklearn california housing dataset to understand how the residual normalised score works and show the multiple ways of using it.">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_ResidualNormalisedScore_tutorial_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_ResidualNormalisedScore_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Focus on residual normalised score</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" This example uses SplitConformalRegressor, JackknifeAfterBootstrapRegressor, with conformal scores that returns adaptive intervals i.e. (~mapie.conformity_scores.GammaConformityScore and ResidualNormalisedScore) as well as ConformalizedQuantileRegressor and  The conditional coverage is computed with the three functions that allows to estimate the conditional coverage in regression :func:~mapie.metrics.regression_ssc`, regression_ssc_score and hsic.">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_conditional_coverage_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_conditional_coverage.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Focus on local (or "conditional") coverage</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" We will use the sklearn california housing dataset as the base for the comparison of the different methods available on MAPIE. Two classes will be used: ConformalizedQuantileRegressor for CQR. We use CrossConformalRegressor and JackknifeAfterBootstrapRegressor for the other methods.">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_cqr_tutorial_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_cqr_tutorial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Conformalized quantile regression on gamma distributed data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" This example verifies that conformal claims are valid in the MAPIE package when using the CP prefit/split methods.">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_coverage_validity_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_coverage_validity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Coverage validity for regression tasks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" In this tutorial, we compare the prediction intervals estimated by MAPIE on a simple, one-dimensional, ground truth function f(x) = x * sin(x). Throughout this tutorial, we will answer the following questions:">

.. only:: html

  .. image:: /examples_regression/2-advanced-analysis/images/thumb/sphx_glr_plot_main-tutorial-regression_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_2-advanced-analysis_plot_main-tutorial-regression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison between conformalized quantile regressor and cross methods</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


-----

3. Simulations from scientific articles
---------------------------------------

The following examples reproduce the simulations from the scientific
articles that introduce the methods implemented
in MAPIE for regression settings.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" CrossConformalRegressor is used to investigate the coverage level and the prediction interval width as a function of the dimension using simulated data points as introduced in Foygel-Barber et al. (2021) [1].">

.. only:: html

  .. image:: /examples_regression/3-scientific-articles/images/thumb/sphx_glr_plot_barber2020_simulations_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_3-scientific-articles_plot_barber2020_simulations.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Predictive inference with the jackknife+, Foygel-Barber et al. (2020)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Note: in this example, we use the following terms employed in the scientific literature:">

.. only:: html

  .. image:: /examples_regression/3-scientific-articles/images/thumb/sphx_glr_plot_zaffran2022_comparison_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_3-scientific-articles_plot_zaffran2022_comparison.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Adaptive conformal predictions for time series, Zaffran et al. (2022)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip=" JackknifeAfterBootstrapRegressor and CrossConformalRegressor are used to reproduce the simulations by Kim et al. (2020) [1] in their article which introduces the jackknife+-after-bootstrap method.">

.. only:: html

  .. image:: /examples_regression/3-scientific-articles/images/thumb/sphx_glr_plot_kim2020_simulations_thumb.png
    :alt:

  :ref:`sphx_glr_examples_regression_3-scientific-articles_plot_kim2020_simulations.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Predictive inference is free with the Jackknife+-after-Bootstrap, Kim et al. (2020)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


-----

4. Other notebooks
--------------------------------------------

This section lists a series of Jupyter notebooks hosted on the MAPIE Github repository that can be run on Google Colab:

  - `Estimating prediction intervals for time series forecast with EnbPI and ACI <https://github.com/scikit-learn-contrib/MAPIE/tree/master/notebooks/regression/ts-changepoint.ipynb>`_




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /examples_regression/1-quickstart/index.rst
   /examples_regression/2-advanced-analysis/index.rst
   /examples_regression/3-scientific-articles/index.rst
   /examples_regression/4-other-notebooks/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: examples_regression_python.zip </examples_regression/examples_regression_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: examples_regression_jupyter.zip </examples_regression/examples_regression_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
