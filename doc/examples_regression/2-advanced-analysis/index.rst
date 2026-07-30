

.. _sphx_glr_examples_regression_2-advanced-analysis:

.. _regression_examples_2:

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


.. toctree::
   :hidden:

   /examples_regression/2-advanced-analysis/plot_conformal_predictive_distribution
   /examples_regression/2-advanced-analysis/plot_cqr_symmetry_difference
   /examples_regression/2-advanced-analysis/plot_nested-cv
   /examples_regression/2-advanced-analysis/plot_both_uncertainties
   /examples_regression/2-advanced-analysis/plot_timeseries_enbpi
   /examples_regression/2-advanced-analysis/plot-coverage-width-based-criterion
   /examples_regression/2-advanced-analysis/plot_ResidualNormalisedScore_tutorial
   /examples_regression/2-advanced-analysis/plot_conditional_coverage
   /examples_regression/2-advanced-analysis/plot_cqr_tutorial
   /examples_regression/2-advanced-analysis/plot_coverage_validity
   /examples_regression/2-advanced-analysis/plot_main-tutorial-regression

