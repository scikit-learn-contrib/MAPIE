:orphan:

.. _risk_control_examples:

All risk control examples
=========================

Following is a collection of notebooks demonstrating how to use MAPIE for risk control.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


1. Quickstart examples
----------------------

The following examples present the main functionalities of MAPIE through basic quickstart risk control problems.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we explain how to do risk control for binary classification with MAPIE.">

.. only:: html

  .. image:: /examples_risk_control/1-quickstart/images/thumb/sphx_glr_plot_risk_control_binary_classification_thumb.png
    :alt:

  :ref:`sphx_glr_examples_risk_control_1-quickstart_plot_risk_control_binary_classification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use MAPIE to control the precision of a binary classifier</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


2. Advanced analysis
--------------------

The following examples use MAPIE for discussing more complex risk control problems.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we explain how to perform risk control for multi-label classification using the Learn-Then-Test (LTT) procedure implemented in MAPIE.">

.. only:: html

  .. image:: /examples_risk_control/2-advanced-analysis/images/thumb/sphx_glr_plot_risk_control_multi-label_classification_thumb.png
    :alt:

  :ref:`sphx_glr_examples_risk_control_2-advanced-analysis_plot_risk_control_multi-label_classification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use MAPIE to control the risk of a multi-label classifier</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="AI is a powerful tool for email sorting (for example between spam and urgent emails). However, because algorithms are not perfect, manual verification is sometimes required. Thus one would like to be able to control the amount of emails sent to human validation. One way to do so is to define a multi-parameter prediction function based on a classifier&#x27;s predicted scores. This would allow defining a rule for email checking, which could be adapted by varying the prediction parameters.">

.. only:: html

  .. image:: /examples_risk_control/2-advanced-analysis/images/thumb/sphx_glr_plot_risk_control_multi_parameter_binary_classification_thumb.png
    :alt:

  :ref:`sphx_glr_examples_risk_control_2-advanced-analysis_plot_risk_control_multi_parameter_binary_classification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use MAPIE to control risk of a binary classifier with multiple prediction parameters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example illustrates how to control the recall of a semantic segmentation model using MAPIE.">

.. only:: html

  .. image:: /examples_risk_control/2-advanced-analysis/images/thumb/sphx_glr_plot_semantic_segmentation_recall_control_thumb.png
    :alt:

  :ref:`sphx_glr_examples_risk_control_2-advanced-analysis_plot_semantic_segmentation_recall_control.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recall control for semantic segmentation with MAPIE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example illustrates how to control the precision of a semantic segmentation model using MAPIE.">

.. only:: html

  .. image:: /examples_risk_control/2-advanced-analysis/images/thumb/sphx_glr_plot_semantic_segmentation_precision_control_thumb.png
    :alt:

  :ref:`sphx_glr_examples_risk_control_2-advanced-analysis_plot_semantic_segmentation_precision_control.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Precision control for semantic segmentation with MAPIE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we explain how to do multi-risk control for binary classification with MAPIE.">

.. only:: html

  .. image:: /examples_risk_control/2-advanced-analysis/images/thumb/sphx_glr_plot_multi_risk_control_binary_classification_thumb.png
    :alt:

  :ref:`sphx_glr_examples_risk_control_2-advanced-analysis_plot_multi_risk_control_binary_classification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use MAPIE to control multiple risks of a binary classifier</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how to use risk control methods for Large Language Models (LLMs) acting as judges. We simulate a scenario where an LLM evaluates answers, and we want to control the risk of hallucination detection. Moreover, we want the judge to abstain from making a decision when it is uncertain.">

.. only:: html

  .. image:: /examples_risk_control/2-advanced-analysis/images/thumb/sphx_glr_plot_risk_control_llm_as_a_judge_thumb.png
    :alt:

  :ref:`sphx_glr_examples_risk_control_2-advanced-analysis_plot_risk_control_llm_as_a_judge.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Risk Control for LLM as a Judge with Abstention</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /examples_risk_control/1-quickstart/index.rst
   /examples_risk_control/2-advanced-analysis/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: examples_risk_control_python.zip </examples_risk_control/examples_risk_control_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: examples_risk_control_jupyter.zip </examples_risk_control/examples_risk_control_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
