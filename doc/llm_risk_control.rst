.. title:: How to control LLM risks?

.. _llm_risk_control:

###########################
How to control LLM risks?
###########################

Risk control techniques in binary classification can be used to enhance LLM trustworthiness by implementing response guardrails, such as censoring undesired content. For an example, see :ref:`sphx_glr_examples_risk_control_2-advanced-analysis_plot_risk_control_llm_as_a_judge.py`.

Furthermore, conformal prediction methods can be applied to LLM-based classifiers. We propose a method presented in `Benchmarking LLMs via Uncertainty Quantification <https://arxiv.org/abs/2401.12794>`_, which reduces a commonsense reasoning task (CosmosQA dataset) to a classification problem, enabling the use of conformal predictions. The idea is to extract only the logits corresponding to the possible answers, and use a softmax so that the LLM can be used as a simple classifier.

The following `repository <https://github.com/Valentin-Laurent/MAPIE-Educational-Content>`_ (not maintained by the MAPIE team) implements part of this paper for educational purposes in the ``MAPIE_for_cosmosqa`` notebook.

Additionally, we invite you to read our `blog article <https://medium.com/capgemini-invent-lab/quantifying-llms-uncertainty-with-conformal-predictions-567870e63e00>`_, where we dive deeper into the topic.