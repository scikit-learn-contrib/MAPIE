.. title:: How to control LLM risks?

.. _llm_risk_control:

###########################
How to control LLM risks?
###########################

Exciting news: we're currently working on extending MAPIE to control LLM systems! This study aims to adapt risk control techniques in binary classification to enhance LLM trustworthiness by implementing response guardrails, such as censoring undesired content.

In the mean time, we propose a method presented in `Benchmarking LLMs via Uncertainty Quantification <https://arxiv.org/abs/2401.12794>`_, which reduces a commonsense reasoning task (CosmosQA dataset) to a classification problem, enabling the use of conformal predictions. The idea is to extract only the logits corresponding to the possible answers, and use a softmax so that the LLM can be used as a simple classifier.

The following `repository <https://github.com/vincentblot28/mapie_llm>`_ (not maintained by the MAPIE team) implements part of this paper for educational purposes.