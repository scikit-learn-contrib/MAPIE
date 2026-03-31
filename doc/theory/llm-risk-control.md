# How to Control LLM Risks?

Risk control techniques in binary classification can be used to enhance **LLM trustworthiness** by implementing **response guardrails**, such as censoring undesired content.

---

## LLM as Binary Classifier

Conformal prediction methods can be applied to LLM-based classifiers. We propose a method presented in [Benchmarking LLMs via Uncertainty Quantification](https://arxiv.org/abs/2401.12794), which:

1. Reduces a commonsense reasoning task (CosmosQA dataset) to a **classification problem**
2. Extracts only the logits corresponding to the possible answers
3. Applies a softmax so the LLM can be used as a simple classifier
4. Enables the use of **conformal predictions**

---

## Resources

!!! info "Educational Repository"
    The following [repository](https://github.com/Valentin-Laurent/MAPIE-Educational-Content) (not maintained by the MAPIE team) implements part of this paper for educational purposes in the `MAPIE_for_cosmosqa` notebook.

!!! tip "Blog Article"
    Read our [blog article on Medium](https://medium.com/capgemini-invent-lab/quantifying-llms-uncertainty-with-conformal-predictions-567870e63e00) where we dive deeper into the topic.
