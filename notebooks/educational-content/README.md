# MAPIE educational content

This directory contains educational notebooks, some of which have been imported from the
[MAPIE Educational Content repository](https://github.com/Valentin-Laurent/MAPIE-Educational-Content/).

## Notebooks

- `regression-tutorial.ipynb`: introduction to measuring regression model
  uncertainty with MAPIE.
- `regression-tutorial-correction.ipynb`: completed version of the regression
  tutorial.
- `regression-use-case.ipynb`: hands-on regression use case.
- `regression-use-case-correction.ipynb`: completed version of the regression
  use case.
- `MAPIE_for_cosmosqa.ipynb`: conformal prediction for a language model on the
  CosmosQA dataset.
- `MAPIE_for_cosmosqa_correction.ipynb`: completed version of the CosmosQA
  notebook.
- `binary_risk_control_for_llm.ipynb` : example that demonstrates how to use risk
control methods from the MAPIE library for binary text classification
- `binary_risk_control_for_llm_correction.ipynb` : completed version of the binary
risk control.

The `cosmosqa_10k.json`, `use_case_files/`, and `utils/` contents are copied
from the same source because the notebooks depend on them.

The `data/` directory contains datasets used by these notebooks :
- `sms-spam-collection-dataset/spam.csv` : classification dataset of SMS as spam or not spam. Originally from Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
- `sms-spam-collection-dataset/sms_spam_llm.csv` : modified version of the dataset above (selection of examples and classification by LLMs)
- `ag-new-classification-dataset` : text classification dataset of news article in four classes. Contructed by Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015) and based on http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html.

## Running the notebooks

Follow the environment setup instructions in the parent
[`notebooks/README.md`](../README.md). Some of these notebooks were originally designed
for Google Colab, and some cells contain paths rooted at
`/content/MAPIE-Educational-Content/notebooks`. When running locally, replace
those paths with paths relative to this directory.
