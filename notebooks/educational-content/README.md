# MAPIE educational content

This directory contains educational notebooks imported from the
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

The `cosmosqa_10k.json`, `use_case_files/`, and `utils/` contents are copied
from the same source because the notebooks depend on them.

## Running the notebooks

Follow the environment setup instructions in the parent
[`notebooks/README.md`](../README.md). These notebooks were originally designed
for Google Colab, and some cells contain paths rooted at
`/content/MAPIE-Educational-Content/notebooks`. When running locally, replace
those paths with paths relative to this directory.
