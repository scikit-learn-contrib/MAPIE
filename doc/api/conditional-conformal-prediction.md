# Conditional Conformal Prediction

Conformal prediction methods with conditional validity guarantees.

These estimators require the optional `conditional` dependency:

```bash
pip install "mapie[conditional]"
```

For background and usage guidance, see the
[theoretical description](../theory/conditional-guarantees.md) and the runnable
[regression](../generated/regression/2-advanced-analysis/plot_conditional_conformal_regression_groups.md)
and
[classification](../generated/classification/2-advanced-analysis/plot_conditional_conformal_classification_groups.md)
examples.

## Regression

::: mapie.conditional_conformal_prediction.ConditionalSplitConformalRegressor
    options:
      heading_level: 3

---

## Classification

::: mapie.conditional_conformal_prediction.ConditionalSplitConformalClassifier
    options:
      heading_level: 3
