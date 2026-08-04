# Risk Control

Machine learning models are often turned into decisions by applying a rule or
threshold to their outputs. A threshold may appear to achieve a target metric
on validation data, but that observed value alone does not guarantee similar
performance on future data.

Risk control uses held-out data to select a decision rule with a statistical
guarantee. The goal is to keep a chosen **risk** below a target level—or,
equivalently, to keep a performance metric such as precision or recall above a
target level. For example, risk control can select a classification threshold
that guarantees a minimum precision with a specified confidence.

## From Predictions to Controlled Decisions

A typical risk-control workflow has four parts:

1. **Fit** the predictive model on training data.
2. **Define** the risk or performance metric, its target level, and the model
   parameter to adjust, such as a classification threshold.
3. **Calibrate** the controller on separate labeled data to identify parameter
   values that satisfy the statistical criterion.
4. **Predict** with a selected valid parameter on new observations.

The confidence level describes how certain the guarantee should be. Higher
confidence generally produces more conservative decisions. A valid parameter
may not exist for every combination of model, dataset, risk target, and
confidence level; risk control cannot compensate for a model that is unable to
meet the requested objective.

## Risk Control in MAPIE

MAPIE supports several kinds of controlled decisions:

- **Binary classification**: select one- or multi-dimensional decision
  thresholds while controlling built-in or custom risks.
- **Multi-label classification**: construct prediction sets with controlled
  precision or recall.
- **Semantic segmentation**: control precision or recall for pixel-level
  prediction sets.

The available methods provide different guarantees and require different
assumptions:

- **Conformal Risk Control (CRC)** controls the expected value of a monotone,
  bounded risk under exchangeability.
- **Risk-Controlling Prediction Sets (RCPS)** provide a high-probability risk
  guarantee for monotone risks under an i.i.d. assumption.
- **Learn Then Test (LTT)** uses multiple hypothesis testing to handle a wider
  range of risks, including non-monotone risks, under an i.i.d. assumption.

## Risk Control and Conformal Prediction

Both approaches use data that was not used to train the underlying model and
rely on statistical assumptions about future observations. Conformal
prediction usually targets the coverage of a prediction interval or set. Risk
control instead starts from an application-specific loss or performance metric
and tunes a decision rule to meet that target. The appropriate approach depends
on whether the desired guarantee concerns predictive coverage or a downstream
decision metric.

## Explore Risk Control in MAPIE

- [Risk-control theory](../theory/risk-control.md) compares CRC, RCPS, and LTT
  and states their guarantees in detail.
- [LLM risk control](../theory/llm-risk-control.md) explains how risk control can
  be applied to an LLM-as-a-judge workflow.
- [Risk-control examples](../generated/risk_control/index.md) demonstrate binary
  classification, multi-label classification, semantic segmentation, custom
  risks, and multi-risk settings.
- [Choosing the right algorithm](../getting-started/choosing-algorithm.md) places
  risk control within MAPIE's broader set of tools.

