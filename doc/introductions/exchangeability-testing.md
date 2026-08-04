# Exchangeability Testing

Many statistical guarantees in conformal prediction and risk control rely on
**exchangeability**. A sequence of observations is exchangeable when reordering
the observations does not change their joint distribution. Informally, the
position of an observation in the sequence should not reveal that it comes from
a different data-generating regime.

Independent and identically distributed observations are exchangeable, but
exchangeability is a weaker assumption: observations can be dependent as long
as their joint distribution is invariant to their order. Distribution shifts,
time trends, and changes in the relationship between features and targets can
violate this assumption and undermine MAPIE's guarantees.

## What Exchangeability Tests Tell You

Exchangeability tests look for evidence that the observation order contains a
systematic pattern. A rejected test indicates evidence against exchangeability
and should prompt investigation before relying on a statistical guarantee.

A test that does not reject exchangeability is not proof that the assumption
holds. No single test can detect every possible violation, and different tests
are sensitive to different kinds of change. Using complementary tests and
combining their results with domain knowledge provides a more complete
assessment.

!!! warning "Keep the original observation order"
    Shuffling data before testing removes the ordering information that the
    tests use to detect a shift. The tested order should represent how the data
    was collected or how observations will arrive after deployment.

## Exchangeability Testing in MAPIE

MAPIE supports three related workflows:

- **Fixed-dataset testing** applies permutation-based or martingale-based tests
  to a labeled dataset. This is useful before conformalizing a model or when
  auditing a historical batch of observations.
- **Online testing** updates martingale tests as labeled observations arrive.
  It is designed to detect departures from exchangeability during deployment
  without repeatedly inflating the false-alarm rate.
- **Risk monitoring** compares a deployed model's risk with a reference risk
  and signals statistically supported harmful degradation. This addresses a
  narrower, performance-oriented question that complements general
  exchangeability testing.

Exchangeability testing often requires labels, including in production. Labels
do not necessarily need to be available immediately or for every observation;
tests can be updated when a representative labeled batch becomes available.

## Interpreting the Result

- **Evidence against exchangeability** means the assumptions behind a MAPIE
  guarantee may no longer be credible for the tested data. Investigate the
  source of the shift, retrain or recalibrate if appropriate, and test again on
  representative data.
- **No detected violation** means the selected test did not find evidence at
  its configured significance level. Continue monitoring and avoid treating
  this result as confirmation that every form of shift is absent.

## Explore Exchangeability Testing in MAPIE

- [Exchangeability theory](../theory/exchangeability.md) introduces
  permutation tests, conformal p-values, martingales, and risk monitoring in
  detail.
- [Exchangeability-testing examples](../generated/exchangeability_testing/index.md)
  cover fixed datasets, online streams, fitted regression and classification
  models, and deployed-model risk monitoring.
- [Conformal Prediction overview](conformal-prediction.md) explains where the
  exchangeability assumption enters the conformal workflow.

