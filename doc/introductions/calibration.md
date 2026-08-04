# Calibration

Many classification models output scores that are interpreted as probabilities
or confidence levels. Those scores are **calibrated** when predictions made with
a given confidence are correct at approximately the same rate. For example,
among predictions assigned 80% confidence, about 80% should be correct.

Models can be accurate but poorly calibrated: they may rank classes correctly
while being systematically overconfident or underconfident. Post-hoc
calibration learns a transformation from the model's scores to probabilities
that better reflect observed outcomes, without retraining the underlying model.

MAPIE provides two complementary approaches:

- **Top-label calibration** calibrates the confidence attached to a
  classifier's most likely class [^1].
- **Venn-Abers calibration** produces calibrated probability estimates for
  binary and multiclass classifiers [^2].

MAPIE also provides metrics for evaluating how closely predicted probabilities
match observed frequencies.

!!! note "Calibration and conformalization"
    Probability calibration and conformal prediction solve different problems.
    Calibration improves the interpretation of probability estimates, while
    conformal prediction constructs intervals or sets with a coverage target.
    The conformal prediction literature sometimes calls its held-out step
    *calibration*; MAPIE uses *conformalization* where possible to distinguish
    the two concepts.

## Explore Calibration in MAPIE

- [Calibration theory](../theory/calibration.md) introduces binary and
  top-label calibration.
- [Calibration metrics](../theory/metrics-calibration.md) describes expected
  and top-label calibration errors.
- [Calibration examples](../generated/calibration/index.md) demonstrate the
  available calibrators on practical problems.
- [Calibration notebooks](../calibration/notebooks.md) links to additional
  tutorials.

For additional post-hoc calibration methods and metrics, see the
[probmetrics](https://github.com/probkit/probmetrics/) library.

[^1]: Gupta, C., and Ramdas, A. K. "Top-label calibration and multiclass-to-binary reductions." *arXiv preprint arXiv:2107.08353* (2021).

[^2]: Vovk, V., Petej, I., and Fedorova, V. "Large-scale probabilistic predictors with and without guarantees of validity." *Advances in Neural Information Processing Systems 28* (2015).
