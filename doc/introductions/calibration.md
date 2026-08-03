Many classification models (e.g., neural networks) predict class probabilities. These probabilities are often used as confidence scores. However, they rarely reflect the true probability of making a correct prediction: they are miscalibrated. A post-processing calibration step is required to realign the predicted probabilities with the true probabilities.

MAPIE contains an implementation of Venn-ABERS calibrators [^1], the **Top-Label Calibration** algorithm [^2], and several metrics.

For more post-hoc calibration methods and metrics, take a look at the [probmetrics](https://github.com/probkit/probmetrics/) library.

[^1]: Vovk, Vladimir, Ivan Petej, and Valentina Fedorova. "Large-scale probabilistic predictors with and without guarantees of validity." Advances in Neural Information Processing Systems 28 (2015). https://arxiv.org/pdf/1511.00213.pdf

[^2]: Gupta, C., and Ramdas, A. K. "Top-label calibration and multiclass-to-binary reductions." *arXiv preprint arXiv:2107.08353* (2021).
