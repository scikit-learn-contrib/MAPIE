=======
History
=======

1.x.x (2025-xx-xx)
------------------
* Introduce VennAbers calibrator both for binary and multiclass classification

* Remove dependency of internal classes on sklearn's check_is_fitted
* Add an example of risk control with LLM as a judge
* Add comparison with naive threshold in risk control quick start example
* Configure self hosted runner for minimal requirements tests
* Choose better thumbnails in the lists of examples in the documentation
* Add a new binary classification risk called `predicted_positive_fraction` and update the examples accordingly.
* Add a disclosure about LLM usage to the pull request template.
* Fix data leakage for time series tutorials
* Improve documentation display (increase width, simplify examples titles, improve API table, improve decision tree for choosing the right algorithm)
* fix bug with CRC and RCPS where the computed lambda was not the best
* add example of LLM as a judge with abstention

1.2.0 (2025-11-17)
------------------
* Implement extension of binary risk control to multi-risk
* Implement extension of binary risk control to multi-dimensional parameters
* Reorganise code structure for risk control
* BinaryClassificationController allows BinaryClassificationRisk and string representations of risks
* Improve quick start documentation for risk control
* Add new tutorial for risk control with multiple risks
* Add new tutorial for risk control with multi-dimensional parameters
* Fix issue 614 to pass Predict Params to RAPS for conformity score calculation and EnsembleClassifier
* Fix issue 790 to make `BlockBootstrap` include all non-training indices in the test set
* Update Python environment: dependancies are now in pyproject.toml
* Update CI and add long_tests folder for tests requiring more time

1.1.0 (2025-09-22)
------------------

* Implement new binary risk control feature, see BinaryClassificationController and BinaryClassificationRisk
* See also the reworked risk control documentation
* Revert incorrect renaming of calibration to conformalization in PrecisionRecallController
* Fix warnings when running tests
* Add scientific references for regression conformity scores
* Fix double inference when using `predict_set` function in split conformal classification
* MAPIE now supports Python versions up to the latest release (currently 3.13)
* Change `prefit` default value to `True` in split methods' docstrings to remain consistent with the implementation
* Fix issue 699 to replace `TimeSeriesRegressor.partial_fit` with `TimeSeriesRegressor.update`

1.0.1 (2025-05-22)
------------------

* Patch following v1.0.0 release: removing dependence to typing_extensions, making MAPIE unusable if this package is not installed

1.0.0 (2025-05-22)
------------------

* Major update, including a complete classification and regression public API rework, and a documentation revamp
* Other parts of the public API have been improved as well
* See the v1_release_notes.rst documentation file for extensive and user-focused release notes
* This update also includes bugfixes and developer experience improvements

0.9.2 (2025-01-15)
------------------

* Fix issue 525 in contribution guidelines with syntax errors in hyperlinks and other formatting issues.
* Fix issue 495 to center correctly the prediction intervals
* Fix issue 528 to correct broken ENS image in the documentation
* Fix issue 548 to correct labels generated in tutorial
* Fix issue 547 to fix wrong warning
* Fix issue 480 (correct display of mathematical equations in generated notebooks)
* Temporary solution waiting for issue 588 to be fixed (optimize_beta not working)
* Remove several irrelevant user warnings
* Limit max sklearn version allowed at MAPIE installation
* Refactor MapieRegressor, EnsembleRegressor, and MapieQuantileRegressor, to prepare for the release of v1.0.0
* Documentation build: fix warnings, fix image generation, update sklearn version requirement
* Documentation test: add a doc testing step (in MAKEFILE and CI)
* Increase max line length from 79 to 88 characters
* Bump wheel version
* Other minor evolutions

0.9.1 (2024-09-13)
------------------

* Fix issue 511 to access non-conformity scores with previous path
* Update gitignore by including the documentation folder generated for Mondrian
* Fix (partially) the set-up with pip instead of conda for new contributors

0.9.0 (2024-09-03)
------------------

* Fix citations and license links
* Fix the CQR tutorial to have same data in both methods
* Add `** predict_params` in fit and predict method for Mapie Classifier
* Add Mondrian Conformal Prediction for regression and classification
* Add `** predict_params` in fit and predict method for Mapie Regression
* Update the ts-changepoint notebook with the tutorial
* Change import related to conformity scores into ts-changepoint notebook
* Replace `assert np.array_equal` by `np.testing.assert_array_equal` in Mapie unit tests
* Replace `github.com/simai-ml/MAPIE` by `github.com/scikit-learn-contrib/MAPIE`in all Mapie files
* Extend `ConformityScore` to support regression (with `BaseRegressionScore`) and to support classification (with `BaseClassificationScore`)
* Extend `EnsembleEstimator` to support regression (with `EnsembleRegressor`) and to support classification (with `EnsembleClassifier`)
* Refactor `MapieClassifier` by separating the handling of the `MapieClassifier` estimator into a new class called `EnsembleClassifier`
* Refactor `MapieClassifier` by separating the handling of the `MapieClassifier` conformity score into a new class called `BaseClassificationScore`
* Add severals non-conformity scores for classification (`LAC`, `APS`, `RAPS`, `TopK`) based on `BaseClassificationScore`
* Transfer the logic of classification methods into the non-conformity score classes (`LAC`, `APS`, `RAPS`, `TopK`)
* Extend the classification strategy definition by supporting `method` and `conformity_score` attributes
* Building unit tests for different `Subsample` and `BlockBooststrap` instances
* Change the sign of C_k in the `Kolmogorov-Smirnov` test documentation
* Building a training set with a fraction between 0 and 1 with `n_samples` attribute when using `split` method from `Subsample` class.

0.8.6 (2024-06-14)
------------------

* Fix the quantile formula to ensure valid coverage (deal with infinite interval production and asymmetric conformal scores).
* Fix sphinx dependencies

0.8.5 (2024-06-07)
------------------

* Issue with update from 0.8.4

0.8.4 (2024-06-07)
------------------

* Fix the quantile formula to ensure valid coverage for any number of calibration data in `ConformityScore`.
* Fix overloading of the value of the `method` attribute when using `MapieRegressor` and `MapieTimeSeriesRegressor`.
* Fix conda versionning.
* Reduce precision for test in `MapieCalibrator`.
* Fix invalid certificate when downloading data.
* Add citations utility to the documentation.
* Add documentation for metrics.
* Add explanation and example for symmetry argument in CQR.

0.8.3 (2024-03-01)
------------------

* Allow the use of `y` and `groups` arguments MapieRegressor and MapieClassifier.
* Add possibility of passing fit parameters used by estimators.
* Fix memory issue CQR when testing for upper and lower bounds.
* Add Winkler Interval Score.

0.8.2 (2024-01-11)
------------------

* Resolve issue still present in 0.8.1 by updating pandas.

0.8.1 (2024-01-11)
------------------

* First attemps at fixing library conda issue.

0.8.0 (2024-01-03)
------------------

* Add Adaptative Conformal Inference (ACI) method for MapieTimeSeriesRegressor.
* Add Coverage Width-based Criterion (CWC) metric.
* Allow to use more split methods for MapieRegressor (ShuffleSplit, PredefinedSplit).
* Allow infinite prediction intervals to be produced in regressor classes.
* Integrate ConformityScore into MapieTimeSeriesRegressor.
* Add (extend) the optimal estimation strategy for the bounds of the prediction intervals for regression via ConformityScore.
* Add new checks for metrics calculations.
* Fix reference for residual normalised score in documentation.

0.7.0 (2023-09-14)
------------------

* Add prediction set estimation for binary classification.
* Add Learn-Then-Test method for multilabel-classification.
* Add documentation and notebooks for LTT.
* Add a new conformity score, ResidualNormalisedScore, that takes X into account and allows to compute adaptive intervals.
* Refactor MapieRegressor and ConformityScore to add the possibility to use X in ConformityScore.
* Separate the handling of the estimator from MapieRegressor into a new class called EnsembleEstimator.
* Rename methods (score to lac and cumulated_score to aps) in MapieClassifier.
* Add more notebooks and examples.
* Fix an unfixed random state in one of the classification tests.
* Add statistical calibration tests in binary classification.
* Fix and preserve the split behavior of the check_cv method with and without random state.

0.6.5 (2023-06-06)
------------------

* Add grouped conditional coverage metrics named SSC for regression and classification
* Add HSIC metric for regression
* Migrate conformity scores classes into conformity_scores module
* Migrate regression classes into regression module
* Add split conformal option for regression and classification
* Update check method for calibration
* Fix bug in MapieClassifier with different number of labels in calibration dataset.

0.6.4 (2023-04-05)
------------------

* Fix runtime warning with RAPS method

0.6.3 (2023-03-23)
------------------

* Fix bug when labels do not start at 0

0.6.2 (2023-03-22)
------------------

* Make MapieClassifier a scikit-learn object
* Update documentation for MapieClassifier

0.6.1 (2023-01-31)
------------------

* Fix still existing bug for classification with very low scores

0.6.0 (2023-01-19)
------------------

* Add RCPS and CRC for multilabel-classification
* Add Top-Label calibration
* Fix bug for classification with very low scores

0.5.0 (2022-10-20)
------------------

* Add RAPS method for classification
* Add theoretical description for RAPS

0.4.2 (2022-09-02)
------------------

* Add tutorial for time series
* Convert existing tutorials in .py
* Add prefit method for CQR
* Add tutorial for CQR

0.4.1 (2022-06-27)
------------------

* Add `packaging` library in requirements
* Fix displaying problem in pypi

0.4.0 (2022-06-24)
------------------

* Relax and fix typing
* Add Split Conformal Quantile Regression
* Add EnbPI method for Time Series Regression
* Add EnbPI Documentation
* Add example with heteroscedastic data
* Add `ConformityScore` class that allows the user to define custom conformity scores

0.3.2 (2022-03-11)
------------------

* Refactorize unit tests
* Add "naive" and "top-k" methods in MapieClassifier
* Include J+aB method in regression tutorial
* Add MNIST example for classification
* Add cross-conformal for classification
* Add `notebooks` folder containing notebooks used for generating documentation tutorials
* Uniformize the use of matrix k_ and add an argument "ensemble" to method "predict" in regression.py
* Add replication of the Chen Xu's tutorial testing Jackknife+aB vs Jackknife+
* Add Jackknife+-after-Bootstrap documentation
* Improve scikit-learn pipelines compatibility

0.3.1 (2021-11-19)
------------------

* Add Jackknife+-after-Bootstrap method and add mean and median as aggregation functions
* Add "cumulative_score" method in MapieClassifier
* Allow image as input in MapieClassifier

0.3.0 (2021-09-10)
------------------

* Renaming estimators.py module to regression.py
* New classification.py module with MapieClassifier class, that estimates prediction sets from softmax score
* New set of unit tests for classification.py module
* Modification of the documentation architecture
* Split example gallery into separate regression and classification galleries
* Add first classification examples
* Add method classification_coverage_score in the module metrics.py
* Fixed code error for plotting of interval widths in tutorial of documentation
* Added missing import statements in tutorial of documentation
* Refactorize tests of `n_jobs` and `verbose` in `utils.py`

0.2.3 (2021-07-09)
------------------

* Inclusion in conda-forge with updated release checklist
* Add time series example
* Add epistemic uncertainty example
* Remove CicleCI redundancy with ReadTheDocs
* Remove Pep8speaks
* Include linting in CI/CD
* Use PyPa github actions for releases

0.2.2 (2021-06-10)
------------------

* Set alpha parameter as predict argument, with None as default value
* Switch to github actions for continuous integration of the code
* Add image explaining MAPIE internals on the README

0.2.1 (2021-06-04)
------------------

* Add `cv="prefit"` option
* Add sample_weight argument in fit method

0.2.0 (2021-05-21)
------------------

* Add n_jobs argument using joblib parallel processing
* Allow `cv` to take the value -1 equivalently to `LeaveOneOut()`
* Introduce the `cv` parameter to get closer to scikit-learn API
* Remove the `n_splits`, `shuffle` and `random_state` parameters
* Simplify the `method` parameter
* Fix typos in documentation and add methods descriptions in sphinx
* Accept alpha parameter as a list or np.ndarray. If alpha is an Iterable, `.predict()` returns a np.ndarray of shape (n_samples, 3, len(alpha)).

0.1.4 (2021-05-07)
------------------

* Move all alpha related operations to predict
* Assume default LinearRegression if estimator is None
* Improve documentation
* `return_pred` argument is now `ensemble` boolean

0.1.3 (2021-04-30)
------------------

* Update PyPi homepage
* Set up publication workflows as a github action
* Update issue and pull request templates
* Increase sklearn compatibility (coverage_score and unit tests)

0.1.2 (2021-04-27)
------------------

* First release on PyPi

0.1.1 (2021-04-27)
------------------

* First release on TestPyPi

0.1.0 (2021-04-27)
------------------

* Implement metrics.coverage
* Implement estimators.MapieRegressor
