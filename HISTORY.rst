=======
History
=======

0.2.2 (2021-06-XX)
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
