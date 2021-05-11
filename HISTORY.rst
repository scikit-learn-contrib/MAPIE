=======
History
=======

0.2.0 (???)
------------------

* Allow `cv` to take the value -1 equivalently to `LeaveOneOut()`
* Introduce the `cv` parameter to get closer to scikit-learn API
* Remove the `n_splits`, `shuffle` and `random_state` parameters
* Simplify the `method` parameter
* Fix typos in documentation and add methods descriptions in sphinx
* Add n_jobs argument using joblib parallel processing

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