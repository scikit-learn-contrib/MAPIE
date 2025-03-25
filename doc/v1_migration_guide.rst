Migrating to MAPIE v1
===========================================

MAPIE v1 introduces several updates, enhancements, and structural changes that simplify the API by breaking down ``MapieRegressor`` and ``MapieClassifier``  into dedicated classes for different conformal prediction techniques.

This guide outlines the differences between MAPIE v0.x and MAPIE v1 and provides instructions for migrating your code to the new API.

1. Overview of class restructuring
-----------------------------------

MAPIE v1 breaks down the ``MapieRegressor`` and ``MapieClassifier`` classes into 5 classes, each dedicated to a particular conformal prediction technique. ``MapieQuantileRegressor`` has also been revamped, and renamed ``ConformalizedQuantileRegressor``.

The rationale behind this is that ``MapieRegressor`` and ``MapieClassifier`` managed several conformal techniques under a single interface, which led to parameter redundancy and ambiguity. In MAPIE v1, each class includes only the relevant parameters specific to its technique.

The ``cv`` parameter is key to understand what new class to use in the v1 API:

.. list-table:: Mapie v0.x -> v1 classes correspondence
   :header-rows: 1

   * - v0.x class
     - ``cv`` parameter value
     - Corresponding v1 class
     - Conformal prediction type
   * - ``MapieRegressor``
     - ``"split"`` or ``"prefit"``
     - ``SplitConformalRegressor``
     - Split
   * - ``MapieRegressor``
     - ``None``, integer, or any ``sklearn.model_selection.BaseCrossValidator``
     - ``CrossConformalRegressor``
     - Cross
   * - ``MapieRegressor``
     - ``subsample.Subsample``
     - ``JackknifeAfterBootstrapRegressor``
     - Cross
   * - ``MapieQuantileRegressor``
     - ``None``, ``"split"`` or ``"prefit"``
     - ``ConformalizedQuantileRegressor``
     - Split
   * - ``MapieClassifier``
     - ``"split"`` or ``"prefit"``
     - ``SplitConformalClassifier``
     - Split
   * - ``MapieClassifier``
     - ``None``, integer, or any ``sklearn.model_selection.BaseCrossValidator``
     - ``CrossConformalClassifier``
     - Cross

For more details regarding the difference between split and cross conformal types, see :doc:`split_cross_conformal`

2. Method changes
-----------------

In MAPIE v1, the conformal prediction workflow is more streamlined and modular, with distinct methods for training, conformalization (named calibration in the scientific literature), and prediction. The conformalization process in v1 consists of four steps.

Step 1: Data splitting
~~~~~~~~~~~~~~~~~~~~~~
In v0.x, data splitting is handled by MAPIE.

In v1, the data splitting is left to the user for split conformal techniques. The user can split the data into training, conformalization, and test sets using scikit-learn's ``train_test_split`` or other methods.

Step 2 & 3: Model training and conformalization (ie: calibration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In v0.x, the ``fit`` method handled both model training and calibration.

In v1.0: MAPIE separates between training and calibration. We decided to name the *calibration* step *conformalization*, to avoid confusion with probability calibration.

For split conformal techniques:

``.fit()`` method:

- In v1, ``fit`` only trains the model on training data, without handling conformalization.
- Additional fitting parameters, like ``sample_weight``, should be included in ``fit_params``, keeping this method focused on training alone.

``.conformalize()`` method:

- Used in split methods only
- This new method performs conformalization after fitting, using separate conformity data ``(X_conformalize, y_conformalize)``.
- ``predict_params`` should be passed here

For cross conformal techniques:

``.fit_conformalize()`` method: because those techniques rely on fitting and conformalizing models in a cross-validation fashion, the fitting and conformalization steps are not distinct.

Step 4: Making predictions (``predict`` and ``predict_interval`` methods)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In MAPIE v0.x, both point predictions and prediction intervals were produced through the ``predict`` method.

MAPIE v1 introduces a new method for prediction, ``.predict_interval()``, that behaves like v0.x ``.predict(alpha=...)`` method. Namely, it predicts points and intervals.
The ``.predict()`` method now focuses solely on producing point predictions.



3. Parameters change
------------------------

``alpha``
~~~~~~~~~~~~~~~~~~~~
Indicates the desired coverage probability of the prediction intervals.

- **v0.x**: Specified as ``alpha`` during prediction, representing error rate.
- **v1**: Replaced with ``confidence_level`` to denote the coverage rate directly. Set at model initialization, improving consistency and clarity. ``confidence_level`` is equivalent to ``1 - alpha``.

``cv``
~~~~~~~
See the first section of this guide. The ``cv`` parameter is now only declared at cross conformal techniques initialization.

``conformity_score``
~~~~~~~~~~~~~~~~~~~~
A parameter used to specify the scoring approach for evaluating model predictions.

- **v0.x**: Only allowed subclass instances of ``BaseRegressionScore``, like AbsoluteConformityScore()
- **v1**: Now also accepts strings, like ``"absolute"``.

``method``
~~~~~~~~~~
Specifies the approach for calculating prediction intervals for cross conformal techniques.

- **v0.x**: Part of ``MapieRegressor``. Configured for the main prediction process.
- **v1**: Specific to ``CrossConformalRegressor`` and ``JackknifeAfterBootstrapRegressor``, indicating the interval calculation approach (``"base"``, ``"plus"``, or ``"minmax"``).

``groups``
~~~~~~~~~~~
The ``groups`` parameter is used to specify group labels for cross-validation, ensuring that the same group is not present in both training and conformity sets.

- **v0.x**: Passed as a parameter to the ``fit`` method.
- **v1**: The ``groups`` present is now only present in ``CrossConformalRegressor``. It is passed in the ``.conformalize()`` method instead of the ``.fit()`` method. In other classes (like ``SplitConformalRegressor``), groups can be directly handled by the user during data splitting.

``prefit``
~~~~~~~~~~
Controls whether the model has been pre-fitted before applying conformal prediction.

- **v0.x**: Indicated through ``cv="prefit"`` in ``MapieRegressor``.
- **v1**: ``prefit`` is now a separate boolean parameter, allowing explicit control over whether the model has been pre-fitted before conformalizing. It is set by default to ``True`` for ``SplitConformalRegressor``, as we believe this will become MAPIE nominal usage.

``fit_params`` (includes ``sample_weight``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dictionary of parameters specifically used during training, such as ``sample_weight`` in scikit-learn.

- **v0.x**: Passed additional parameters in a flexible but less explicit manner.
- **v1**: Now explicitly structured as a dedicated dictionary, ``fit_params``, ensuring parameters used during training are clearly defined and separated from other stages.

``predict_params``
~~~~~~~~~~~~~~~~~~
Defines additional parameters exclusively for prediction.

- **v0.x**: Passed additional parameters in a flexible but less explicit manner, sometimes mixed within training configurations.
- **v1**: Now structured as a dedicated dictionary, ``predict_params``, to be used during calibration (``conformalize`` method) and prediction stages, ensuring no overlap with training parameters.

``agg_function`` and ``ensemble``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
How to aggregate predictions in cross conformal methods.

- **v0.x**: Previously, the ``agg_function`` parameter had two usage: to aggregate predictions when setting ``ensemble=True`` in the ``predict`` method, and to specify the aggregation used in ``JackknifeAfterBootstrapRegressor``.
- **v1**:

  - The ``agg_function`` parameter has been split into two distinct parameters: ``aggregate_predictions`` and ``aggregation_method``. ``aggregate_predictions`` is specific to ``CrossConformalRegressor``, and it specifies how predictions from multiple conformal regressors are aggregated when making point predictions. ``aggregation_method`` is specific to ``JackknifeAfterBootstrapRegressor``, and it specifies the aggregation technique for combining predictions across different bootstrap samples during conformalization.
  - Note that for both cross conformal techniques, predictions points are now computed by default using mean aggregation. This is to avoid prediction points outside of prediction intervals in the default setting.

``random_state``
~~~~~~~~~~~~~~~~~~

- **v0.x**: This parameter was used to control the randomness of the data splitting.
- **v1**: This parameter has been removed in cases where data splitting is now manual. Future evolutions may reintroduce it as a general purpose randomness control parameter.

``symmetry``
~~~~~~~~~~~~~~~~~~

- **v0.x**: This parameter of the `predict` method of the MapieQuantileRegressor was set to True by default
- **v1**: This parameter is now named `symmetric_correction` and is set to False by default, because the resulting intervals are smaller. It is used in the `predict_interval` method of the ConformalizedQuantileRegressor.

``optimize_beta``
~~~~~~~~~~~~~~~~~~

This parameter used during interval prediction in regression has been renamed ``minimize_interval_width`` for clarity.

None defaults
~~~~~~~~~~~~~~~~~~~~
No more parameters with incorrect ``None`` defaults.

- **v0.x**: Eg: ``estimator`` had a ``None`` default value, even though the actual default value is ``LinearRegression()``. This was the case for other parameters as well.
- **v1**: All parameters now have explicit defaults.


4. Migration example: MAPIE v0.x to MAPIE v1
----------------------------------------------------------------------------------------

Below is a side-by-side example of code in MAPIE v0.x and its equivalent in MAPIE v1

Example 1: Split Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
############
Split conformal prediction is a widely used technique for generating prediction intervals, it splits the data into training, conformity, and test sets. The model is trained on the training set, calibrated on the conformity set, and then used to make predictions on the test set. In `MAPIE v1`, the `SplitConformalRegressor` replaces the older `MapieRegressor` with a more modular design and simplified API.

MAPIE v0.x Code
###############

Below is a MAPIE v0.x code for split conformal prediction in case of pre-fitted model:

.. testcode::

    from sklearn.linear_model import LinearRegression
    from mapie.regression import MapieRegressor
    from mapie.conformity_scores import ResidualNormalisedScore
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    X_train, X_conf_test, y_train, y_conf_test = train_test_split(X, y)
    X_conf, X_test, y_conf, y_test = train_test_split(X_conf_test, y_conf_test)

    prefit_model = LinearRegression().fit(X_train, y_train)

    v0 = MapieRegressor(
        estimator=prefit_model,
        cv="prefit",
        conformity_score=ResidualNormalisedScore()
    )

    v0.fit(X_conf, y_conf)

    prediction_points_v0, prediction_intervals_v0 = v0.predict(X_test, alpha=0.1)
    prediction_points_v0 = v0.predict(X_test)

Equivalent MAPIE v1 code
########################

Below is the equivalent MAPIE v1 code for split conformal prediction:

.. testcode::

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from mapie_v1.regression import SplitConformalRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

    X_train, X_conf_test, y_train, y_conf_test = train_test_split(X, y)
    X_conf, X_test, y_conf, y_test = train_test_split(X_conf_test, y_conf_test)

    prefit_model = LinearRegression().fit(X_train, y_train)

    v1 = SplitConformalRegressor(
        estimator=prefit_model,
        confidence_level=0.9,
        conformity_score="residual_normalized",
    )

    # Here we're not using v1.fit(), because the provided model is already fitted
    v1.conformalize(X_conf, y_conf)

    prediction_points_v1, prediction_intervals_v1 = v1.predict_interval(X_test)
    prediction_points_v1 = v1.predict(X_test)

Example 2: Cross-Conformal Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Description
############

Cross-conformal prediction extends split conformal prediction by using multiple cross-validation folds to improve the efficiency of the prediction intervals. In MAPIE v1, `CrossConformalRegressor`` replaces the older `MapieRegressor`` for this purpose.

MAPIE v0.x code
###############

Below is a MAPIE v0.x code for cross-conformal prediction:

.. testcode::

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from mapie.regression import MapieRegressor
    from sklearn.model_selection import train_test_split, GroupKFold
    from sklearn.datasets import make_regression

    X_full, y_full = make_regression(n_samples=100, n_features=2, noise=0.1)
    X, X_test, y, y_test = train_test_split(X_full, y_full)
    groups = np.random.randint(0, 10, X.shape[0])
    sample_weight = np.random.rand(X.shape[0])

    regression_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5
    )

    v0 = MapieRegressor(
        estimator=regression_model,
        cv=GroupKFold(),
        agg_function="median",
    )

    v0.fit(X, y, sample_weight=sample_weight, groups=groups)

    prediction_points_v0, prediction_intervals_v0 = v0.predict(X_test, alpha=0.1)
    prediction_points_v0 = v0.predict(X_test, ensemble=True)

Equivalent MAPIE v1 code
########################

Below is the equivalent MAPIE v1 code for cross-conformal prediction:

.. testcode::

    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, GroupKFold
    from mapie_v1.regression import CrossConformalRegressor
    from sklearn.datasets import make_regression

    X_full, y_full = make_regression(n_samples=100, n_features=2, noise=0.1)
    X, X_test, y, y_test = train_test_split(X_full, y_full)
    groups = np.random.randint(0, 10, X.shape[0])
    sample_weight = np.random.rand(X.shape[0])

    regression_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5
    )

    v1 = CrossConformalRegressor(
        estimator=regression_model,
        confidence_level=0.9,
        cv=GroupKFold(),
        conformity_score="absolute",
    )

    v1.fit_conformalize(X, y, groups=groups, fit_params={"sample_weight": sample_weight})

    prediction_points_v1, prediction_intervals_v1 = v1.predict_interval(X_test)
    prediction_points_v1 = v1.predict(X_test, aggregate_predictions="median")
