.. code-block:: python

    from lightgbm import LGBMRegressor
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import (TextArea, AnnotationBbox)
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
    from sklearn.datasets import fetch_california_housing
    from scipy.stats import randint, uniform


.. code-block:: python

    install_mapie = False
    if install_mapie is True:
        !pip install git+https://github.com/scikit-learn-contrib/MAPIE.git@add_prefit_cqr

.. code-block:: python

    from mapie.metrics import (
        regression_coverage_score,
        regression_mean_width_score
        )
    from mapie.quantile_regression import MapieQuantileRegressor

.. code-block:: python

    data = fetch_california_housing(as_frame=True)
    X = pd.DataFrame(data=data.data, columns=data.feature_names)
    y = pd.DataFrame(data=data.target)*100

.. code-block:: python

    y = y['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train,
        y_train,
    )

.. code-block:: python

    list_estimators = []
    estimator_low = LGBMRegressor(
        objective='quantile',
        alpha=0.05,
    )
    estimator_low.fit(X_train, y_train)
    list_estimators.append(estimator_low)
    
    estimator_high = LGBMRegressor(
        objective='quantile',
        alpha=0.95,
    )
    estimator_high.fit(X_train, y_train)
    list_estimators.append(estimator_high)
    
    
    estimator = LGBMRegressor(
        objective='quantile',
        alpha=0.5,
    )
    estimator.fit(X_train, y_train)
    list_estimators.append(estimator)


.. code-block:: python

    mapie = MapieQuantileRegressor(list_estimators, cv="prefit")
    mapie.fit(X_calib, y_calib)
    prefit_predict = mapie.predict(X_test)



    present issues as the upper quantile values might be higher than the
    lower quantile values.


.. code-block:: python

    mapie = MapieQuantileRegressor(estimator)
    mapie.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
    split_predict = mapie.predict(X_test)

.. code-block:: python

    for i in range(len(prefit_predict)):
        assert (prefit_predict[i]==split_predict[i]).all()

