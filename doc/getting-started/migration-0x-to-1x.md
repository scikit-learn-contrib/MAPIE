# MAPIE 0.x → 1.x Migration Guide

## Overview

MAPIE 1.x introduced a significant architectural redesign for conformal regression APIs.

The previous unified `MapieRegressor` class used in MAPIE 0.x was replaced by specialized conformal regression classes in MAPIE 1.x.

This guide helps users migrate older notebooks, Kaggle examples, tutorials, and production pipelines.

## Class Mapping

+--------------------------------------+--------------------------------------+
| MAPIE 0.x                            | MAPIE 1.x                            |
+======================================+======================================+
| MapieRegressor(method='plus')        | CrossConformalRegressor              |
+--------------------------------------+--------------------------------------+
| MapieRegressor(method='base')        | SplitConformalRegressor              |
+--------------------------------------+--------------------------------------+
| method='quantile'                    | ConformalizedQuantileRegressor       |
+--------------------------------------+--------------------------------------+
| bootstrap-based conformal            | JackknifeAfterBootstrapRegressor     |
+--------------------------------------+--------------------------------------+

## Old API Example (0.x)

.. code-block:: python

```
from mapie.regression import MapieRegressor

mapie = MapieRegressor(
    estimator=model,
    method='plus',
    cv=5
)

mapie.fit(X_train, y_train)

y_pred, y_pis = mapie.predict(X_test, alpha=0.10)
```

## New API Example (1.x)

.. code-block:: python

```
from mapie.regression import CrossConformalRegressor

mapie = CrossConformalRegressor(
    estimator=model,
    method='plus',
    cv=5,
    confidence_level=0.90
)

mapie.fit(X_train, y_train)

y_pred, y_pis = mapie.predict(X_test)
```

## Parameter Changes

+---------------------+----------------------+
| MAPIE 0.x           | MAPIE 1.x            |
+=====================+======================+
| alpha=0.10          | confidence_level=0.90|
+---------------------+----------------------+

## Prediction Interval Shape Changes

In MAPIE 0.x:

.. code-block:: python

```
lower = y_pis[:, 0, 0]
upper = y_pis[:, 1, 0]
```

In MAPIE 1.x:

.. code-block:: python

```
lower = y_pis[:, 0]
upper = y_pis[:, 1]
```

## Notebook / Kaggle Installation Warning

When installing or downgrading MAPIE inside Jupyter, Kaggle, or Colab notebooks:

.. code-block:: python

```
pip install mapie==<version>
```

Python may continue using the already-imported MAPIE module from `sys.modules`.

This can create confusion where:

* pip reports successful installation
* but `mapie.__version__` still shows the old version

Recommended workflow:

.. code-block:: python

```
# install package
pip install mapie==<version>

# restart notebook kernel before importing again
```

## Environment Notes

MAPIE 0.x may not fully support Python 3.12 environments.

Users running:

* Kaggle
* Colab
* Python 3.12+

are encouraged to migrate toward MAPIE 1.x APIs instead of downgrading older versions.
