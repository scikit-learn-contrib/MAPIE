---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: Python 3.10.4 ('mapie-dev')
    language: python
    name: python3
---

```python id="AyjfITskyO-k"
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, AnnotationBbox)
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.datasets import fetch_california_housing
from scipy.stats import randint, uniform

```

```python colab={"base_uri": "https://localhost:8080/"} id="3NHXiRcByUXB" outputId="78a81275-6c22-4617-caa0-ae233a8f6c7e"
install_mapie = False
if install_mapie is True:
    !pip install git+https://github.com/scikit-learn-contrib/MAPIE.git@add_prefit_cqr
```

```python id="It2WS2JLyR6F"
from mapie.metrics import (
    regression_coverage_score,
    regression_mean_width_score
    )
from mapie.quantile_regression import MapieQuantileRegressor
```

```python id="vAx4udWgzgNa"
data = fetch_california_housing(as_frame=True)
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.DataFrame(data=data.target)*100
```

```python id="EjWL4IdLz5wL"
y = y['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train,
    y_train,
)
```

```python id="pHPFpYhYzj9c"
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

```

```python colab={"base_uri": "https://localhost:8080/", "height": 427} id="qMR7PcQdzw1d" outputId="e50b458b-ba49-45d2-9dd7-b095658312b5"
mapie = MapieQuantileRegressor(list_estimators, cv="prefit")
mapie.fit(X_calib, y_calib)
prefit_predict = mapie.predict(X_test)
```

```python id="v0iCmVf10xv0"
mapie = MapieQuantileRegressor(estimator)
mapie.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
split_predict = mapie.predict(X_test)
```

```python
for i in range(len(prefit_predict)):
    assert (prefit_predict[i]==split_predict[i]).all()
```

```python

```
