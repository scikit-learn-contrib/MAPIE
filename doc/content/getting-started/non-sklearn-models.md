# Using models from other frameworks

MAPIE can use models from PyTorch, statsmodels, LightGBM, and other libraries
once they expose the small part of the scikit-learn estimator interface that
MAPIE needs. For regression, a wrapper must provide `fit(X, y)` and
`predict(X)`.

Inheriting from scikit-learn's `BaseEstimator` and `RegressorMixin` makes a
wrapper behave like a scikit-learn regressor. Its constructor should only store
parameters, while learned state should be stored in attributes ending in `_`.

The sections below define three wrappers. A single MAPIE workflow that works
with all three follows the wrapper definitions.

## PyTorch wrapper

A PyTorch wrapper converts NumPy arrays to tensors inside `fit` and `predict`,
then converts predictions back to a one-dimensional NumPy array.

```python
import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class TorchRegressor(RegressorMixin, BaseEstimator):
    """A cloneable scikit-learn wrapper for a small PyTorch network."""

    def __init__(
        self,
        hidden_size=32,
        max_epochs=100,
        learning_rate=0.01,
        random_state=42,
    ):
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        torch.manual_seed(self.random_state)

        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(self.n_features_in_, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, 1),
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate
        )
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y).reshape(-1, 1)

        self.model_.train()
        for _ in range(self.max_epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(
                self.model_(X_tensor), y_tensor
            )
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float32)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(X)).numpy().ravel()
```

This follows the same adapter pattern as the
[PyTorch language-model classifier wrapper](https://github.com/scikit-learn-contrib/MAPIE/blob/master/notebooks/educational-content/MAPIE_for_cosmosqa_correction.ipynb)
in MAPIE's educational content. A classifier wrapper additionally exposes
`classes_` and returns an `(n_samples, n_classes)` probability array from
`predict_proba`.

## statsmodels wrapper

statsmodels receives its data when a model is constructed, and its `fit` method
returns a separate results object. The following wrapper adapts ordinary least
squares to the scikit-learn convention:

```python
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class StatsmodelsOLSRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-compatible wrapper around statsmodels OLS."""

    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept

    def _prepare_exog(self, X):
        if self.add_intercept:
            X = sm.add_constant(X, has_constant="add")
        return X

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        model = sm.OLS(y, self._prepare_exog(X))
        self.results_ = model.fit()
        return self

    def predict(self, X):
        check_is_fitted(self, "results_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        return np.asarray(self.results_.predict(self._prepare_exog(X)))
```

The same structure can wrap `GLM`, `WLS`, or `QuantReg`: construct the chosen
statsmodels model inside `fit`, store its fitted results, and delegate `predict`
to those results. statsmodels is not a MAPIE dependency, so install it
separately before running this example:

```bash
pip install statsmodels
```

## Saved LightGBM Booster wrapper

LightGBM's `LGBMRegressor` already implements the scikit-learn interface and can
be passed directly to MAPIE. The lower-level `lightgbm.Booster`, including one
loaded from a model file or pickle, has `predict` and `refit` methods but no
`fit` method. This adapter supplies the missing interface:

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y


class LightGBMBoosterRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn adapter for an already-fitted LightGBM Booster."""

    def __init__(self, booster, decay_rate=0.9):
        self.booster = booster
        self.decay_rate = decay_rate

    @property
    def n_features_in_(self):
        booster = getattr(self, "booster_", self.booster)
        return booster.num_feature()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.booster_ = self.booster.refit(
            X,
            y,
            decay_rate=self.decay_rate,
        )
        return self

    def predict(self, X):
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        booster = getattr(self, "booster_", self.booster)
        return np.asarray(booster.predict(X))
```

The wrapper is immediately usable when `booster` is already fitted. Its `fit`
method delegates to `Booster.refit` if explicit refitting is needed.

## Regression with any wrapper

First create independent training, conformalization, and test sets:

```python
from mapie.utils import train_conformalize_test_split
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=500,
    n_features=2,
    noise=10.0,
    random_state=42,
)
X_train, X_conf, X_test, y_train, y_conf, y_test = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.6,
        conformalize_size=0.2,
        test_size=0.2,
        random_state=42,
    )
)
```

Choose one wrapper and assign the fitted estimator to `wrapped_regressor`:

=== "PyTorch"

    ```python
    wrapped_regressor = TorchRegressor().fit(X_train, y_train)
    ```

=== "statsmodels"

    ```python
    wrapped_regressor = StatsmodelsOLSRegressor().fit(X_train, y_train)
    ```

=== "Saved LightGBM Booster"

    ```python
    import lightgbm as lgb

    booster = lgb.Booster(model_file="model.txt")
    wrapped_regressor = LightGBMBoosterRegressor(booster)
    ```

The MAPIE code is then identical for all three wrappers:

```python
from mapie.regression import SplitConformalRegressor

mapie_regressor = SplitConformalRegressor(
    estimator=wrapped_regressor,
    confidence_level=0.9,
    prefit=True,
)
mapie_regressor.conformalize(X_conf, y_conf)
y_pred, y_intervals = mapie_regressor.predict_interval(X_test)
```

The conformalization set must not have been used to train or refit the wrapped
model. For a saved LightGBM Booster, use this adapter only with
`SplitConformalRegressor`. Cross-conformal methods need independent models
trained from scratch on each fold; refitting copies of an existing booster does
not provide that workflow.

## Time-series forecasting

MAPIE's `TimeSeriesRegressor` expects one prediction for every row of `X`.
Create time windows before passing the data to MAPIE so that every window is
one supervised row with one target. The following function includes lagged
target values and, when supplied, lagged exogenous variables:

```python
def make_supervised_windows(target, exogenous=None, n_lags=12):
    """Turn a series into rows of lagged target and exogenous values."""
    target = np.asarray(target, dtype=np.float32)
    if exogenous is None:
        history = target[:, None]
    else:
        exogenous = np.asarray(exogenous, dtype=np.float32)
        history = np.column_stack((target, exogenous))

    X = [
        history[end - n_lags : end].ravel()
        for end in range(n_lags, len(target))
    ]
    y = target[n_lags:]
    return np.asarray(X), y
```

Assuming `target` and `exogenous` are ordered in time, use a chronological
split and the PyTorch wrapper with block bootstrap:

```python
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

X, y = make_supervised_windows(target, exogenous, n_lags=12)
train_end = int(0.8 * len(X))
X_train, X_test = X[:train_end], X[train_end:]
y_train, y_test = y[:train_end], y[train_end:]

cv = BlockBootstrap(
    n_resamplings=10,
    n_blocks=10,
    overlapping=False,
    random_state=42,
)
mapie_ts = TimeSeriesRegressor(
    estimator=TorchRegressor(),
    method="enbpi",
    cv=cv,
    agg_function="mean",
)
mapie_ts.fit(X_train, y_train)

y_pred, y_intervals = mapie_ts.predict(
    X_test,
    confidence_level=0.9,
    ensemble=True,
    allow_infinite_bounds=True,
)
```

To use ACI with the same wrapper and supervised data, set `method="aci"` and
follow the update workflow shown in the
[time-series examples](../../generated/regression/index.md).

statsmodels models such as `ARIMA`, `SARIMAX`, and `ExponentialSmoothing`
forecast a number of steps rather than predicting one value for every row of
`X`. They therefore do not fit the wrappers above directly. MAPIE's current
time-series interface instead expects a supervised representation like the one
shown here.

## Wrapper checklist

- Keep `__init__` free of training logic and store every constructor argument
  unchanged so that scikit-learn can clone the wrapper.
- Return `self` from `fit` and store learned state in attributes ending in `_`.
- Make `predict(X)` return exactly one value per row of `X`.
- For classification, define `classes_` after fitting and return normalized
  class probabilities from `predict_proba(X)`.
- Use `prefit=True` when wrapping an already-trained model. Use `prefit=False`
  only when the wrapper implements training in `fit`.
