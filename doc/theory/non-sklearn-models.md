# Using Non-scikit-learn Models (e.g. statsmodels)

MAPIE is *model agnostic*, but it expects the underlying model to expose the **scikit-learn estimator interface**: a `fit(X, y)` method that returns the estimator, and a `predict(X)` method that returns point predictions. Any model that follows this convention works with MAPIE — including models from libraries such as [statsmodels](https://www.statsmodels.org/), TensorFlow, or PyTorch — provided you wrap them in a thin scikit-learn-compatible adapter.

This page shows how to write such a wrapper for a statsmodels regression model, using ordinary least squares (`sm.OLS`) as an example. The same pattern applies to most statsmodels regression models that take `(endog, exog)` data, e.g. `GLM`, `WLS`, or `QuantReg`.

!!! note
    statsmodels is **not** a dependency of MAPIE, so the snippets below are not executed as part of the documentation build. Install statsmodels separately (`pip install statsmodels`) to run them.

---

## Why a wrapper is needed

statsmodels and scikit-learn follow different conventions:

| | scikit-learn | statsmodels |
|---|---|---|
| Data is passed | to `fit(X, y)` | to the model constructor, as `(endog, exog)` (i.e. `(y, X)`) |
| Fitting | mutates the estimator and returns `self` | returns a separate *results* object |
| Intercept | handled by the estimator | must be added explicitly with `sm.add_constant` |
| Prediction | `estimator.predict(X)` | `results.predict(exog)` |

MAPIE validates that the estimator it receives has `fit` and `predict` methods, and calls them the scikit-learn way. The wrapper below bridges the two conventions.

## Wrapping a statsmodels regression model

```python
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class StatsmodelsOLSRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn-compatible wrapper around statsmodels OLS."""

    def __init__(self, add_intercept=True):
        # Per scikit-learn conventions, __init__ only stores parameters.
        self.add_intercept = add_intercept

    def _prepare_exog(self, X):
        X = np.asarray(X)
        if self.add_intercept:
            # has_constant="add" ensures a constant column is always added,
            # even if a feature happens to be constant in a small batch.
            X = sm.add_constant(X, has_constant="add")
        return X

    def fit(self, X, y):
        # statsmodels takes the data at model construction: (endog, exog).
        # Build the model here so that __init__ stays parameter-only.
        self.results_ = sm.OLS(np.asarray(y), self._prepare_exog(X)).fit()
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "results_")
        return np.asarray(self.results_.predict(self._prepare_exog(X)))
```

A few points worth noting:

- **`__init__` only stores hyperparameters.** Inheriting from `sklearn.base.BaseEstimator` then provides `get_params` / `set_params` for free, which makes the wrapper usable in pipelines, grid searches, and cloning.
- **The statsmodels model is built inside `fit`**, because statsmodels models take the data `(endog, exog)` at construction time, whereas scikit-learn estimators receive the data in `fit`.
- **The fitted results are stored in attributes with a trailing underscore** (`results_`, `n_features_in_`), following the scikit-learn convention for fitted attributes, and `predict` raises a clear error if called before `fit` (via `check_is_fitted`). Setting `n_features_in_` also lets MAPIE recognize the wrapper as fitted when you use `prefit=True` (otherwise it emits a `UserWarning`).
- **`RegressorMixin`** marks the wrapper as a regressor and provides a default `score` method.

To wrap another statsmodels model, expose its constructor and `fit` options as `__init__` parameters, for example:

```python
class StatsmodelsGLMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, family=None):
        self.family = family

    def fit(self, X, y):
        family = self.family if self.family is not None else sm.families.Gaussian()
        self.results_ = sm.GLM(
            np.asarray(y), sm.add_constant(np.asarray(X), has_constant="add"),
            family=family,
        ).fit()
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "results_")
        return np.asarray(
            self.results_.predict(sm.add_constant(np.asarray(X), has_constant="add"))
        )
```

## Using the wrapper with MAPIE

Once wrapped, the model can be used like any scikit-learn regressor, e.g. with [`SplitConformalRegressor`](../api/regression.md):

```python
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=2, noise=1.0)
(
    X_train, X_conformalize, X_test,
    y_train, y_conformalize, y_test,
) = train_conformalize_test_split(
    X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
)

mapie_regressor = SplitConformalRegressor(
    estimator=StatsmodelsOLSRegressor(),
    confidence_level=0.95,
    prefit=False,
)
mapie_regressor.fit(X_train, y_train)
mapie_regressor.conformalize(X_conformalize, y_conformalize)

predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)
```

If you have already fitted the wrapper yourself, pass it with `prefit=True` (the default) and skip the `fit` step:

```python
model = StatsmodelsOLSRegressor().fit(X_train, y_train)

mapie_regressor = SplitConformalRegressor(
    estimator=model, confidence_level=0.95, prefit=True
)
mapie_regressor.conformalize(X_conformalize, y_conformalize)

predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)
```

## What about time-series models (ARIMA, ExponentialSmoothing)?

statsmodels time-series models such as `ARIMA`, `SARIMAX`, or `ExponentialSmoothing` do **not** map cleanly onto the `(X, y)` interface: they are fitted on the target series alone (plus, optionally, exogenous variables) and forecast a number of *steps ahead* rather than predicting from a feature matrix `X`. A wrapper like the one above is therefore not directly applicable.

For time-series forecasting, MAPIE provides [`TimeSeriesRegressor`](../api/regression.md), which implements EnbPI and adaptive conformal inference (ACI). It works with scikit-learn-style regressors applied to exogenous and/or lagged features — see the [time-series examples](../generated/regression/index.md) for a complete workflow.

Native support for statsmodels (and other non-scikit-learn) time-series models is not available yet; it awaits a rework of MAPIE's model backend. Progress is tracked in the related discussions on GitHub — see [issue #884](https://github.com/scikit-learn-contrib/MAPIE/issues/884).
