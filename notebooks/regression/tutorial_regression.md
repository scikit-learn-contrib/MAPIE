---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: mapie_local
    language: python
    name: mapie_local
---

# Tutorial for regression


In this tutorial, we compare the prediction intervals estimated by MAPIE on a 
simple, one-dimensional, ground truth function

$$
f(x) = x \sin(x)
$$

Throughout this tutorial, we will answer the following questions:

- How well do the MAPIE strategies capture the aleatoric uncertainty existing in the data?

- How do the prediction intervals estimated by the resampling strategies
  evolve for new *out-of-distribution* data? 

- How do the prediction intervals vary between regressor models?

Throughout this tutorial, we estimate the prediction intervals first using 
a polynomial function, and then using a boosting model, and a simple neural network. 

**For practical problems, we advise using the faster CV+ strategies. 
For conservative prediction interval estimates, you can alternatively 
use the CV-minmax strategies.**



## 1. Estimating the aleatoric uncertainty of homoscedastic noisy data


Let's start by defining the $x \times \sin(x)$ function and another simple function
that generates one-dimensional data with normal noise uniformely in a given interval.

```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
def x_sinx(x):
    """One-dimensional x*sin(x) function."""
    return x*np.sin(x)
```

```python
def get_1d_data_with_constant_noise(funct, min_x, max_x, n_samples, noise):
    """
    Generate 1D noisy data uniformely from the given function 
    and standard deviation for the noise.
    """
    np.random.seed(59)
    X_train = np.linspace(min_x, max_x, n_samples)
    np.random.shuffle(X_train)
    X_test = np.linspace(min_x, max_x, n_samples*5)
    y_train, y_mesh, y_test = funct(X_train), funct(X_test), funct(X_test)
    y_train += np.random.normal(0, noise, y_train.shape[0])
    y_test += np.random.normal(0, noise, y_test.shape[0])
    return X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh
```

We first generate noisy one-dimensional data uniformely on an interval. 
Here, the noise is considered as *homoscedastic*, since it remains constant 
over $x$.

```python
min_x, max_x, n_samples, noise = -5, 5, 100, 0.5
X_train, y_train, X_test, y_test, y_mesh = get_1d_data_with_constant_noise(
    x_sinx, min_x, max_x, n_samples, noise
)
```

Let's visualize our noisy function. 

```python
import matplotlib.pyplot as plt
plt.xlabel("x") ; plt.ylabel("y")
plt.scatter(X_train, y_train, color="C0")
_ = plt.plot(X_test, y_mesh, color="C1")
```

As mentioned previously, we fit our training data with a simple
polynomial function. Here, we choose a degree equal to 10 so the function 
is able to perfectly fit $x \times \sin(x)$.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

degree_polyn = 10
polyn_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=degree_polyn)),
        ("linear", LinearRegression())
    ]
)
```

We then estimate the prediction intervals for all the strategies very easily with a
`fit` and `predict` process. The prediction interval's lower and upper bounds
are then saved in a DataFrame. Here, we set an alpha value of 0.05
in order to obtain a 95% confidence for our prediction intervals.

```python
from typing import Union
from typing_extensions import TypedDict
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample 
Params = TypedDict("Params", {"method": str, "cv": Union[int, Subsample]})
STRATEGIES = {
    "naive": Params(method="naive"),
    "jackknife": Params(method="base", cv=-1),
    "jackknife_plus": Params(method="plus", cv=-1),
    "jackknife_minmax": Params(method="minmax", cv=-1),
    "cv": Params(method="base", cv=10),
    "cv_plus": Params(method="plus", cv=10),
    "cv_minmax": Params(method="minmax", cv=10),
    "jackknife_plus_ab": Params(method="plus", cv=Subsample(n_resamplings=50)),
    "jackknife_minmax_ab": Params(method="minmax", cv=Subsample(n_resamplings=50)),
}
y_pred, y_pis = {}, {}
for strategy, params in STRATEGIES.items():
    mapie = MapieRegressor(polyn_model, **params)
    mapie.fit(X_train, y_train)
    y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.05)
```

Let’s now compare the confidence intervals with the predicted intervals with obtained 
by the Jackknife+, Jackknife-minmax, CV+, CV-minmax, Jackknife+-after-Boostrap, and Jackknife-minmax-after-Bootstrap strategies. Note that for the Jackknife-after-Bootstrap method, we call the :class:`mapie.subsample.Subsample` object that allows us to train bootstrapped models.

```python
def plot_1d_data(
    X_train,
    y_train, 
    X_test,
    y_test,
    y_sigma,
    y_pred, 
    y_pred_low, 
    y_pred_up,
    ax=None,
    title=None
):
    ax.set_xlabel("x") ; ax.set_ylabel("y")
    ax.fill_between(X_test, y_pred_low, y_pred_up, alpha=0.3)
    ax.scatter(X_train, y_train, color="red", alpha=0.3, label="Training data")
    ax.plot(X_test, y_test, color="gray", label="True confidence intervals")
    ax.plot(X_test, y_test - y_sigma, color="gray", ls="--")
    ax.plot(X_test, y_test + y_sigma, color="gray", ls="--")
    ax.plot(X_test, y_pred, color="blue", alpha=0.5, label="Prediction intervals")
    if title is not None:
        ax.set_title(title)
    ax.legend()
```

```python
strategies = ["jackknife_plus", "jackknife_minmax" , "cv_plus", "cv_minmax", "jackknife_plus_ab", "jackknife_minmax_ab"]
n_figs = len(strategies)
fig, axs = plt.subplots(3, 2, figsize=(9, 13))
coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]
for strategy, coord in zip(strategies, coords):
    plot_1d_data(
        X_train.ravel(),
        y_train.ravel(),
        X_test.ravel(),
        y_mesh.ravel(),
        1.96*noise,
        y_pred[strategy].ravel(),
        y_pis[strategy][:, 0, 0].ravel(),
        y_pis[strategy][:, 1, 0].ravel(),
        ax=coord,
        title=strategy
    )
```

At first glance, the four strategies give similar results and the
prediction intervals are very close to the true confidence intervals.
Let’s confirm this by comparing the prediction interval widths over
$x$ between all strategies.

```python
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.axhline(1.96*2*noise, ls="--", color="k", label="True width")
for strategy in STRATEGIES:
    ax.plot(X_test, y_pis[strategy][:, 1, 0] - y_pis[strategy][:, 0, 0], label=strategy)
ax.set_xlabel("x")
ax.set_ylabel("Prediction Interval Width")
_ = ax.legend(fontsize=10, loc=[1, 0.4])
```

As expected, the prediction intervals estimated by the Naive method
are slightly too narrow. The Jackknife, Jackknife+, CV, CV+, JaB, and J+aB give
similar widths that are very close to the true width. On the other hand,
the widths estimated by Jackknife-minmax and CV-minmax are slightly too
wide. Note that the widths given by the Naive, Jackknife, and CV strategies
are constant because there is a single model used for prediction,
perturbed models are ignored at prediction time.


Let’s now compare the *effective* coverage, namely the fraction of test
points whose true values lie within the prediction intervals, given by
the different strategies. 

```python
import pandas as pd
from mapie.metrics import regression_coverage_score
pd.DataFrame([
    [
        regression_coverage_score(
            y_test, y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
        ),
        (
            y_pis[strategy][:, 1, 0] - y_pis[strategy][:, 0, 0]
        ).mean()
    ] for strategy in STRATEGIES
], index=STRATEGIES, columns=["Coverage", "Width average"]).round(2)
```

All strategies except the Naive one give effective coverage close to the expected 
0.95 value (recall that alpha = 0.05), confirming the theoretical garantees.


## 2. Estimating the epistemic uncertainty of out-of-distribution data


Let’s now consider one-dimensional data without noise, but normally distributed.
The goal is to explore how the prediction intervals evolve for new data 
that lie outside the distribution of the training data in order to see how the strategies
can capture the *epistemic* uncertainty. 
For a comparison of the epistemic and aleatoric uncertainties, please have a look at this
[source](https://en.wikipedia.org/wiki/Uncertainty_quantification).


Lets" start by generating and showing the data. 

```python
def get_1d_data_with_normal_distrib(funct, mu, sigma, n_samples, noise):
    """
    Generate noisy 1D data with normal distribution from given function 
    and noise standard deviation.
    """
    np.random.seed(59)
    X_train = np.random.normal(mu, sigma, n_samples)
    X_test = np.arange(mu-4*sigma, mu+4*sigma, sigma/20.)
    y_train, y_mesh, y_test = funct(X_train), funct(X_test), funct(X_test)
    y_train += np.random.normal(0, noise, y_train.shape[0])
    y_test += np.random.normal(0, noise, y_test.shape[0])
    return X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh
```

```python
mu = 0 ; sigma = 2 ; n_samples = 300 ; noise = 0.
X_train, y_train, X_test, y_test, y_mesh = get_1d_data_with_normal_distrib(
    x_sinx, mu, sigma, n_samples, noise
)
```

```python
plt.xlabel("x") ; plt.ylabel("y")
plt.scatter(X_train, y_train, color="C0")
_ = plt.plot(X_test, y_test, color="C1")
```

As before, we estimate the prediction intervals using a polynomial
function of degree 10 and show the results for the Jackknife+ and CV+
strategies.

```python
Params = TypedDict("Params", {"method": str, "cv": Union[int, Subsample]})
STRATEGIES = {
    "naive": Params(method="naive"),
    "jackknife": Params(method="base", cv=-1),
    "jackknife_plus": Params(method="plus", cv=-1),
    "jackknife_minmax": Params(method="minmax", cv=-1),
    "cv": Params(method="base", cv=10),
    "cv_plus": Params(method="plus", cv=10),
    "cv_minmax": Params(method="minmax", cv=10),
    "jackknife_plus_ab": Params(method="plus", cv=Subsample(n_resamplings=50)),
    "jackknife_minmax_ab": Params(method="minmax", cv=Subsample(n_resamplings=50)),
}
y_pred, y_pis = {}, {}
for strategy, params in STRATEGIES.items():
    mapie = MapieRegressor(polyn_model, **params)
    mapie.fit(X_train, y_train)
    y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.05)
```

```python
strategies = ["jackknife_plus", "jackknife_minmax" , "cv_plus", "cv_minmax", "jackknife_plus_ab", "jackknife_minmax_ab"]
n_figs = len(strategies)
fig, axs = plt.subplots(3, 2, figsize=(9, 13))
coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]
for strategy, coord in zip(strategies, coords): 
    plot_1d_data(
        X_train.ravel(),
        y_train.ravel(), 
        X_test.ravel(),
        y_mesh.ravel(),
        1.96*noise, 
        y_pred[strategy].ravel(),
        y_pis[strategy][:, 0, :].ravel(),
        y_pis[strategy][:, 1, :].ravel(), 
        ax=coord,
        title=strategy
    )
```

At first glance, our polynomial function does not give accurate
predictions with respect to the true function when $|x > 6|$. 
The prediction intervals estimated with the Jackknife+ do not seem to 
increase significantly, unlike the CV+ method whose prediction intervals
capture a high uncertainty when $x > 6$.


Let's now compare the prediction interval widths between all strategies. 


```python
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_yscale("log")
for strategy in STRATEGIES:
    ax.plot(X_test, y_pis[strategy][:, 1, 0] - y_pis[strategy][:, 0, 0], label=strategy)
ax.set_xlabel("x")
ax.set_ylabel("Prediction Interval Width")
ax.legend(fontsize=10, loc=[1, 0.4]);
```

The prediction interval widths start to increase exponentially
for $|x| > 4$ for the Jackknife-minmax, CV+, and CV-minmax
strategies. On the other hand, the prediction intervals estimated by
Jackknife+ remain roughly constant until $|x| \sim 5$ before
increasing.

```python
pd.DataFrame([
    [
        regression_coverage_score(
            y_test, y_pis[strategy][:, 0, 0], y_pis[strategy][:, 1, 0]
        ),
        (
            y_pis[strategy][:, 1, 0] - y_pis[strategy][:, 0, 0]
        ).mean()
    ] for strategy in STRATEGIES
], index=STRATEGIES, columns=["Coverage", "Width average"]).round(3)
```

In conclusion, the Jackknife-minmax, CV+, CV-minmax, or Jackknife-minmax-ab strategies are more
conservative than the Jackknife+ strategy, and tend to result in more
reliable coverages for *out-of-distribution* data. It is therefore
advised to use the three former strategies for predictions with new
out-of-distribution data.
Note however that there are no theoretical guarantees on the coverage level 
for out-of-distribution data.


## 3. Estimating the uncertainty with different sklearn-compatible regressors


MAPIE can be used with any kind of sklearn-compatible regressor. Here, we
illustrate this by comparing the prediction intervals estimated by the CV+ method using
different models:

- the same polynomial function as before.
 
- a XGBoost model using the Scikit-learn API.

- a simple neural network, a Multilayer Perceptron with three dense layers, using the KerasRegressor wrapper.

Once again, let’s use our noisy one-dimensional data obtained from a
uniform distribution.

```python
min_x, max_x, n_samples, noise = -5, 5, 100, 0.5
X_train, y_train, X_test, y_test, y_mesh = get_1d_data_with_constant_noise(
    x_sinx, min_x, max_x, n_samples, noise
)
```

```python
plt.xlabel("x") ; plt.ylabel("y")
plt.plot(X_test, y_mesh, color="C1")
_ = plt.scatter(X_train, y_train)
```

Let's then define the models. The boosing model considers 100 shallow trees with a max depth of 2 while
the Multilayer Perceptron has two hidden dense layers with 20 neurons each followed by a relu activation.


```python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # disable debugging logs from Tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
def mlp():
    """
    Two-layer MLP model
    """
    model = Sequential([
        Dense(units=20, input_shape=(1,), activation="relu"),
        Dense(units=20, activation="relu"),
        Dense(units=1)
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model
```

```python
polyn_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=degree_polyn)),
        ("linear", LinearRegression(fit_intercept=False))
    ]
)
```

```python
from xgboost import XGBRegressor
xgb_model = XGBRegressor(
    max_depth=2,
    n_estimators=100,
    tree_method="hist",
    random_state=59,
    learning_rate=0.1,
    verbosity=0,
    nthread=-1
)
mlp_model = KerasRegressor(
    build_fn=mlp, 
    epochs=500, 
    verbose=0
)
```

Let's now use MAPIE to estimate the prediction intervals using the CV+ method 
and compare their prediction interval.

```python
models = [polyn_model, xgb_model, mlp_model]
model_names = ["polyn", "xgb", "mlp"]
prediction_interval = {}
for name, model in zip(model_names, models):
    mapie = MapieRegressor(model, method="plus", cv=5)
    mapie.fit(X_train, y_train)
    y_pred[name], y_pis[name] = mapie.predict(X_test, alpha=0.05)
```

```python
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
for name, ax in zip(model_names, axs):
    plot_1d_data(
        X_train.ravel(),
        y_train.ravel(),
        X_test.ravel(),
        y_mesh.ravel(),
        1.96*noise,
        y_pred[name].ravel(),
        y_pis[name][:, 0, 0].ravel(),
        y_pis[name][:, 1, 0].ravel(),
        ax=ax,
        title=name
    )
```

```python
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
for name in model_names:
    ax.plot(X_test, y_pis[name][:, 1, 0] - y_pis[name][:, 0, 0])
ax.axhline(1.96*2*noise, ls="--", color="k")
ax.set_xlabel("x")
ax.set_ylabel("Prediction Interval Width")
ax.legend(model_names + ["True width"], fontsize=8);
```

As expected with the CV+ method, the prediction intervals are a bit 
conservative since they are slightly wider than the true intervals.
However, the CV+ method on the three models gives very promising results 
since the prediction intervals closely follow the true intervals with $x$. 
