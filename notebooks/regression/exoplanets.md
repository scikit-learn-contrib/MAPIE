---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: mapie_local
    language: python
    name: mapie_local
---

# Estimating the uncertainties in the exoplanet masses


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-learn-contrib/MAPIE/blob/master/notebooks/regression/exoplanets.ipynb)



In this notebook, we quantify the uncertainty in exoplanet masses predicted by several machine learning models, based on the exoplanet properties. To this aim, we use the exoplanet dataset downloaded from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and estimate the prediction intervals using the methods implemented in MAPIE.

```python
install_mapie = True
if install_mapie:
    !pip install mapie
```

```python
from typing_extensions import TypedDict
from typing import Union
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler
)
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mapie.metrics import regression_coverage_score
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample

warnings.filterwarnings("ignore")
```

## 1. Data Loading


Let's start by loading the `exoplanets` dataset and looking at the main information.

```python
url_file = "https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/master/notebooks/regression/exoplanets_mass.csv"
exo_df = pd.read_csv(url_file, index_col=0)
```

```python
exo_df.info()
```

The dataset contains 21 features giving complementary information about the properties of the discovered planet, the star around which the planet revolves, together with the type of discovery method. 7 features are categorical, and 14 are continuous.


Some properties show high variance among exoplanets and stars due to the astronomical nature of such systems. We therefore decide to use a log transformation for the following features to approach a normal distribution.

```python
exo_df["Stellar_Mass_[Solar_mass]"] = exo_df["Stellar_Mass_[Solar_mass]"].replace(0, np.nan)
vars2log = [
    "Planet_Orbital_Period_[day]",
    "Planet_Orbital_SemiMajorAxis_[day]",
    "Planet_Radius_[Earth_radius]",
    "Planet_Mass_[Earth_mass]",
    "Stellar_Radius_[Solar_radius]",
    "Stellar_Mass_[Solar_mass]",
    "Stellar_Effective_Temperature_[K]"
]
for var in vars2log:
    exo_df[var+"_log"] = np.log(exo_df[var])
```

```python
vars2keep = list(set(exo_df.columns) - set(vars2log))
exo_df = exo_df[vars2keep]
```

```python
exo_df.head()
```

Throughout this tutorial, the target variable will be `Planet_Mass_[Earth_mass]_log`.

```python
target = "Planet_Mass_[Earth_mass]_log"
```

```python
num_cols = list(exo_df.columns[exo_df.dtypes == "float64"])
cat_cols = list(exo_df.columns[exo_df.dtypes != "float64"])
exo_df[cat_cols] = exo_df[cat_cols].astype(str)
```

```python
planet_cols = [col for col in num_cols if "Planet_" in col]
star_cols = [col for col in num_cols if "Stellar_" in col]
system_cols = [col for col in num_cols if "System_" in col]
```

## 2. Data visualization

```python
sns.pairplot(exo_df[planet_cols])
```

```python
sns.pairplot(exo_df[star_cols])
```

## 3. Data preprocessing


In this section, we perform a simple preprocessing of the dataset in order to impute the missing values and encode the categorical features.

```python
endos = list(set(exo_df.columns) - set([target]))
X = exo_df[endos]
y = exo_df[target]
```

```python
num_cols = list(X.columns[X.dtypes == "float64"])
cat_cols = list(X.columns[X.dtypes != "float64"])
X[cat_cols] = X[cat_cols].astype(str)
```

```python
imputer_num = SimpleImputer(strategy="mean")
scaler_num = RobustScaler()
imputer_cat = SimpleImputer(strategy="constant", fill_value=-1)
encoder_cat = OneHotEncoder(
    categories="auto",
    drop=None,
    sparse=False,
    handle_unknown="ignore",
)
```

```python
numerical_transformer = Pipeline(
    steps=[("imputer", imputer_num), ("scaler", scaler_num)]
)
categorical_transformer = Pipeline(
    steps=[("ordinal", OrdinalEncoder()), ("imputer", imputer_cat), ("encoder", encoder_cat)]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_transformer, num_cols),
        ("categorical", categorical_transformer, cat_cols)
    ],
    remainder="drop",
    sparse_threshold=0,
)
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
```

```python
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
```

## 4. First estimation of the uncertainties with MAPIE


### Uncertainty estimation


Here, we build our first prediction intervals with MAPIE. To this aim, we adopt the CV+ strategy with 5 folders, using `method="plus"` and `cv=KFold(n_splits=5, shuffle=True)` as input arguments.

```python
def get_regressor(name):
    if name == "linear":
        mdl = LinearRegression()
    elif name == "polynomial":
        degree_polyn = 2
        mdl = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=degree_polyn)),
                ("linear", LinearRegression())
            ]
        )
    elif name == "random_forest":
        mdl = RandomForestRegressor()
    return mdl
```

```python
mdl = get_regressor("random_forest")
```

```python
mapie = MapieRegressor(mdl, method="plus", cv=KFold(n_splits=5, shuffle=True))
```

```python
mapie.fit(X_train, y_train)
```

We build prediction intervals for a range of alpha values between 0 and 1.

```python
alpha = np.arange(0.05, 1, 0.05)
y_train_pred, y_train_pis = mapie.predict(X_train, alpha=alpha)
y_test_pred, y_test_pis = mapie.predict(X_test, alpha=alpha)
```

### Visualization


The following function offers to visualize the error bars estimated by MAPIE for the selected method and the given confidence level.

```python
def plot_predictionintervals(
    y_train,
    y_train_pred,
    y_train_pred_low,
    y_train_pred_high,
    y_test,
    y_test_pred,
    y_test_pred_low,
    y_test_pred_high,
    suptitle: str,
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    ax1.errorbar(
        x=y_train,
        y=y_train_pred,
        yerr=(y_train_pred - y_train_pred_low, y_train_pred_high - y_train_pred),
        alpha=0.8,
        label="train",
        fmt=".",
    )
    ax1.errorbar(
        x=y_test,
        y=y_test_pred,
        yerr=(y_test_pred - y_test_pred_low, y_test_pred_high - y_test_pred),
        alpha=0.8,
        label="test",
        fmt=".",
    )
    ax1.plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        color="gray",
        alpha=0.5,
    )
    ax1.set_xlabel("True values", fontsize=12)
    ax1.set_ylabel("Predicted values", fontsize=12)
    ax1.legend()
    
    ax2.scatter(
        x=y_train, y=y_train_pred_high - y_train_pred_low, alpha=0.8, label="train", marker="."
    )
    ax2.scatter(x=y_test, y=y_test_pred_high - y_test_pred_low, alpha=0.8, label="test", marker=".")
    ax2.set_xlabel("True values", fontsize=12)
    ax2.set_ylabel("Interval width", fontsize=12)
    ax2.set_xscale("linear")
    ax2.set_ylim([0, np.max(y_test_pred_high - y_test_pred_low)*1.1])
    ax2.legend()
    std_all = np.concatenate([
        y_train_pred_high - y_train_pred_low, y_test_pred_high - y_test_pred_low
    ])
    type_all = np.array(["train"] * len(y_train) + ["test"] * len(y_test))
    x_all = np.arange(len(std_all))
    order_all = np.argsort(std_all)
    std_order = std_all[order_all]
    type_order = type_all[order_all]
    ax3.scatter(
        x=x_all[type_order == "train"],
        y=std_order[type_order == "train"],
        alpha=0.8,
        label="train",
        marker=".",
    )
    ax3.scatter(
        x=x_all[type_order == "test"],
        y=std_order[type_order == "test"],
        alpha=0.8,
        label="test",
        marker=".",
    )
    ax3.set_xlabel("Order", fontsize=12)
    ax3.set_ylabel("Interval width", fontsize=12)
    ax3.legend()
    ax1.set_title("True vs predicted values")
    ax2.set_title("Prediction interval width vs true values")
    ax3.set_title("Ordered prediction interval width")
    plt.suptitle(suptitle, size=20)
    plt.show()

```

```python
alpha_plot = int(np.where(alpha == 0.1)[0])
plot_predictionintervals(
    y_train,
    y_train_pred,
    y_train_pis[:, 0, alpha_plot],
    y_train_pis[:, 1, alpha_plot],
    y_test,
    y_test_pred,
    y_test_pis[:, 0, alpha_plot],
    y_test_pis[:, 1, alpha_plot],
    "Prediction intervals for alpha=0.1",
)
```

## 5. Comparison of the uncertainty quantification methods


In the last section, we compare the calibration of several uncertainty-quantification methods provided by MAPIE using Random Forest as base model. To this aim, we build so-called "calibration plots" which compare the effective marginal coverage obtained on the test set with the target $1-\alpha$ coverage.

```python
Params = TypedDict("Params", {"method": str, "cv": Union[int, Subsample]})
STRATEGIES = {
    "naive": Params(method="naive"),
    "cv": Params(method="base", cv=5),
    "cv_plus": Params(method="plus", cv=5),
    "cv_minmax": Params(method="minmax", cv=5),
    "jackknife_plus_ab": Params(method="plus", cv=Subsample(n_resamplings=20)),
}
mdl = get_regressor("random_forest")
```

```python
y_pred, y_pis, scores = {}, {}, {}
for strategy, params in STRATEGIES.items():
    mapie = MapieRegressor(mdl, **params)
    mapie.fit(X_train, y_train)
    y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=alpha)
    scores[strategy] = [
        regression_coverage_score(y_test, y_pis[strategy][:, 0, i], y_pis[strategy][:, 1, i])
        for i, _ in enumerate(alpha)
    ]
```

```python
plt.figure(figsize=(7, 6))
plt.xlabel("Target coverage (1 - alpha)")
plt.ylabel("Effective coverage")
for strategy, params in STRATEGIES.items():
    plt.plot(1 - alpha, scores[strategy], label=strategy)
plt.plot([0, 1], [0, 1], ls="--", color="k")
plt.legend(loc=[1, 0])
```

The calibration plot clearly demonstrates that the "naive" method underestimates the coverage by giving too narrow prediction intervals, due to the fact that they are built from training data. All other methods show much more robust calibration plots : the effective coverages follow almost linearly the expected coverage levels.
