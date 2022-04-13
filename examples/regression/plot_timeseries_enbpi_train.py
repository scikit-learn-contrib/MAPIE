"""
==================================================================
Estimating prediction intervals of time series forecast with EnbPI
==================================================================
This example uses
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor` to estimate
prediction intervals associated with time series forecast. It follows [6] and
an alternative expermimental implemetation inspired from [2]

We use here the Victoria electricity demand dataset used in the book
"Forecasting: Principles and Practice" by R. J. Hyndman and G. Athanasopoulos.
The electricity demand features daily and weekly seasonalities and is impacted
by the temperature, considered here as a exogeneous variable.

A Random Forest model is aloready fitted on data. The hyper-parameters are
optimized with a :class:`sklearn.model_selection.RandomizedSearchCV` using a
sequential :class:`sklearn.model_selection.TimeSeriesSplit` cross validation,
in which the training set is prior to the validation set.
The best model is then feeded into
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor` to estimate the
associated prediction intervals. We compare two approaches: one with no
`partial_fit` call and one with `partial_fit` every step.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

warnings.simplefilter("ignore")

# Load input data and feature engineering
demand_df = pd.read_csv(
    "../data/demand_temperature.csv", parse_dates=True, index_col=0
)

demand_df["Date"] = pd.to_datetime(demand_df.index)
demand_df["Weekofyear"] = demand_df.Date.dt.isocalendar().week.astype("int64")
demand_df["Weekday"] = demand_df.Date.dt.isocalendar().day.astype("int64")
demand_df["Hour"] = demand_df.index.hour

n_lags = 5
for hour in range(1, n_lags):
    demand_df[f"Lag_{hour}"] = demand_df["Demand"].shift(hour)

# Train/validation/test split
num_test_steps = 24 * 7
demand_train = demand_df.iloc[:-num_test_steps, :].copy()
demand_test = demand_df.iloc[-num_test_steps:, :].copy()
features = ["Weekofyear", "Weekday", "Hour", "Temperature"] + [
    f"Lag_{hour}" for hour in range(1, n_lags)
]

X_train = demand_train.loc[
    ~np.any(demand_train[features].isnull(), axis=1), features
]
y_train = demand_train.loc[X_train.index, "Demand"]
X_test = demand_test.loc[:, features]
y_test = demand_test["Demand"]

# CV parameter search
n_iter = 100
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
random_state = 59
rf_model = RandomForestRegressor(random_state=random_state)
rf_params = {"max_depth": randint(2, 30), "n_estimators": randint(10, 100)}
cv_obj = RandomizedSearchCV(
    rf_model,
    param_distributions=rf_params,
    n_iter=n_iter,
    cv=tscv,
    scoring="neg_root_mean_squared_error",
    random_state=random_state,
    verbose=0,
    n_jobs=-1,
)
cv_obj.fit(X_train, y_train)
print(cv_obj.best_estimator_)
