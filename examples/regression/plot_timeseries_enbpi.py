"""
==================================================================
Estimating prediction intervals of time series forecast with EnbPI
==================================================================
This example uses
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor` to estimate
prediction intervals associated with time series forecast. The implementation
is still at its first step, based on Jackknife+-after-bootsrtap, to estimate
residuals and associated prediction intervals.

We use here the Victoria electricity demand dataset used in the book
"Forecasting: Principles and Practice" by R. J. Hyndman and G. Athanasopoulos.
The electricity demand features daily and weekly seasonalities and is impacted
by the temperature, considered here as a exogeneous variable.

The data is modelled by a Random Forest model with a
:class:`sklearn.model_selection.RandomizedSearchCV` using a sequential
:class:`sklearn.model_selection.TimeSeriesSplit` cross validation, in which the
training set is prior to the validation set.
The best model is then feeded into
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor` to estimate the
associated prediction intervals. We compare two approaches: one with no
`partial_fit` call and one with `partial_fit` every 5 steps.
"""
import warnings

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from sklearn.ensemble import RandomForestRegressor

from mapie.metrics import regression_coverage_score
from mapie.subsample import Subsample
from mapie.time_series_regression import MapieTimeSeriesRegressor

warnings.simplefilter("ignore")

# Load input data and feature engineering
demand_df = pd.read_csv(
    "../data/demand_temperature.csv", parse_dates=True, index_col=0
)
demand_df["Date"] = pd.to_datetime(demand_df.index)
demand_df["Weekofyear"] = demand_df.Date.dt.isocalendar().week.astype("int64")
demand_df["Weekday"] = demand_df.Date.dt.isocalendar().day.astype("int64")
demand_df["Hour"] = demand_df.index.hour
for hour in range(1, 5):
    demand_df[f"Lag_{hour}"] = demand_df["Demand"].shift(hour)

# Train/validation/test split
num_test_steps = 24 * 7 * 2
demand_train = demand_df.iloc[:-num_test_steps, :].copy()
demand_test = demand_df.iloc[-num_test_steps:, :].copy()
features = ["Weekofyear", "Weekday", "Hour", "Temperature"] + [
    f"Lag_{hour}" for hour in range(1, 5)
]
X_train = demand_train.loc[
    ~np.any(demand_train[features].isnull(), axis=1), features
]
y_train = demand_train.loc[X_train.index, "Demand"]
X_test = demand_test.loc[:, features]
y_test = demand_test["Demand"]

# Model
model = RandomForestRegressor(max_depth=15, n_estimators=673, random_state=59)

# Estimate prediction intervals on test set with best estimator
alpha = 0.1
cv_Mapie = Subsample(30, random_state=59)
mapie = MapieTimeSeriesRegressor(
    model, method="plus", cv=cv_Mapie, agg_function="median", n_jobs=-1
)
mapie.fit(X_train, y_train)

# With no partial_fit
y_pred, y_pis = mapie.predict(X_test, alpha=alpha)
coverage = regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0])
width = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()

# With partial_fit every five hours
y_pred_5_steps, y_pis_5_steps = mapie.predict(X_test.iloc[:5, :], alpha=alpha)

for step in range(5, len(X_test), 5):
    mapie.partial_fit(
        X_test.iloc[(step - 5): step, :], y_test.iloc[(step - 5):step]
    )
    y_pred_step, y_pis_step = mapie.predict(
        X_test.iloc[step: (step + 5), :], alpha=alpha
    )
    y_pred_5_steps = np.concatenate((y_pred_5_steps, y_pred_step), axis=0)
    y_pis_5_steps = np.concatenate((y_pis_5_steps, y_pis_step), axis=0)

coverage_5_step = regression_coverage_score(
    y_test, y_pis_5_steps[:, 0, 0], y_pis_5_steps[:, 1, 0]
)
width_5_step = (y_pis_5_steps[:, 1, 0] - y_pis_5_steps[:, 0, 0]).mean()


# Print results
print(
    "Coverage and prediction interval width mean for MapieTimeSeriesRegressor:"
    "\nWithout any partial_fit:"
    f"{coverage:.3f}, {width:.3f}"
)

# Plot estimated prediction intervals on test set
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel("Hourly demand (GW)")
ax.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax.plot(demand_test.index, y_pred, lw=2, c="C2", label="Predictions")
ax.fill_between(
    demand_test.index,
    y_pis[:, 0, 0],
    y_pis[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax.legend()
plt.show()

print(
    "Coverage and prediction interval width mean for MapieTimeSeriesRegressor:"
    "\nWith partial_fit every 5 steps:"
    f"{coverage_5_step:.3f}, {width_5_step:.3f}"
)

# Plot estimated prediction intervals on test set
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel("Hourly demand (GW)")
ax.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax.plot(demand_test.index, y_pred_5_steps, lw=2, c="C2", label="Predictions")
ax.fill_between(
    demand_test.index,
    y_pis_5_steps[:, 0, 0],
    y_pis_5_steps[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax.legend()
plt.show()
