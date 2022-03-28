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

A Random Forest model is fitted on data. The hyper-parameters are optimized
with a :class:`sklearn.model_selection.RandomizedSearchCV` using a sequential
:class:`sklearn.model_selection.TimeSeriesSplit` cross validation, in which the
training set is prior to the validation set.
The best model is then feeded into
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor` to estimate the
associated prediction intervals. We compare two approaches: one with no
`partial_fit` call and one with `partial_fit` every step.
"""
import warnings

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from sklearn.ensemble import RandomForestRegressor

from mapie.metrics import regression_coverage_score
from mapie.subsample import BlockBootstrap
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
for hour in range(1, 3):
    demand_df[f"Lag_{hour}"] = demand_df["Demand"].shift(hour)

# Train/validation/test split
num_test_steps = 24 * 7
demand_train = demand_df.iloc[:-num_test_steps, :].copy()
demand_test = demand_df.iloc[-num_test_steps:, :].copy()
features = ["Weekofyear", "Weekday", "Hour", "Temperature"] + [
    f"Lag_{hour}" for hour in range(1, 2)
]

X_train = demand_train.loc[
    ~np.any(demand_train[features].isnull(), axis=1), features
]
y_train = demand_train.loc[X_train.index, "Demand"]
X_test = demand_test.loc[:, features]
y_test = demand_test["Demand"]

# Model: Random Forest previously optimized with a cross-validation
model = RandomForestRegressor(max_depth=17, n_estimators=150, random_state=59)

# Estimate prediction intervals on test set with best estimator
alpha = 0.1
cv_MapieTimeSeries = BlockBootstrap(50, length=48, random_state=59)
mapie = MapieTimeSeriesRegressor(
    model, method="plus", cv=cv_MapieTimeSeries, agg_function="mean", n_jobs=-1
)
mapie.fit(X_train, y_train)

# With no partial_fit
y_pred, y_pis = mapie.predict(X_test, alpha=alpha)
coverage = regression_coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0])
width = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()

# With partial_fit every hour
gap = 1

y_pred_steps, y_pis_steps = mapie.predict(X_test.iloc[:gap, :], alpha=alpha)

for step in range(gap, len(X_test), gap):
    mapie.partial_fit(
        X_test.iloc[(step - gap):step, :], y_test.iloc[(step - gap):step]
    )
    y_pred_gap_step, y_pis_gap_step = mapie.predict(
        X_test.iloc[step:(step + gap), :],
        alpha=alpha,
    )
    y_pred_steps = np.concatenate((y_pred_steps, y_pred_gap_step), axis=0)
    y_pis_steps = np.concatenate((y_pis_steps, y_pis_gap_step), axis=0)

coverage_steps = regression_coverage_score(
    y_test, y_pis_steps[:, 0, 0], y_pis_steps[:, 1, 0]
)
width_steps = (y_pis_steps[:, 1, 0] - y_pis_steps[:, 0, 0]).mean()

# Print results
print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor: "
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
plt.title("Without partial_fit")
plt.show()

print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor "
    "\nWith partial_fit every step: "
    f"{coverage_steps:.3f}, {width_steps:.3f}"
)

# Plot estimated prediction intervals on test set
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel("Hourly demand (GW)")
ax.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax.plot(demand_test.index, y_pred_steps, lw=2, c="C2", label="Predictions")
ax.fill_between(
    demand_test.index,
    y_pis_steps[:, 0, 0],
    y_pis_steps[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax.legend()
plt.title("With partial_fit")
plt.show()
