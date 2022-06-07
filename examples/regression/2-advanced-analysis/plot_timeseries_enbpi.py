"""
==================================================================
Estimating prediction intervals of time series forecast with EnbPI
==================================================================

This example uses
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor` to estimate
prediction intervals associated with time series forecast. It follows [6].

We use here the Victoria electricity demand dataset used in the book
"Forecasting: Principles and Practice" by R. J. Hyndman and G. Athanasopoulos.
The electricity demand features daily and weekly seasonalities and is impacted
by the temperature, considered here as a exogeneous variable.

A Random Forest model is already fitted on data. The hyper-parameters are
optimized with a :class:`sklearn.model_selection.RandomizedSearchCV` using a
sequential :class:`sklearn.model_selection.TimeSeriesSplit` cross validation,
in which the training set is prior to the validation set.
The best model is then feeded into
:class:`mapie.time_series_regression.MapieTimeSeriesRegressor` to estimate the
associated prediction intervals. We compare four approaches: with or without
``partial_fit`` called at every step, and following [6]. It appears that
``partial_fit`` offer higher coverage, but with higher width of PIs and is much
slower.
"""

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from mapie.metrics import (
    regression_coverage_score,
    regression_mean_width_score,
)
from mapie.subsample import BlockBootstrap
from mapie.time_series_regression import MapieTimeSeriesRegressor

# Load input data and feature engineering
url_file = ("https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/" +
            "master/examples/data/demand_temperature.csv"
            )
demand_df = pd.read_csv(url_file, parse_dates=True, index_col=0)

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

perform_hyperparameters_search = False
if perform_hyperparameters_search:
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
    model = cv_obj.best_estimator_
else:
    # Model: Random Forest previously optimized with a cross-validation
    model = RandomForestRegressor(
        max_depth=10, n_estimators=50, random_state=59
    )

# Estimate prediction intervals on test set with best estimator
alpha = 0.05
cv_mapietimeseries = BlockBootstrap(
    n_resamplings=100, length=48, overlapping=True, random_state=59
)

mapie_enpbi = MapieTimeSeriesRegressor(
    model,
    method="enbpi",
    cv=cv_mapietimeseries,
    agg_function="mean",
    n_jobs=-1,
)

print("EnbPI, with no partial_fit, width optimization")
mapie_enpbi = mapie_enpbi.fit(X_train, y_train)
y_pred_npfit_enbpi, y_pis_npfit_enbpi = mapie_enpbi.predict(
    X_test, alpha=alpha, ensemble=True, optimize_beta=True
)
coverage_npfit_enbpi = regression_coverage_score(
    y_test, y_pis_npfit_enbpi[:, 0, 0], y_pis_npfit_enbpi[:, 1, 0]
)

width_npfit_enbpi = regression_mean_width_score(
    y_pis_npfit_enbpi[:, 1, 0], y_pis_npfit_enbpi[:, 0, 0]
)

print("EnbPI with partial_fit, width optimization")
mapie_enpbi = mapie_enpbi.fit(X_train, y_train)
y_pred_pfit_enbpi = np.zeros(y_pred_npfit_enbpi.shape)
y_pis_pfit_enbpi = np.zeros(y_pis_npfit_enbpi.shape)

step_size = 1
(
    y_pred_pfit_enbpi[:step_size],
    y_pis_pfit_enbpi[:step_size, :, :],
) = mapie_enpbi.predict(
    X_test.iloc[:step_size, :], alpha=alpha, ensemble=True, optimize_beta=True
)

for step in range(step_size, len(X_test), step_size):
    mapie_enpbi.partial_fit(
        X_test.iloc[(step - step_size):step, :],
        y_test.iloc[(step - step_size):step],
    )
    (
        y_pred_pfit_enbpi[step:step + step_size],
        y_pis_pfit_enbpi[step:step + step_size, :, :],
    ) = mapie_enpbi.predict(
        X_test.iloc[step:(step + step_size), :],
        alpha=alpha,
        ensemble=True,
        optimize_beta=True,
    )
coverage_pfit_enbpi = regression_coverage_score(
    y_test, y_pis_pfit_enbpi[:, 0, 0], y_pis_pfit_enbpi[:, 1, 0]
)
width_pfit_enbpi = regression_mean_width_score(
    y_pis_pfit_enbpi[:, 1, 0], y_pis_pfit_enbpi[:, 0, 0]
)

# Print results
print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor: "
    "\nEnbPI without any partial_fit:"
    f"{coverage_npfit_enbpi :.3f}, {width_npfit_enbpi:.3f}"
)
print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor: "
    "\nEnbPI with partial_fit:"
    f"{coverage_pfit_enbpi:.3f}, {width_pfit_enbpi:.3f}"
)

# Plot estimated prediction intervals on test set
fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(30, 25), sharey="row", sharex="col"
)

for ax in [ax1, ax2]:
    ax.set_ylabel("Hourly demand (GW)")
    ax.plot(demand_test.Demand, lw=2, label="Test data", c="C1")

ax1.plot(
    demand_test.index, y_pred_npfit_enbpi, lw=2, c="C2", label="Predictions"
)
ax1.fill_between(
    demand_test.index,
    y_pis_npfit_enbpi[:, 0, 0],
    y_pis_npfit_enbpi[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax1.set_title(
    "EnbPI, without partial_fit.\n"
    f"Coverage:{coverage_npfit_enbpi:.3f}  Width:{width_npfit_enbpi:.3f}"
)

ax2.plot(
    demand_test.index, y_pred_pfit_enbpi, lw=2, c="C2", label="Predictions"
)
ax2.fill_between(
    demand_test.index,
    y_pis_pfit_enbpi[:, 0, 0],
    y_pis_pfit_enbpi[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax2.set_title(
    "EnbPI with partial_fit.\n"
    f"Coverage:{coverage_pfit_enbpi:.3f}  Width:{width_pfit_enbpi:.3f}"
)

ax1.legend()
plt.show()
