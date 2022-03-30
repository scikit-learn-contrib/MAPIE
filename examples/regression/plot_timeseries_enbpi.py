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
import copy
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
model = RandomForestRegressor(max_depth=15, n_estimators=1, random_state=59)

# Estimate prediction intervals on test set with best estimator
alpha = 0.1
cv_MapieTimeSeries = BlockBootstrap(20, length=48, random_state=59)

mapie_model = MapieTimeSeriesRegressor(
    model, method="plus", cv=cv_MapieTimeSeries, agg_function="mean", n_jobs=-1
)
mapie_model = mapie_model.fit(X_train, y_train)
mapie_no_pfit = mapie_model.fit(X_train, y_train)
mapie_pfit_JAB_F = mapie_model.fit(X_train, y_train)
mapie_pfit_JAB_T = mapie_model.fit(X_train, y_train)

gap_pfit = 1

# With no partial_fit, JAB_like is False
y_pred_npfit_JAB_F, y_pis_npfit_JAB_F = mapie_no_pfit.predict(
    X_test, alpha=alpha, ensemble=True
)
coverage_npfit_JAB_F = regression_coverage_score(
    y_test, y_pis_npfit_JAB_F[:, 0, 0], y_pis_npfit_JAB_F[:, 1, 0]
)
width_npfit_JAB_F = (
    y_pis_npfit_JAB_F[:, 1, 0] - y_pis_npfit_JAB_F[:, 0, 0]
).mean()

# With partial_fit every hour, JAB_like is False

y_pred_pfit_JAB_F, y_pis_pfit_JAB_F = mapie_pfit_JAB_F.predict(
    X_test.iloc[:gap_pfit, :], alpha=alpha, ensemble=True
)

for step in range(gap_pfit, len(X_test), gap_pfit):
    mapie_pfit_JAB_F.partial_fit(
        X_test.iloc[(step - gap_pfit) : step, :],
        y_test.iloc[(step - gap_pfit) : step],
    )
    y_pred_gap_step, y_pis_gap_step = mapie_pfit_JAB_F.predict(
        X_test.iloc[step : (step + gap_pfit), :], alpha=alpha, ensemble=True
    )
    y_pred_pfit_JAB_F = np.concatenate(
        (y_pred_pfit_JAB_F, y_pred_gap_step), axis=0
    )
    y_pis_pfit_JAB_F = np.concatenate(
        (y_pis_pfit_JAB_F, y_pis_gap_step), axis=0
    )

coverage_pfit_JAB_F = regression_coverage_score(
    y_test, y_pis_pfit_JAB_F[:, 0, 0], y_pis_pfit_JAB_F[:, 1, 0]
)
width_pfit_JAB_F = (
    y_pis_pfit_JAB_F[:, 1, 0] - y_pis_pfit_JAB_F[:, 0, 0]
).mean()


# With no partial_fit, JAB_like is True
y_pred_npfit_JAB_T, y_pis_npfit_JAB_T = mapie_no_pfit.predict(
    X_test, alpha=alpha, JAB_Like=True
)
coverage_npfit_JAB_T = regression_coverage_score(
    y_test, y_pis_npfit_JAB_T[:, 0, 0], y_pis_npfit_JAB_T[:, 1, 0]
)
width_npfit_JAB_T = (
    y_pis_npfit_JAB_T[:, 1, 0] - y_pis_npfit_JAB_T[:, 0, 0]
).mean()

# With partial_fit every hour, JAB_like is True
y_pred_pfit_JAB_T, y_pis_pfit_JAB_T = mapie_no_pfit.predict(
    X_test.iloc[:gap_pfit, :], alpha=alpha, JAB_Like=True
)
for step in range(gap_pfit, len(X_test), gap_pfit):
    mapie_pfit_JAB_T.partial_fit(
        X_test.iloc[(step - gap_pfit) : step, :],
        y_test.iloc[(step - gap_pfit) : step],
    )
    y_pred_gap_step, y_pis_gap_step = mapie_pfit_JAB_T.predict(
        X_test.iloc[step : (step + gap_pfit), :],
        alpha=alpha,
        JAB_Like=True,
        ensemble=True,
    )
    y_pred_pfit_JAB_T = np.concatenate(
        (y_pred_pfit_JAB_T, y_pred_gap_step), axis=0
    )
    y_pis_pfit_JAB_T = np.concatenate(
        (y_pis_pfit_JAB_T, y_pis_gap_step), axis=0
    )

coverage_pfit_JAB_T = regression_coverage_score(
    y_test, y_pis_pfit_JAB_T[:, 0, 0], y_pis_pfit_JAB_T[:, 1, 0]
)
width_pfit_JAB_T = (
    y_pis_pfit_JAB_T[:, 1, 0] - y_pis_pfit_JAB_T[:, 0, 0]
).mean()

# Print results
print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor: "
    "\nWithout any partial_fit. JAB_like is False:"
    f"{coverage_npfit_JAB_F:.3f}, {width_npfit_JAB_F:.3f}"
)
print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor: "
    "\nWithout any partial_fit. JAB_like is True:"
    f"{coverage_npfit_JAB_T:.3f}, {width_npfit_JAB_T:.3f}"
)
print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor: "
    "\nWith partial_fit. JAB_like is False:"
    f"{coverage_pfit_JAB_F:.3f}, {width_pfit_JAB_F:.3f}"
)
print(
    "Coverage / prediction interval width mean for MapieTimeSeriesRegressor: "
    "\nWith partial_fit. JAB_like is True:"
    f"{coverage_pfit_JAB_T:.3f}, {width_pfit_JAB_T:.3f}"
)

# Plot estimated prediction intervals on test set
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows=2, ncols=2, figsize=(30, 10), sharey="row", sharex="col"
)

ax1.set_ylabel("Hourly demand (GW)")
ax1.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax1.plot(
    demand_test.index, y_pred_npfit_JAB_F, lw=2, c="C2", label="Predictions"
)
ax1.fill_between(
    demand_test.index,
    y_pis_npfit_JAB_F[:, 0, 0],
    y_pis_npfit_JAB_F[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax1.legend()
ax1.set_title(
    "Without partial_fit, JAB False."
    f"Coverage:{coverage_npfit_JAB_F:.3f}  Width:{width_npfit_JAB_F:.3f}"
)

ax2.set_ylabel("Hourly demand (GW)")
ax2.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax2.plot(
    demand_test.index, y_pred_npfit_JAB_T, lw=2, c="C2", label="Predictions"
)
ax2.fill_between(
    demand_test.index,
    y_pis_npfit_JAB_T[:, 0, 0],
    y_pis_npfit_JAB_T[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax2.legend()
ax2.set_title(
    "Without partial_fit, JAB True."
    f"Coverage:{coverage_npfit_JAB_T:.3f}  Width:{width_npfit_JAB_T:.3f}"
)

ax3.set_ylabel("Hourly demand (GW)")
ax3.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax3.plot(
    demand_test.index, y_pred_npfit_JAB_F, lw=2, c="C2", label="Predictions"
)
ax3.fill_between(
    demand_test.index,
    y_pis_npfit_JAB_F[:, 0, 0],
    y_pis_npfit_JAB_F[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax3.legend()
ax3.set_title(
    "With partial_fit, JAB False."
    f"Coverage:{coverage_npfit_JAB_F:.3f}  Width:{width_npfit_JAB_F:.3f}"
)

ax4.set_ylabel("Hourly demand (GW)")
ax4.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax4.plot(
    demand_test.index, y_pred_pfit_JAB_T, lw=2, c="C2", label="Predictions"
)
ax4.fill_between(
    demand_test.index,
    y_pis_pfit_JAB_T[:, 0, 0],
    y_pis_pfit_JAB_T[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="MapieTimeSeriesRegressor PIs",
)
ax4.legend()
ax4.set_title(
    "With partial_fit, JAB True."
    f"Coverage:{coverage_npfit_JAB_T:.3f}  Width:{width_npfit_JAB_T:.3f}"
)
plt.show()
