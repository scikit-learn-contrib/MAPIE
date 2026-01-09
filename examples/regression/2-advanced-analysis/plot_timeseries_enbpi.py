"""
==================================================================
Time series: example of the EnbPI technique
==================================================================

Note: in this example, we use the following terms employed in the scientific literature:

- `alpha` is equivalent to `1 - confidence_level`. It can be seen as a *risk level*
- *calibrate* and *calibration* are equivalent to *conformalize* and *conformalization*.

—

This example uses
:class:`~mapie.time_series_regression.TimeSeriesRegressor` to estimate
prediction intervals associated with time series forecast. It follows [6].

We use here the Victoria electricity demand dataset used in the book
"Forecasting: Principles and Practice" by R. J. Hyndman and G. Athanasopoulos.
The electricity demand features daily and weekly seasonalities and is impacted
by the temperature, considered here as a exogeneous variable.

A Random Forest model is already fitted on data. The hyper-parameters are
optimized with a :class:`~sklearn.model_selection.RandomizedSearchCV` using a
sequential :class:`~sklearn.model_selection.TimeSeriesSplit` cross validation,
in which the training set is prior to the validation set.
The best model is then feeded into
:class:`~mapie.time_series_regression.TimeSeriesRegressor` to estimate the
associated prediction intervals. We compare two approaches: with or without calling
``update`` at every step, following [6]. The results show coverage closer
to the target, along with narrower PIs.
"""

import warnings
from typing import cast

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from numpy.typing import NDArray
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

warnings.simplefilter("ignore")


# Load input data and feature engineering
url_file = (
    "https://raw.githubusercontent.com/scikit-learn-contrib/MAPIE/"
    + "master/examples/data/demand_temperature.csv"
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
num_val_steps = 24 * 7
num_test_steps = 24 * 7
demand_train = demand_df.iloc[: -num_val_steps - num_test_steps, :].copy()
demand_val = demand_df.iloc[-num_val_steps - num_test_steps : -num_test_steps, :].copy()
demand_test = demand_df.iloc[-num_test_steps:, :].copy()
features = ["Weekofyear", "Weekday", "Hour", "Temperature"] + [
    f"Lag_{hour}" for hour in range(1, n_lags)
]

X_train = demand_train.loc[~np.any(demand_train[features].isnull(), axis=1), features]
y_train = demand_train.loc[X_train.index, "Demand"]
X_val = demand_val.loc[:, features]
y_val = demand_val["Demand"]
X_test = demand_test.loc[:, features]
y_test = demand_test["Demand"]

perform_hyperparameters_search = False
if perform_hyperparameters_search:
    # CV parameter search
    X_param_search = pd.concat([X_train, X_val], axis=0)
    y_param_search = pd.concat([y_train, y_val], axis=0)
    test_fold = np.concatenate([-1 * np.ones(len(X_train)), 0 * np.ones(len(X_val))])
    ps = PredefinedSplit(test_fold)
    n_iter = 100
    random_state = 59
    rf_model = RandomForestRegressor(random_state=random_state)
    rf_params = {"max_depth": randint(2, 30), "n_estimators": randint(10, 100)}
    cv_obj = RandomizedSearchCV(
        rf_model,
        param_distributions=rf_params,
        n_iter=n_iter,
        cv=ps,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        verbose=0,
        n_jobs=-1,
    )
    cv_obj.fit(X_param_search, y_param_search)
    model = cv_obj.best_estimator_
else:
    # Model: Random Forest previously optimized with a cross-validation
    model = RandomForestRegressor(max_depth=25, n_estimators=31, random_state=59)

# Estimate prediction intervals on test set with best estimator
alpha = 0.05
cv_mapietimeseries = BlockBootstrap(
    n_resamplings=10, n_blocks=10, overlapping=False, random_state=59
)

mapie_enpbi = TimeSeriesRegressor(
    model,
    method="enbpi",
    cv=cv_mapietimeseries,
    agg_function="mean",
    n_jobs=-1,
)

print("EnbPI, with no update, width optimization")
mapie_enpbi = mapie_enpbi.fit(X_train, y_train)
y_pred_n_update_enbpi, y_pis_n_update_enbpi = mapie_enpbi.predict(
    X_test, confidence_level=1 - alpha, ensemble=True, optimize_beta=True
)
coverage_n_update_enbpi = regression_coverage_score(y_test, y_pis_n_update_enbpi)[0]

width_n_update_enbpi = regression_mean_width_score(y_pis_n_update_enbpi)[0]

print("EnbPI with update, width optimization")
mapie_enpbi = mapie_enpbi.fit(X_train, y_train)
y_pred_update_enbpi = np.zeros(y_pred_n_update_enbpi.shape)
y_pis_update_enbpi = np.zeros(y_pis_n_update_enbpi.shape)

step_size = 1
(
    y_pred_update_enbpi[:step_size],
    y_pis_update_enbpi[:step_size, :, :],
) = mapie_enpbi.predict(
    X_test.iloc[:step_size, :],
    confidence_level=1 - alpha,
    ensemble=True,
    optimize_beta=True,
)

for step in range(step_size, len(X_test), step_size):
    mapie_enpbi.update(
        X_test.iloc[(step - step_size) : step, :],
        y_test.iloc[(step - step_size) : step],
    )
    (
        y_pred_update_enbpi[step : step + step_size],
        y_pis_update_enbpi[step : step + step_size, :, :],
    ) = mapie_enpbi.predict(
        X_test.iloc[step : (step + step_size), :],
        confidence_level=1 - alpha,
        ensemble=True,
    )
coverage_update_enbpi = regression_coverage_score(y_test, y_pis_update_enbpi)[0]
width_update_enbpi = regression_mean_width_score(y_pis_update_enbpi)[0]

# Print results
print(
    "Coverage / prediction interval width mean for TimeSeriesRegressor: "
    "\nEnbPI without any update:"
    f"{coverage_n_update_enbpi:.3f}, {width_n_update_enbpi:.3f}"
)
print(
    "Coverage / prediction interval width mean for TimeSeriesRegressor: "
    "\nEnbPI with update:"
    f"{coverage_update_enbpi:.3f}, {width_update_enbpi:.3f}"
)

enbpi_no_update = {
    "y_pred": y_pred_n_update_enbpi,
    "y_pis": y_pis_n_update_enbpi,
    "coverage": coverage_n_update_enbpi,
    "width": width_n_update_enbpi,
}

enbpi_update = {
    "y_pred": y_pred_update_enbpi,
    "y_pis": y_pis_update_enbpi,
    "coverage": coverage_update_enbpi,
    "width": width_update_enbpi,
}

results = [enbpi_no_update, enbpi_update]

# Plot estimated prediction intervals on test set
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 12), sharex="col")

for i, (ax, w, result) in enumerate(
    zip(axs, ["EnbPI, without update", "EnbPI with update"], results)
):
    ax.set_ylabel("Hourly demand (GW)", fontsize=20)
    ax.plot(demand_test.Demand, lw=2, label="Test data", c="C1")

    ax.plot(
        demand_test.index,
        result["y_pred"],
        lw=2,
        c="C2",
        label="Predictions",
    )

    y_pis = cast(NDArray, result["y_pis"])

    ax.fill_between(
        demand_test.index,
        y_pis[:, 0, 0],
        y_pis[:, 1, 0],
        color="C2",
        alpha=0.2,
        label="TimeSeriesRegressor PIs",
    )

    ax.set_title(
        w + f"\nCoverage:{result['coverage']:.3f}  Width:{result['width']:.3f}",
        fontweight="bold",
        size=20,
    )
    plt.xticks(size=15, rotation=45)
    plt.yticks(size=15)

axs[0].legend(prop={"size": 22})
plt.show()
