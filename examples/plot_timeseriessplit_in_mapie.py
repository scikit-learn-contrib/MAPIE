"""
=======================================================
Estimating prediction intervals of time series forecast
=======================================================
This example uses MAPIE to estimate prediction intervals associated with
time series forecast. We compare the standard cross-validation approach and
a sequential cross-validation, used with time-series data in which the training
set is prior to the validation set, to estimate residuals and associated
prediction intervals.
We use here the Victoria electricity demand dataset used in the book
"Forecasting: Principles and Practice" by R. J. Hyndman and G. Athanasopoulos.
The electricity demand features daily and weekly seasonalities and is impacted
by the temperature, considered here as a exogeneous variable.
The data is modelled by a Random Forest model with a
:class:`sklearn.model_selection.RandomizedSearchCV`
using a `sklearn.model_selection.TimeSeriesSplit` cross validation.
The best model is then feeded into :class:`mapie.estimators.MapieRegressor`
to estimate the associated
prediction intervals. We consider four strategies, with the CV and CV+
resampling method and using either a standard `sklearn.model_selection.KFold`
or a sequential
`sklearn.model_selection.TimeSeriesSplit` method for estimating the residuals.
It is found that the sequential cross-validation induces larger prediction
intervals since the perturbed models are trained on smaller training sets
than with the standard cross-validation, hence inducing larger differences
of their predictions.
"""
import pandas as pd
from scipy.stats import randint
from matplotlib import pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from mapie.estimators import MapieRegressor
from mapie.metrics import coverage_score

# Load input data and feature engineering
demand_df = pd.read_csv(
    "data/demand_temperature.csv",
    parse_dates=True,
    index_col=0
)
demand_df["Date"] = pd.to_datetime(demand_df.index)
demand_df["Weekofyear"] = demand_df.Date.dt.isocalendar().week.astype('int64')
demand_df["Weekday"] = demand_df.Date.dt.isocalendar().day.astype('int64')
demand_df["Hour"] = demand_df.index.hour

# Train/test split
num_forecast_steps = 24 * 7 * 2
demand_train = demand_df.iloc[:-num_forecast_steps, :].copy()
demand_test = demand_df.iloc[-num_forecast_steps:, :].copy()
X_train = demand_train.loc[:, ["Weekofyear", "Weekday", "Hour", "Temperature"]]
y_train = demand_train["Demand"]
X_test = demand_test.loc[:, ["Weekofyear", "Weekday", "Hour", "Temperature"]]
y_test = demand_test["Demand"]

# CV parameter search
n_iter = 10
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
random_state = 59
rf_model = RandomForestRegressor(random_state=random_state)
rf_params = {
    "max_depth": randint(2, 30),
    "n_estimators": randint(10, 1e3)
}
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
best_est = cv_obj.best_estimator_

# Estimate prediction intervals on test set with best estimator
alpha = 0.1
strategies = {
    "cv_standard": dict(method="base", cv=n_splits),
    "cv_sequential": dict(method="base", cv=tscv),
    "cv_plus_standard": dict(method="plus", cv=5),
    "cv_plus_sequential": dict(method="plus", cv=tscv)
}
y_pred, y_pis, coverages, widths = {}, {}, {}, {}
for strategy, params in strategies.items():
    mapie = MapieRegressor(best_est, **params)
    mapie.fit(X_train, y_train)
    y_pred_, y_pis_ = mapie.predict(X_test, alpha=alpha)
    y_pred[strategy] = y_pred_
    y_pis[strategy] = y_pis_
    coverages[strategy] = coverage_score(
        y_test, y_pis_[:, 0, 0], y_pis_[:, 1, 0]
    )
    widths[strategy] = (y_pis_[:, 1, 0] - y_pis_[:, 0, 0]).mean()

# Print results
for strategy in strategies:
    print(
        "Coverage and prediction interval width mean for "
        f"{strategy:18} strategy: "
        f"{coverages[strategy]:.3f}, {widths[strategy]:.3f}"
    )

# Plot estimated prediction intervals on test set
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel("Hourly demand (GW)")
ax.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax.plot(
    demand_test.index,
    best_est.predict(X_test),
    lw=2,
    c="C2",
    label="Predictions"
)
ax.fill_between(
    demand_test.index,
    y_pis["cv_plus_standard"][:, 0, 0],
    y_pis["cv_plus_standard"][:, 1, 0],
    color="C2",
    alpha=0.2,
    label="CV+(standard)"
)
ax.plot(
    demand_test.index,
    y_pis["cv_plus_sequential"][:, 0, 0],
    color="C2",
    ls="--",
    label="CV+(sequential)"
)
ax.plot(
    demand_test.index,
    y_pis["cv_plus_sequential"][:, 1, 0],
    color="C2",
    ls="--"
)
ax.legend()
plt.show()
