"""
=======================================================
Estimating prediction intervals of time series forecast
=======================================================


"""
import pandas as pd
from scipy.stats import randint
from matplotlib import pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
from mapie.estimators import MapieRegressor
from mapie.metrics import coverage_score

# load input data and feature engineering
demand_df = pd.read_csv(
    "data/demand_temperature.csv",
    parse_dates=True,
    index_col=0
)
demand_df["Date"] = pd.to_datetime(demand_df.index)
demand_df["Weekofyear"] = demand_df.Date.dt.isocalendar().week.astype('int64')
demand_df["Weekday"] = demand_df.Date.dt.isocalendar().day.astype('int64')
demand_df["Hour"] = demand_df.index.hour

# train/test split
num_forecast_steps = 24 * 7 * 2
demand_train = demand_df.iloc[:-num_forecast_steps, :].copy()
demand_test = demand_df.iloc[-num_forecast_steps:, :].copy()
X_train = demand_train[["Weekofyear", "Weekday", "Hour", "Temperature"]]
y_train = demand_train["Demand"]
X_test = demand_test[["Weekofyear", "Weekday", "Hour", "Temperature"]]
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
    scoring=mean_absolute_percentage_error,
    random_state=random_state,
    verbose=0,
    n_jobs=-1,
)
cv_obj.fit(X_train, y_train)
best_est = cv_obj.best_estimator_

# prediction intervals
alpha = 0.1
mapie = MapieRegressor(best_est, alpha=alpha, method="plus", cv=n_splits)
mapie.fit(X_train, y_train)
y_pred, y_pis = mapie.predict(X_test)

coverage = coverage_score(y_test, y_pis[:, 0, 0], y_pis[:, 1, 0])
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel("Hourly demand (GW)")
ax.plot(demand_test.Demand, lw=2, label="Test data", c="C1")
ax.plot(demand_test.index, y_pred, lw=2, label="Predictions", c="C2", ls='-')
ax.fill_between(
    demand_test.index, y_pis[:, 0, 0], y_pis[:, 1, 0], color="C2", alpha=0.2
)
plt.title(
    f"Target and effective coverages for "
    f"alpha={alpha:.2f}: ({1-alpha:.3f}, {coverage:.3f})"
)
ax.legend()
