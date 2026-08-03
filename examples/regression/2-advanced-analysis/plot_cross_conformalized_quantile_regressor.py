"""
Cross conformalized quantile regression on synthetic data
=========================================================



This example illustrates how to use `CrossConformalizedQuantileRegressor`
to estimate prediction intervals on a synthetic regression task.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from mapie.metrics.regression import (
    regression_coverage_score,
    regression_mean_width_score,
)
from mapie.regression import CrossConformalizedQuantileRegressor

RANDOM_STATE = 1
CONFIDENCE_LEVEL = 0.8

##############################################################################
# Generate synthetic data and split into training and testing sets.

X, y = make_regression(
    n_samples=1000,
    n_features=1,
    noise=20,
    random_state=RANDOM_STATE,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

##############################################################################
# Fit and conformalize the cross-conformalized quantile regressor.

gb_reg = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.5,
    random_state=RANDOM_STATE,
)

mapie_cross_cqr = CrossConformalizedQuantileRegressor(
    estimator=gb_reg,
    confidence_level=CONFIDENCE_LEVEL,
    cv=5,
    method="plus",
)
mapie_cross_cqr.fit_conformalize(X_train, y_train)

y_pred, y_pis = mapie_cross_cqr.predict_interval(X_test)

coverage = regression_coverage_score(y_test, y_pis)[0]
width = regression_mean_width_score(y_pis)[0]

print(f"Coverage: {coverage:.3f}")
print(f"Mean width: {width:.3f}")

##############################################################################
# Plot predictions and prediction intervals.

order = np.argsort(X_test[:, 0])
X_plot = X_test[order, 0]
y_pred_plot = y_pred[order]

y_low = y_pis[order, 0, 0]
y_up = y_pis[order, 1, 0]

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], y_test, s=8, alpha=0.3, label="Test data")
plt.plot(X_plot, y_pred_plot, color="C1", label="Predictions")
plt.fill_between(
    X_plot,
    y_low,
    y_up,
    color="C1",
    alpha=0.2,
    label="Prediction intervals",
)
plt.title(
    "CrossConformalizedQuantileRegressor\n"
    f"confidence_level={CONFIDENCE_LEVEL}, coverage={coverage:.3f}"
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
