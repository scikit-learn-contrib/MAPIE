"""
======================================================
Plotting MAPIE prediction intervals with a toy dataset
======================================================

An example plot of :class:`mapie.estimators.MapieRegressor` used
in the Quickstart.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from mapie.estimators import MapieRegressor
from mapie.metrics import coverage_score

regressor = LinearRegression()
X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)

mapie = MapieRegressor(regressor, method="jackknife_plus")
mapie.fit(X, y)
y_preds = mapie.predict(X)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X, y, alpha=0.3)
plt.plot(X, y_preds[:, 0], color='C1')
order = np.argsort(X[:, 0])
plt.fill_between(X[order].ravel(), y_preds[:, 1][order], y_preds[:, 2][order], alpha=0.3)
plt.title(f"Target coverage = 0.9; Effective coverage = {coverage_score(y, y_preds[:, 1], y_preds[:, 2])}")
plt.show()
