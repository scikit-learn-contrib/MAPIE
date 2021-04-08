"""
======================================================
Plotting MAPIE prediction intervals with a toy dataset
======================================================

An example plot of :class:`mapie.MapieRegressor` used
in the Quickstart.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from mapie import MapieRegressor

regressor = LinearRegression()
X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)

mapie = MapieRegressor(regressor)
mapie.fit(X, y)
X_pi = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_preds = mapie.predict(X_pi)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X, y, alpha=0.3)
plt.plot(X_pi, y_preds[:, 0], color='C1')
plt.fill_between(X_pi.ravel(), y_preds[:, 1], y_preds[:, 2], alpha=0.3)
plt.show()
