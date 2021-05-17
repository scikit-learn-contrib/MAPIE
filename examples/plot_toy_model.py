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

alpha = [0.05, 0.32]
mapie = MapieRegressor(regressor, alpha=alpha, method="plus")
mapie.fit(X, y)
y_preds = mapie.predict(X)

plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X, y, alpha=0.3)
plt.plot(X, y_preds[:, 0, 0], color="C1")
order = np.argsort(X[:, 0])
plt.plot(X[order], y_preds[order][:, 1, 1], color="C1", ls="--")
plt.plot(X[order], y_preds[order][:, 2, 1], color="C1", ls="--")
plt.fill_between(X[order].ravel(), y_preds[:, 1, 0][order].ravel(), y_preds[:, 2, 0][order].ravel(), alpha=0.2)
coverage_scores = [coverage_score(y, y_preds[:, 1, i], y_preds[:, 2, i]) for i, _ in enumerate(alpha)]
plt.title(
    f"Target and effective coverages for alpha={alpha[0]:.2f}: ({1-alpha[0]:.3f}, {coverage_scores[0]:.3f})\n" +
    f"Target and effective coverages for alpha={alpha[1]:.2f}: ({1-alpha[1]:.3f}, {coverage_scores[1]:.3f})"
)
plt.show()
