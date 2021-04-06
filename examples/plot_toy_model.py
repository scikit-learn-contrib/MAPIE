"""
==============================================
Plotting PredictionInterval with a toy dataset
==============================================

An example plot of :class:`mapie.MapieRegressor`
"""
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from mapie import MapieRegressor

regressor = LinearRegression()
X_train, y_train = make_regression(n_samples=500, n_features=1)
y_train += np.random.normal(0, 20, y_train.shape[0])
X_test = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)

mapie = MapieRegressor(regressor)
mapie.fit(X_train, y_train)
y_preds = mapie.predict(X_test)

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X_train, y_train, alpha=0.3)
plt.plot(X_test, y_preds[:, 0], color='C1')
plt.fill_between(X_test.ravel(), y_preds[:, 1], y_preds[:, 2], alpha=0.3)
plt.show()
