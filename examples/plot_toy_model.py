"""
==============================================
Plotting PredictionInterval with a toy dataset
==============================================

An example plot of :class:`mapie.PredictionInterval`
"""
import numpy as np

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

from mapie import PredictionInterval

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])

pireg = PredictionInterval(LinearRegression())
pireg.fit(X_toy, y_toy)

y_preds = pireg.predict(X_toy)

y_pred, y_low, y_up = y_preds[:, 0], y_preds[:, 1], y_preds[:, 2]

plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X_toy, y_toy)
plt.plot(X_toy, y_toy)

plt.show()
