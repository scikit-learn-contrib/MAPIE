"""
========================================================
Example use of the prefit parameter with neural networks
========================================================

:class:`mapie.estimators.MapieRegressor` is used to calibrate
uncertainties for large models for which the cost of cross-validation
is too high. Typically, neural networks rely on a single validation set.

In this example, we first fit a neural network on the training set. We
then compute residuals on a validation set with the `cv="prefit"` parameter.
Finally, we evaluate the model with prediction intervals on a testing set.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt

from mapie.estimators import MapieRegressor
from mapie.metrics import coverage_score


def f(x: np.ndarray) -> np.ndarray:
    """Polynomial function used to generate one-dimensional data"""
    return 5*x + 5*x**4 - 9*x**2


# Generate data
sigma = 0.1
n_samples = 10000
X = np.linspace(0, 1, n_samples)
y = f(X) + np.random.normal(0, sigma, n_samples)

# Train/validation/test split
train_cutoff = int(n_samples*0.8)
val_cutoff = int(n_samples*0.9)
X_train, y_train = X[:train_cutoff], y[:train_cutoff]
X_val, y_val = X[train_cutoff:val_cutoff], y[train_cutoff:val_cutoff]
X_test, y_test = X[val_cutoff:], y[val_cutoff:]

# Train model on training set
model = MLPRegressor()
model.fit(X_train, y_train)

# Calibrate uncertainties on validation set
mapie = MapieRegressor(model, cv="prefit")
mapie.fit(X_val, y_val)

# Evaluate prediction and coverage level on testing set
y_pred, y_pred_low, y_pred_up = mapie.predict(X_test)[:, :, 0].T
coverage_score = coverage_score(y_test, y_pred_low, y_pred_up)

plt.figure(figsize=(18, 12))
plt.set_xlabel("x")
plt.set_ylabel("y")
plt.set_xlim([0, 1.1])
plt.set_ylim([0, 1])
plt.scatter(X_train, y_train, color="red", alpha=0.3, label="training")
plt.plot(X_test, y_test, color="gray", label="True confidence intervals")
plt.plot(X_test, y_test - y_test_sigma, color="gray", ls="--")
plt.plot(X_test, y_test + y_test_sigma, color="gray", ls="--")
plt.plot(X_test, y_pred, label="Prediction intervals")
plt.fill_between(X_test, y_pred_low, y_pred_up, alpha=0.3)
plt.set_title(title)
plt.legend()
