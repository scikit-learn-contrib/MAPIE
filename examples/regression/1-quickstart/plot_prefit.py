"""
========================================================
Example use of the prefit parameter with neural networks
========================================================

:class:`mapie.regression.MapieRegressor` and
:class:`mapie.quantile_regression.MapieQuantileRegressor``
are used to calibrate uncertainties for large models for
which the cost of cross-validation is too high. Typically,
neural networks rely on a single validation set.

In this example, we first fit a neural network on the training set. We
then compute residuals on a validation set with the `cv="prefit"` parameter.
Finally, we evaluate the model with prediction intervals on a testing set.
"""


import numpy as np
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
import scipy
from sklearn.model_selection import train_test_split

from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.metrics import regression_coverage_score
from mapie._typing import NDArray


def f(x: NDArray) -> NDArray:
    """Polynomial function used to generate one-dimensional data."""
    return np.array(5 * x + 5 * x**4 - 9 * x**2)


# Generate data
sigma = 0.1
n_samples = 10000
X = np.linspace(0, 1, n_samples)
y = f(X) + np.random.normal(0, sigma, n_samples)

# Train/validation/test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=1 / 10
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=1 / 9
)

# Train model on training set for MapieRegressor
model = estimator = LGBMRegressor(
    objective='quantile',
    alpha=0.5,
)
model.fit(X_train.reshape(-1, 1), y_train)

# Calibrate uncertainties on validation set
mapie = MapieRegressor(model, cv="prefit")
mapie.fit(X_val.reshape(-1, 1), y_val)

# Evaluate prediction and coverage level on testing set
alpha = 0.1
y_pred, y_pis = mapie.predict(X_test.reshape(-1, 1), alpha=alpha)
y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
coverage = regression_coverage_score(y_test, y_pred_low, y_pred_up)

# Train models for MapieQuantileRegressor
list_estimators = []
estimator_low = LGBMRegressor(
    objective='quantile',
    alpha=(alpha/2),
)
estimator_low.fit(X_train.reshape(-1, 1), y_train)
list_estimators.append(estimator_low)

estimator_high = LGBMRegressor(
    objective='quantile',
    alpha=(1-(alpha/2)),
)
estimator_high.fit(X_train.reshape(-1, 1), y_train)
list_estimators.append(estimator_high)


estimator = LGBMRegressor(
    objective='quantile',
    alpha=0.5,
)  # Note that this is the same model as used for QR
estimator.fit(X_train.reshape(-1, 1), y_train)
list_estimators.append(estimator)

# Calibrate uncertainties on validation set
mapie_cqr = MapieQuantileRegressor(list_estimators, cv="prefit")
mapie_cqr.fit(X_val.reshape(-1, 1), y_val)
y_pred_cqr, y_pis_cqr = mapie_cqr.predict(X_test.reshape(-1, 1))
y_pred_low_cqr, y_pred_up_cqr = y_pis_cqr[:, 0, 0], y_pis_cqr[:, 1, 0]
coverage_cqr = regression_coverage_score(y_test, y_pred_low_cqr, y_pred_up_cqr)

# Plot obtained prediction intervals on testing set
theoretical_semi_width = scipy.stats.norm.ppf(1 - alpha) * sigma
y_test_theoretical = f(X_test)
order = np.argsort(X_test)

plt.scatter(X_test, y_test, color="red", alpha=0.3, label="testing", s=2)
plt.plot(
    X_test[order],
    y_test_theoretical[order],
    color="gray",
    label="True confidence intervals",
)
plt.plot(
    X_test[order],
    y_test_theoretical[order] - theoretical_semi_width,
    color="gray",
    ls="--",
)
plt.plot(
    X_test[order],
    y_test_theoretical[order] + theoretical_semi_width,
    color="gray",
    ls="--",
)
plt.plot(X_test[order], y_pred[order], label="Predictions")
plt.fill_between(
    X_test[order],
    y_pred_low[order],
    y_pred_up[order],
    alpha=0.4,
    label="prediction intervals QR"
)
plt.fill_between(
    X_test[order],
    y_pred_low_cqr[order],
    y_pred_up_cqr[order],
    alpha=0.4,
    label="prediction intervals CQR"
)
plt.title(
    f"Target and effective coverages for:\n "
    f"QR alpha={alpha}: ({1 - alpha:.3f}, {coverage:.3f})\n"
    f"CQR alpha={alpha}: ({1 - alpha:.3f}, {coverage_cqr:.3f})"
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
