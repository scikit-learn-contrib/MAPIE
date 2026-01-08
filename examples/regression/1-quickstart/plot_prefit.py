"""
==========================================================================================================
Use MAPIE with a pre-trained model
==========================================================================================================


:class:`~mapie.regression.SplitConformalRegressor` and
:class:`~mapie.regression.ConformalizedQuantileRegressor`
are used to conformalize uncertainties for large models for
which the cost of cross-validation is too high. Typically,
neural networks rely on a single validation set.

In this example, we first fit a neural network on the training set. We
then compute residuals on a validation set with the ``prefit=True`` parameter.
Finally, we evaluate the model with prediction intervals on a testing set.
In a second part, we will also show how to use the prefit method in the
conformalized quantile regressor.
"""

import warnings

import numpy as np
import scipy
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor

from numpy.typing import NDArray
from mapie.metrics.regression import regression_coverage_score
from mapie.regression import SplitConformalRegressor, ConformalizedQuantileRegressor
from mapie.utils import train_conformalize_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 1
confidence_level = 0.9

##############################################################################
#
#
# We start by defining a function that we will use to generate data. We then
# add random noise to the y values. Then we split the dataset to have
# a training, conformalize and test set.


def f(x: NDArray) -> NDArray:
    """Polynomial function used to generate one-dimensional data."""
    return np.array(5 * x + 5 * x**4 - 9 * x**2)


# Generate data
rng = np.random.default_rng(59)
sigma = 0.1
n_samples = 10000
X = np.linspace(0, 1, n_samples)
y = f(X) + rng.normal(0, sigma, n_samples)

# Train/conformalize/test split
(X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.8,
        conformalize_size=0.1,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )
)


##############################################################################
# 1. Use a neural network
# -----------------------------------------------------------------------------
# 1.1 Pre-train a neural network
# -----------------------------------------------------------------------------
#
# For this example, we will train a
# :class:`~sklearn.neural_network.MLPRegressor` for
# :class:`~mapie.regression.SplitConformalRegressor`.


# Train a MLPRegressor for SplitConformalRegressor
est_mlp = MLPRegressor(activation="relu", random_state=RANDOM_STATE)
est_mlp.fit(X_train.reshape(-1, 1), y_train)


##############################################################################
# 1.2 Use MAPIE to conformalize the models
# -----------------------------------------------------------------------------
#
# We will now proceed to conformalize the models using MAPIE. To this aim, we set
# `prefit=True` so that we use the model that we already trained prior.
# We then predict using the test set and evaluate its coverage.


# Conformalize uncertainties on conformalize set
mapie = SplitConformalRegressor(
    estimator=est_mlp, confidence_level=confidence_level, prefit=True
)
mapie.conformalize(X_conformalize.reshape(-1, 1), y_conformalize)

# Evaluate prediction and coverage level on testing set
y_pred, y_pis = mapie.predict_interval(X_test.reshape(-1, 1))
coverage = regression_coverage_score(y_test, y_pis)[0]


##############################################################################
# 1.3 Plot results
# -----------------------------------------------------------------------------
#
# In order to view the results, we will plot the predictions of the
# the multi-layer perceptron (MLP) with their prediction intervals calculated with
# :class:`~mapie.regression.SplitConformalRegressor`.

# Plot obtained prediction intervals on testing set
theoretical_semi_width = scipy.stats.norm.ppf(1 - confidence_level) * sigma
y_test_theoretical = f(X_test)
order = np.argsort(X_test)

plt.figure(figsize=(8, 8))
plt.plot(X_test[order], y_pred[order], label="Predictions MLP", color="green")
plt.fill_between(
    X_test[order],
    y_pis[:, 0, 0][order],
    y_pis[:, 1, 0][order],
    alpha=0.4,
    label="prediction intervals SCR",
    color="green",
)

plt.title(
    f"Target and effective coverages for:\n "
    f"MLP with SplitConformalRegressor, confidence_level={confidence_level}: "
    + f"(coverage is {coverage:.3f})\n"
)
plt.scatter(X_test, y_test, color="red", alpha=0.7, label="testing", s=2)
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
plt.xlabel("x")
plt.ylabel("y")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3
)
plt.show()


##############################################################################
# 2. Use LGBM models
# -----------------------------------------------------------------------------
# 2.1 Pre-train LGBM models
# -----------------------------------------------------------------------------
#
# For this example, we will train multiple LGBMRegressor with a
# quantile objective as this is a requirement to perform conformalized
# quantile regression using
# :class:`~mapie.regression.ConformalizedQuantileRegressor`. Note that the
# three estimators need to be trained at quantile values of
# ``(1+confidence_level)/2, (1-confidence_level)/2, 0.5)``.

# Train LGBMRegressor models for _MapieQuantileRegressor
list_estimators_cqr = []
for alpha_ in [(1 - confidence_level) / 2, (1 + confidence_level) / 2, 0.5]:
    estimator_ = LGBMRegressor(
        objective="quantile",
        alpha=alpha_,
    )
    estimator_.fit(X_train.reshape(-1, 1), y_train)
    list_estimators_cqr.append(estimator_)

##############################################################################
# 2.2 Use MAPIE to conformalize the models
# -----------------------------------------------------------------------------
#
# We will now proceed to conformalize the models using MAPIE. To this aim, we set
# `prefit=True` so that we use the models that we already trained prior.
# We then predict using the test set and evaluate its coverage.

# Conformalize uncertainties on conformalize set
mapie_cqr = ConformalizedQuantileRegressor(
    list_estimators_cqr, confidence_level=0.9, prefit=True
)
mapie_cqr.conformalize(X_conformalize.reshape(-1, 1), y_conformalize)

# Evaluate prediction and coverage level on testing set
y_pred_cqr, y_pis_cqr = mapie_cqr.predict_interval(X_test.reshape(-1, 1))
coverage_cqr = regression_coverage_score(y_test, y_pis_cqr)[0]


##############################################################################
# 2.3 Plot results
# -----------------------------------------------------------------------------
#
# As fdor the MLP predictions, we plot the predictions of the LGBMRegressor
# with their prediction intervals calculated with
# :class:`~mapie.regression.ConformalizedQuantileRegressor`.

# Plot obtained prediction intervals on testing set
theoretical_semi_width = scipy.stats.norm.ppf(1 - confidence_level) * sigma
y_test_theoretical = f(X_test)
order = np.argsort(X_test)

plt.figure(figsize=(8, 8))

plt.plot(X_test[order], y_pred_cqr[order], label="Predictions LGBM", color="blue")
plt.fill_between(
    X_test[order],
    y_pis_cqr[:, 0, 0][order],
    y_pis_cqr[:, 1, 0][order],
    alpha=0.4,
    label="prediction intervals CQR",
    color="blue",
)
plt.title(
    f"Target and effective coverages for:\n "
    f"LGBM with ConformalizedQuantileRegressor, confidence_level={confidence_level}: "
    + f"(coverage is {coverage_cqr:.3f})"
)
plt.scatter(X_test, y_test, color="red", alpha=0.7, label="testing", s=2)
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
plt.xlabel("x")
plt.ylabel("y")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3
)
plt.show()
