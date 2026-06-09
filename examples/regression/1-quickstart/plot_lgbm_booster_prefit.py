"""
Use a LightGBM Booster loaded from disk
=======================================

LightGBM models trained with the native API (``lgb.train``), or saved to disk
and reloaded, are `lightgbm.Booster` objects. A `Booster` has no ``fit``
method, so it cannot be passed to MAPIE directly (see issue #403).

The solution is a thin scikit-learn-compatible wrapper exposing ``fit`` and
``predict``, used with `SplitConformalRegressor` in a prefit setting: since
the model is already trained, MAPIE never calls ``fit`` and only needs
``predict`` to compute conformity scores.

In this example, we train a LightGBM model with the native API, save it to
disk and reload it as a `Booster` — mimicking the common real-world scenario
of conformalizing a model loaded from a model registry or artifact store.
We then wrap the `Booster`, conformalize it on a calibration set, and
evaluate prediction intervals on a test set.
"""

import os
import tempfile
import warnings

import lightgbm as lgb
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from mapie.metrics.regression import regression_coverage_score
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 1
confidence_level = 0.9

##############################################################################
# We start by generating a one-dimensional, non-linear regression dataset
# with random noise, and split it into training, conformalization and test
# sets.


def f(x: NDArray) -> NDArray:
    """Polynomial function used to generate one-dimensional data."""
    return np.array(5 * x + 5 * x**4 - 9 * x**2)


rng = np.random.default_rng(RANDOM_STATE)
sigma = 0.1
n_samples = 10000
X = np.linspace(0, 1, n_samples).reshape(-1, 1)
y = f(X.ravel()) + rng.normal(0, sigma, n_samples)

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
# 1. Train a LightGBM model with the native API and save it to disk
# -----------------------------------------------------------------------------
#
# We train the model with ``lgb.train``, which returns a `lightgbm.Booster`.
# We then save it to disk and reload it with ``lgb.Booster(model_file=...)``,
# exactly as one would do with a model retrieved from a model store. Note
# that the reloaded object only has a ``predict`` method: there is no ``fit``.

params = {
    "objective": "regression",
    "verbosity": -1,
    "seed": RANDOM_STATE,
}
train_set = lgb.Dataset(X_train, label=y_train)
booster = lgb.train(params, train_set, num_boost_round=100)

with tempfile.TemporaryDirectory() as tmp_dir:
    model_path = os.path.join(tmp_dir, "lgbm_model.txt")
    booster.save_model(model_path)
    loaded_booster = lgb.Booster(model_file=model_path)

print(f"Reloaded model type: {type(loaded_booster).__name__}")
print(f"Has a 'fit' method: {hasattr(loaded_booster, 'fit')}")

##############################################################################
# 2. Wrap the Booster in a scikit-learn-compatible estimator
# -----------------------------------------------------------------------------
#
# MAPIE expects estimators with ``fit`` and ``predict`` methods. Since the
# `Booster` is already trained, the wrapper below simply delegates
# ``predict`` to the booster, and ``fit`` raises an error to prevent misuse
# (with ``prefit=True``, MAPIE never calls ``fit``).
#
# The trailing-underscore attribute ``fitted_`` and the
# ``__sklearn_is_fitted__`` method signal to MAPIE and scikit-learn that the
# wrapped model is already trained and ready to predict.


class BoosterRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn-compatible wrapper around a trained ``lightgbm.Booster``."""

    def __init__(self, booster: lgb.Booster) -> None:
        self.booster = booster
        self.fitted_ = True

    def fit(self, X: NDArray, y: NDArray) -> "BoosterRegressor":
        raise NotImplementedError(
            "The wrapped Booster is already trained: use this wrapper with prefit=True."
        )

    def predict(self, X: NDArray) -> NDArray:
        return self.booster.predict(X)

    def __sklearn_is_fitted__(self) -> bool:
        return True


wrapped_booster = BoosterRegressor(loaded_booster)

##############################################################################
# 3. Conformalize the wrapped model with MAPIE
# -----------------------------------------------------------------------------
#
# We pass the wrapper to `SplitConformalRegressor` with ``prefit=True``,
# estimate the conformity scores on the conformalization set, and compute
# prediction intervals together with the effective coverage on the test set.

mapie_regressor = SplitConformalRegressor(
    estimator=wrapped_booster, confidence_level=confidence_level, prefit=True
)
mapie_regressor.conformalize(X_conformalize, y_conformalize)

y_pred, y_pis = mapie_regressor.predict_interval(X_test)
coverage = regression_coverage_score(y_test, y_pis)[0]

print(
    f"For a confidence level of {confidence_level:.2f}, "
    f"the target coverage is {confidence_level:.3f}, "
    f"and the effective coverage is {coverage:.3f}."
)

##############################################################################
# 4. Plot the prediction intervals
# -----------------------------------------------------------------------------
#
# The effective coverage is close to the target coverage: the prediction
# intervals computed on the reloaded `Booster` are valid, even though the
# model was trained outside of scikit-learn and loaded from disk.

X_test_1d = X_test.ravel()
order = np.argsort(X_test_1d)

plt.figure(figsize=(8, 6))
plt.scatter(X_test_1d, y_test, color="red", alpha=0.3, label="testing", s=2)
plt.plot(
    X_test_1d[order],
    y_pred[order],
    color="green",
    label="Predictions LightGBM Booster",
)
plt.fill_between(
    X_test_1d[order],
    y_pis[:, 0, 0][order],
    y_pis[:, 1, 0][order],
    alpha=0.4,
    color="green",
    label="prediction intervals",
)
plt.title(
    f"LightGBM Booster with SplitConformalRegressor, "
    f"confidence_level={confidence_level}:\n"
    f"target coverage is {confidence_level:.3f}, "
    f"effective coverage is {coverage:.3f}"
)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3
)
plt.show()
