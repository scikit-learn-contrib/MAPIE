"""
================================================================================
Conformal Predictive Distribution with MAPIE
================================================================================

"""

##############################################################################
# In this advanced analysis, we propose to use MAPIE for Conformal Predictive
# Distribution (CPD) in few steps. Here are some reference papers for more
# information about CPD:
#
# [1] Schweder, T., & Hjort, N. L. (2016). Confidence, likelihood, probability
# (Vol. 41). Cambridge University Press.
#
# [2] Vovk, V., Shen, J., Manokhin, V., & Xie, M. G. (2017, May). Nonparametric
# predictive distributions based on conformal prediction. In Conformal and
# probabilistic prediction and applications (pp. 82-102). PMLR.

import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from mapie.conformity_scores import AbsoluteConformityScore, ResidualNormalisedScore
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 15


##############################################################################
# 1. Generating toy dataset
# -------------------------
#
# Here, we propose just to generate data for regression task, then split it.

X, y = make_regression(
    n_samples=1000, n_features=1, noise=20, random_state=RANDOM_STATE
)

(X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.6,
        conformalize_size=0.2,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
)


plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_train, y_train, alpha=0.3)
plt.show()


##############################################################################
# 2. Defining a Conformal Predictive Distribution class with MAPIE
# ------------------------------------------------------------------
#
# To be able to obtain the cumulative distribution function of
# a prediction with MAPIE, we propose here to wrap the
# :class:`~mapie.regression.SplitConformalRegressor` to add a new method named
# `get_cumulative_distribution_function`.


class MapieConformalPredictiveDistribution(SplitConformalRegressor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_cumulative_distribution_function(self, X):
        y_pred, _ = self.predict_interval(X)
        cs = self._mapie_regressor.conformity_scores_[
            ~np.isnan(self._mapie_regressor.conformity_scores_)
        ]
        res = self._conformity_score.get_estimation_distribution(
            y_pred.reshape((-1, 1)), cs, X=X
        )
        return res


##############################################################################
# Now, we propose to use it with two different conformity scores -
# :class:`~mapie.conformity_scores.AbsoluteConformityScore` and
# :class:`~mapie.conformity_scores.ResidualNormalisedScore` -
# in split-conformal inference.

mapie_regressor_1 = MapieConformalPredictiveDistribution(
    estimator=LinearRegression(),
    conformity_score=AbsoluteConformityScore(sym=False),
    prefit=False,
)

mapie_regressor_1.fit(X_train, y_train)
mapie_regressor_1.conformalize(X_conformalize, y_conformalize)
y_pred_1, _ = mapie_regressor_1.predict_interval(X_test)
y_cdf_1 = mapie_regressor_1.get_cumulative_distribution_function(X_test)

mapie_regressor_2 = MapieConformalPredictiveDistribution(
    estimator=LinearRegression(),
    conformity_score=ResidualNormalisedScore(sym=False, random_state=RANDOM_STATE),
    prefit=False,
)

mapie_regressor_2.fit(X_train, y_train)
mapie_regressor_2.conformalize(X_conformalize, y_conformalize)
y_pred_2, _ = mapie_regressor_2.predict_interval(X_test)
y_cdf_2 = mapie_regressor_2.get_cumulative_distribution_function(X_test)

plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_test, y_test, alpha=0.3)
plt.plot(X_test, y_pred_1, color="C1")
plt.show()


##############################################################################
# 3. Visualizing the cumulative distribution function
# ---------------------------------------------------
#
# We now propose to visualize the cumulative distribution functions of
# the predictive distribution in a graph in order to compare the two methods.


nb_bins = 100


def plot_cdf(data, bins, **kwargs):
    counts, bins = np.histogram(data, bins=bins)
    cdf = np.cumsum(counts) / np.sum(counts)

    plt.plot(
        np.vstack((bins, np.roll(bins, -1))).T.flatten()[:-2],
        np.vstack((cdf, cdf)).T.flatten(),
        **kwargs,
    )


plot_cdf(y_cdf_1[0], bins=nb_bins, label="Absolute Residual Score", alpha=0.8)
plot_cdf(y_cdf_2[0], bins=nb_bins, label="Normalized Residual Score", alpha=0.8)
plt.vlines(y_pred_1[0], 0, 1, label="Prediction", color="C2", linestyles="dashed")
plt.legend(loc=2)
plt.show()
