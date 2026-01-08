"""
=====================================================================
Focus on residual normalised score
=====================================================================


We will use the sklearn california housing dataset to understand how the
residual normalised score works and show the multiple ways of using it.

We will explicit the experimental setup below.
"""

# sphinx_gallery_thumbnail_number = 2

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from numpy.typing import ArrayLike
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mapie.conformity_scores import ResidualNormalisedScore
from mapie.metrics.regression import regression_coverage_score, regression_ssc_score
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split

warnings.filterwarnings("ignore")

RANDOM_STATE = 1
rng = np.random.default_rng(RANDOM_STATE)

##############################################################################
# 1. Data
# --------------------------------------------------------------------------
# The target variable of this dataset is the median house value for the
# California districts. This dataset is composed of 8 features, including
# variables such as the age of the house, the median income of the
# neighborhood, the average number of rooms or bedrooms or even the location in
# latitude and longitude. In total there are around 20k observations.


data = fetch_california_housing()
X = data.data
y = data.target


##############################################################################
# Now let's visualize a histogram of the price of the houses.

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.hist(y, bins=50)
axs.set_xlabel("Median price of houses")
axs.set_title("Histogram of house prices")
axs.xaxis.set_major_formatter(FormatStrFormatter("%.0f" + "k"))
plt.show()


##############################################################################
# Let's now create the different splits for the dataset, with a training,
# conformalize, residual and test set. Recall that the conformalize set is used
# for calibrating the prediction intervals and the residual set is used to fit
# the residual estimator used by the
# :class:`~mapie.conformity_scores.ResidualNormalisedScore`.

np.array(X)
np.array(y)

(X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.7,
        conformalize_size=0.28,
        test_size=0.02,
        random_state=RANDOM_STATE,
    )
)

X_conformalize_prefit, X_res, y_conformalize_prefit, y_res = train_test_split(
    X_conformalize, y_conformalize, random_state=RANDOM_STATE, test_size=0.5
)


##############################################################################
# 2. Models
# --------------------------------------------------------------------------
# We will now define 4 different ways of using the residual normalised score.
# Remember that this score is only available in the split setup. First, the
# simplest one with all the default parameters :
# a :class:`~sklearn.linear_model.LinearRegression` is used for the residual
# estimator. (Note that to avoid negative values it is trained with the log
# of the features and the exponential of the predictions are used).
# It is also possible to use it with ``prefit=True`` i.e. with
# the base model trained beforehand. The third setup that we illustrate here
# is with the residual model prefitted : we can set the estimator in parameters
# of the class, not forgetting to specify ``prefit="True"``. Finally, as an
# example of the exotic parameterisation we use as a residual
# estimator a :class:`~sklearn.linear_model.LinearRegression` wrapped to avoid
# negative values like it is done by default in the class.


class PosEstim(LinearRegression):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        super().fit(X, np.log(np.maximum(y, np.full(y.shape, np.float64(1e-8)))))
        return self

    def predict(self, X):
        y_pred = super().predict(X)
        return np.exp(y_pred)


base_model = RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE)
base_model = base_model.fit(X_train, y_train)

residual_estimator = RandomForestRegressor(
    n_estimators=20, max_leaf_nodes=70, min_samples_leaf=7, random_state=RANDOM_STATE
)
residual_estimator = residual_estimator.fit(
    X_res, np.abs(np.subtract(y_res, base_model.predict(X_res)))
)
wrapped_residual_estimator = PosEstim().fit(
    X_res, np.abs(np.subtract(y_res, base_model.predict(X_res)))
)

CONFIDENCE_LEVEL = 0.9

# Estimating prediction intervals
STRATEGIES = {
    "Default": {
        "class": SplitConformalRegressor,
        "init_params": dict(
            confidence_level=CONFIDENCE_LEVEL,
            prefit=False,
            conformity_score=ResidualNormalisedScore(),
        ),
    },
    "Base model prefit": {
        "class": SplitConformalRegressor,
        "init_params": dict(
            estimator=base_model,
            confidence_level=CONFIDENCE_LEVEL,
            prefit=True,
            conformity_score=ResidualNormalisedScore(
                split_size=0.5,
                random_state=RANDOM_STATE,
            ),
        ),
    },
    "Base and residual model prefit": {
        "class": SplitConformalRegressor,
        "init_params": dict(
            estimator=base_model,
            confidence_level=CONFIDENCE_LEVEL,
            prefit=True,
            conformity_score=ResidualNormalisedScore(
                residual_estimator=residual_estimator,
                random_state=RANDOM_STATE,
                prefit=True,
            ),
        ),
    },
    "Wrapped residual model": {
        "class": SplitConformalRegressor,
        "init_params": dict(
            estimator=base_model,
            confidence_level=CONFIDENCE_LEVEL,
            prefit=True,
            conformity_score=ResidualNormalisedScore(
                residual_estimator=wrapped_residual_estimator,
                random_state=RANDOM_STATE,
                prefit=True,
            ),
        ),
    },
}

y_pred, y_pis, coverage, cond_coverage = {}, {}, {}, {}
num_bins = 10
for strategy_name, strategy_params in STRATEGIES.items():
    init_params = strategy_params["init_params"]
    class_ = strategy_params["class"]
    mapie = class_(**init_params)
    if mapie._prefit:
        mapie.conformalize(X_conformalize_prefit, y_conformalize_prefit)
    else:
        mapie.fit(X_train, y_train)
        mapie.conformalize(X_conformalize, y_conformalize)
    y_pred[strategy_name], y_pis[strategy_name] = mapie.predict_interval(X_test)
    coverage[strategy_name] = regression_coverage_score(y_test, y_pis[strategy_name])
    cond_coverage[strategy_name] = regression_ssc_score(
        y_test, y_pis[strategy_name], num_bins=num_bins
    )


def yerr(y_pred, intervals) -> ArrayLike:
    """
    Returns the error bars with the point prediction and its interval

    Parameters
    ----------
    y_pred: ArrayLike
        Point predictions.
    intervals: ArrayLike
        Predictions intervals.

    Returns
    -------
    ArrayLike
        Error bars.
    """
    return np.abs(
        np.concatenate(
            [
                np.expand_dims(y_pred, 0) - intervals[:, 0, 0].T,
                intervals[:, 1, 0].T - np.expand_dims(y_pred, 0),
            ],
            axis=0,
        )
    )


def plot_predictions(y, y_pred, intervals, coverage, cond_coverage, ax=None):
    """
    Plots y_true against y_pred with the associated interval.

    Parameters
    ----------
    y: ArrayLike
       Observed targets
    y_pred: ArrayLike
       Predictions
    intervals: ArrayLike
       Prediction intervals
    coverage: float
       Global coverage
    cond_coverage: float
        Maximum violation coverage
    ax: matplotlib axes
       An ax can be provided to include this plot in a subplot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.set_xlim([-0.5, 5.5])
    ax.set_ylim([-0.5, 5.5])

    error = y_pred - intervals[:, 0, 0]
    warning1 = y_test > y_pred + error
    warning2 = y_test < y_pred - error
    warnings = warning1 + warning2
    ax.errorbar(
        y[~warnings],
        y_pred[~warnings],
        yerr=np.abs(error[~warnings]),
        color="g",
        alpha=0.2,
        linestyle="None",
        label="Inside prediction interval",
    )
    ax.errorbar(
        y[warnings],
        y_pred[warnings],
        yerr=np.abs(error[warnings]),
        color="r",
        alpha=0.3,
        linestyle="None",
        label="Outside prediction interval",
    )

    ax.scatter(y, y_pred, s=3, color="black")
    ax.plot([0, max(max(y), max(y_pred))], [0, max(max(y), max(y_pred))], "-r")
    ax.set_title(
        f"{strategy} - coverage={coverage:.0%} "
        + f"- max violation={cond_coverage:.0%}"
    )
    ax.set_xlabel("y true")
    ax.set_ylabel("y pred")
    ax.legend()
    ax.grid()


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
for ax, strategy in zip(axs.flat, STRATEGIES.keys()):
    plot_predictions(
        y_test,
        y_pred[strategy],
        y_pis[strategy],
        coverage[strategy][0],
        cond_coverage[strategy][0],
        ax=ax,
    )

fig.suptitle(f"Predicted values and intervals of level {CONFIDENCE_LEVEL}")
plt.tight_layout()
plt.show()

##############################################################################
# The results show that all the setups reach the global coverage guaranteed of
# confidence_level.
# It is interesting to note that the "base model prefit" and the "wrapped
# residual model" give exactly the same results. And this is because they are
# the same models : one prefitted and one fitted directly in the class.
