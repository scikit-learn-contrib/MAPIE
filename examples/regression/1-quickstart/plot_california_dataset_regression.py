"""
==========================================================
Using California housing dataset for conformal regressions
==========================================================

We will use the sklearn california housing dataset as the
base for the comparaison of the different methods available
on MAPIE. Two classes will be used:
:class:`mapie.regression.MapieRegressor` and
:class:`mapie.quantile_regression.MapieQuantileRegressor`.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict
from matplotlib.offsetbox import (TextArea, AnnotationBbox)

from lightgbm import LGBMRegressor
from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from mapie.metrics import (
    regression_coverage_score,
    regression_mean_width_score
    )
from sklearn.datasets import fetch_california_housing
from scipy.stats import randint, uniform

from mapie.regression import MapieRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.subsample import Subsample
from mapie._typing import ArrayLike, NDArray

round_to = 3
random_state = 23
rng = np.random.default_rng(random_state)

warnings.filterwarnings("ignore")

##############################################################################
# 1. Data
# --------------------------------------------------------------------------


data = fetch_california_housing(as_frame=True)
X = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.DataFrame(data=data.target)*100

##############################################################################
# Let's visualize the two-dimensional dataset, the correlations between the
# independent variables and a histogram of the price of the houses.


df = pd.concat([X, y], axis=1)
pear_corr = df.corr(method='pearson')
print(pear_corr.style.background_gradient(cmap='Greens', axis=0))

fig, axs = plt.subplots(1, 1, figsize=(5, 5))
axs.hist(y, bins=50)
axs.set_xlabel("Prix median des maisons")
axs.set_title("Histogram des prix de maisons")
axs.xaxis.set_major_formatter(FormatStrFormatter('%.0f'+"k"))
plt.show()


##############################################################################
# Let's now create the different splits for the dataset, with a training,
# calibration and test set.


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y['MedHouseVal'],
    random_state=random_state
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train,
    y_train,
    random_state=random_state
)


##############################################################################
# 2. Optimizing estimator
# --------------------------------------------------------------------------
# Optimization of the `LGBMRegressor` using `RandomizedSearchCV` to find the
# optimal model to predict the house prices.


estimator = LGBMRegressor(
    objective='quantile',
    alpha=0.5,
    random_state=random_state
)
params_distributions = dict(
    num_leaves=randint(low=10, high=50),
    max_depth=randint(low=3, high=20),
    n_estimators=randint(low=50, high=300),
    learning_rate=uniform()
)
optim_model = RandomizedSearchCV(
    estimator,
    param_distributions=params_distributions,
    n_jobs=-1,
    n_iter=100,
    cv=KFold(n_splits=5),
    verbose=-1
)
optim_model.fit(X_train, y_train)
estimator = estimator.set_params(**optim_model.best_params_)


##############################################################################
# 3. Comparaison of MAPIE methods
# --------------------------------------------------------------------------
# We will now proceed to compare the different regression method available
# on MAPIE. For this tutorial we will compare the "naive", "jackknife plus ab",
# "cv plus" and "conformalized quantile regression".


def sort(
    y_test: NDArray, y_pred: NDArray, y_pis: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Sorting the dataset such that you can make better plots.

    Parameters
    ----------
    y_test : NDArray
        _description_
    y_pred : NDArray
        _description_
    y_pis : NDArray
        _description_

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        _description_
    """
    indices = np.argsort(y_test)
    y_test_sorted = np.array(y_test)[indices]
    y_pred_sorted = y_pred[indices]
    y_lower_bound = y_pis[:, 0, 0][indices]
    y_upper_bound = y_pis[:, 1, 0][indices]
    return y_test_sorted, y_pred_sorted, y_lower_bound, y_upper_bound


def plot(
    title: str,
    axs: plt.Axes,
    y_test_sorted: NDArray,
    y_pred_sorted: NDArray,
    lower_bound: NDArray,
    upper_bound: NDArray,
    coverage: float,
    width: float,
    num_plots_idx: NDArray,
) -> None:
    """
    Plotting of the different error bars, in red the test points
    that fall outside of the error bars.

    Parameters
    ----------
    title : str
        _description_
    axs : plt.Axes
        _description_
    y_test_sorted : NDArray
        _description_
    y_pred_sorted : NDArray
        _description_
    lower_bound : NDArray
        _description_
    upper_bound : NDArray
        _description_
    coverage : float
        _description_
    width : float
        _description_
    num_plots_idx : NDArray
        _description_
    """
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.0f'+"k"))
    axs.xaxis.set_major_formatter(FormatStrFormatter('%.0f'+"k"))

    lower_bound_ = np.take(lower_bound, num_plots_idx)
    y_pred_sorted_ = np.take(y_pred_sorted, num_plots_idx)
    y_test_sorted_ = np.take(y_test_sorted, num_plots_idx)

    error = y_pred_sorted_-lower_bound_

    warning1 = y_test_sorted_ > y_pred_sorted_+error
    warning2 = y_test_sorted_ < y_pred_sorted_-error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=error[~warnings],
        capsize=5, marker="o", elinewidth=2, linewidth=0
        )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=error[warnings],
        capsize=5, marker="o", elinewidth=2, linewidth=0, color="red"
        )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        marker="*", color="green"
    )
    axs.set_xlabel("Prix des maisons en $")
    axs.set_ylabel("Prédiction de prix des maisons en $")
    ab = AnnotationBbox(
        TextArea(f"Couverture: {coverage}\nLongeur d'intervalles: {width}"),
        xy=(np.min(y_test_sorted_)*3, np.max(y_pred_sorted_+error)*0.95),
        )
    axs.add_artist(ab)
    axs.set_title(title, fontweight='bold')


STRATEGIES = {
    "naive": {"method": "naive"},
    "cv_plus": {"method": "plus", "cv": 10},
    "jackknife_plus_ab": {"method": "plus", "cv": Subsample(n_resamplings=50)},
    "cqr": {"method": "quantile", "cv": "split", "alpha": 0.2},
}
y_pred, y_pis = {}, {}
y_test_sorted, y_pred_sorted, lower_bound, upper_bound = {}, {}, {}, {}
coverage, width = {}, {}
for strategy, params in STRATEGIES.items():
    if strategy == "cqr":
        mapie = MapieQuantileRegressor(estimator, **params)  # type: ignore
        mapie.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test)
    else:
        mapie = MapieRegressor(estimator, **params)  # type: ignore
        mapie.fit(X_train, y_train)
        y_pred[strategy], y_pis[strategy] = mapie.predict(X_test, alpha=0.2)
    (
        y_test_sorted[strategy],
        y_pred_sorted[strategy],
        lower_bound[strategy],
        upper_bound[strategy]
    ) = sort(y_test, y_pred[strategy], y_pis[strategy])
    coverage[strategy] = np.round(regression_coverage_score(
        y_test,
        y_pis[strategy][:, 0, 0],
        y_pis[strategy][:, 1, 0]
        ), round_to)
    width[strategy] = np.round(regression_mean_width_score(
        y_pis[strategy][:, 0, 0],
        y_pis[strategy][:, 1, 0]
        ), round_to)


perc_obs_plot = 0.02
num_plots = rng.choice(
    len(y_test), int(perc_obs_plot*len(y_test)), replace=False
    )
fig, axs = plt.subplots(2, 2, figsize=(15, 13))
coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
for strategy, coord in zip(STRATEGIES.keys(), coords):
    plot(
        strategy,
        coord,
        y_test_sorted[strategy],
        y_pred_sorted[strategy],
        lower_bound[strategy],
        upper_bound[strategy],
        coverage[strategy],
        width[strategy],
        num_plots
        )
plt.show()


##############################################################################
# We notice more adaptability of the prediction intervals for the
# conformalized quantile regression while the other methods have fixed
# interval width.


def get_bins(
    want: str,
    y_test: Dict,
    y_pred: Dict,
    lower_bound: Dict,
    upper_bound: Dict,
    STRATEGIES: Dict,
    bins: ArrayLike
) -> pd.DataFrame:
    """
    Splits the data into different sections according to the bins selected.
    Computes the coverage and width of that split.

    Parameters
    ----------
    want : str
        _description_
    y_test : Dict
        _description_
    y_pred : Dict
        _description_
    lower_bound : Dict
        _description_
    upper_bound : Dict
        _description_
    STRATEGIES : Dict
        _description_
    bins : ArrayLike
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    cuts = []
    cuts_ = pd.qcut(y_test["naive"], bins).unique()[:-1]
    for item in cuts_:
        cuts.append(item.left)
    cuts.append(cuts_[-1].right)
    cuts.append(np.max(y_test["naive"])+1)
    recap = {}  # type: ignore
    for i in range(len(cuts)-1):
        cut1, cut2 = cuts[i], cuts[i+1]
        name = f"[{np.round(cut1, 0)}, {np.round(cut2, 0)}]"
        recap[name] = []
        for strategy in STRATEGIES:
            indices = np.where(
                (y_test[strategy] > cut1)*(y_test[strategy] <= cut2)
                )
            y_test_trunc = np.take(y_test[strategy], indices)
            y_low_ = np.take(lower_bound[strategy], indices)
            y_high_ = np.take(upper_bound[strategy], indices)
            if want == "coverage":
                recap[name].append(regression_coverage_score(
                    y_test_trunc[0],
                    y_low_[0],
                    y_high_[0]))
            elif want == "width":
                recap[name].append(
                    regression_mean_width_score(y_low_[0], y_high_[0])
                    )
    recap_df = pd.DataFrame(recap, index=STRATEGIES)
    return recap_df


bins = list(np.arange(0, 1, 0.1))
binned_data = get_bins(
    "coverage",
    y_test_sorted,
    y_pred_sorted,
    lower_bound,
    upper_bound,
    STRATEGIES,
    bins
    )


##############################################################################
# To confirm this insights, we will now observe what happens when we plot
# the conditional coverage and interval width on these intervals.


fig = plt.figure()
binned_data.T.plot.bar(figsize=(12, 4))
plt.axhline(0.80, ls="--", color="k")
plt.ylabel("Conditional coverage")
plt.xlabel("House binned prices")
plt.xticks(rotation=345)
plt.ylim(0.3, 1.0)
plt.legend(loc=[1, 0])
plt.show()


##############################################################################
# What we observe from these results is that none of the methods seems to
# have conditional coverage. It is however suprising to see that the
# conformalized quantile regression does not outperform the other methods.


binned_data = get_bins(
    "width",
    y_test_sorted,
    y_pred_sorted,
    lower_bound,
    upper_bound,
    STRATEGIES,
    bins
    )


fig = plt.figure()
binned_data.T.plot.bar(figsize=(12, 4))
plt.ylabel("Interval width")
plt.xlabel("House binned prices")
plt.xticks(rotation=350)
plt.legend(loc=[1, 0])
plt.show()


##############################################################################
# When observing the values of the the interval width we again see what was
# observed in the previous graphs with the interval widths. We can again see
# that the prediction intervals are larger as the price of the houses
# increases, interestingly, it's important to note that the prediction
# intervals are shorter when the estimator is more certain.
