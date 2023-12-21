"""
======================================================================
Reproduction of part of the paper experiments of Zaffran et al. (2022)
======================================================================

:class:`~mapie.regression.MapieTimeSeriesRegressor` is used to reproduce a
part of the paper experiments of Zaffran et al. (2022) in their article [1]
which we argue that Adaptive Conformal Inference (ACI, Gibbs & Candès, 2021)
[2], developed for distribution-shift time series, is a good procedure for
time series with general dependency.

For a given model, the simulation adjusts the MAPIE regressors using aci
method, on a dataset taken from the article and available on the github
repository https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries
and compares the bounds of the PIs.

In order to reproduce the results of the github repository, we reuse the
``RandomForestRegressor`` regression model and follow the same conformal
prediction procedure
(see https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries/blob/131656fe4c25251bad745f52db3c2d7cb1c24bbb/models.py). # noqa

This simulation is carried out to check that the aci method implemented in
MAPIE gives the same results as [1], and that the bounds of the PIs are
obtained.

[1] Zaffran, M., Féron, O., Goude, Y., Josse, J., & Dieuleveut, A. (2022).
Adaptive conformal predictions for time series.
In International Conference on Machine Learning (pp. 25834-25866). PMLR.

[2] Gibbs, I., & Candes, E. (2021). Adaptive conformal inference under
distribution shift.
Advances in Neural Information Processing Systems, 34, 1660-1672.
"""
import warnings

from typing import Tuple
from urllib.request import urlopen
import ssl
import pickle

import datetime
import numpy as np
import pandas as pd
from matplotlib import pylab as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.time_series_regression import MapieTimeSeriesRegressor

from mapie._typing import NDArray

warnings.simplefilter("ignore")


#########################################################
# Global random forests parameters
#########################################################


def init_model():
    # the number of trees in the forest
    n_estimators = 1000

    # the minimum number of samples required to be at a leaf node
    # (default skgarden's parameter)
    min_samples_leaf = 1

    # the number of features to consider when looking for the best split
    # (default skgarden's parameter)
    max_features = 6

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=1,
    )

    return model


#########################################################
# Get data
#########################################################


def get_data() -> pd.DataFrame:
    """
    TODO

    Returns
    -------
    TODO
    """
    website = "https://raw.githubusercontent.com/"
    page = "mzaffran/AdaptiveConformalPredictionsTimeSeries/"
    folder = "131656fe4c25251bad745f52db3c2d7cb1c24bbb/data_prices/"
    file = "Prices_2016_2019_extract.csv"
    url = website + page + folder + file
    ssl._create_default_https_context = ssl._create_unverified_context
    df = pd.read_csv(url)
    return df


#########################################################
# Get & Present data
#########################################################

data = get_data()

date_data = pd.to_datetime(data.Date)

plt.figure(figsize=(10, 5))
plt.plot(date_data, data.Spot, color="black", linewidth=0.6)

locs, labels = plt.xticks()
new_labels = ["2016", "2017", "2018", "2019", "2020"]
plt.xticks(locs[0:len(locs):2], labels=new_labels)

plt.xlabel("Date")
plt.ylabel("Spot price (\u20AC/MWh)")

# plt.show()


#########################################################
# Prepare data
#########################################################

limit = datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc)
id_train = data.index[pd.to_datetime(data['Date'], utc=True) < limit].tolist()

data_train = data.iloc[id_train, :]
sub_data_train = data_train.loc[:, [
    'hour', 'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
    ] + ['lag_24_%d' % i for i in range(24)] +
    ['lag_168_%d' % i for i in range(24)] + ['conso']
]
all_x_train = [
    np.array(sub_data_train.loc[sub_data_train.hour == h]) for h in range(24)
]

sub_data = data.loc[:, [
    'hour', 'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6'
    ] + ['lag_24_%d' % i for i in range(24)] +
    ['lag_168_%d' % i for i in range(24)] + ['conso']
]

all_x = [np.array(sub_data.loc[sub_data.hour == h]) for h in range(24)]
all_y = [np.array(data.loc[data.hour == h, 'Spot']) for h in range(24)]


#########################################################
# Reproduce experiment and results
#########################################################

h = 0  # Let define hour = 0

X = all_x[h]
Y = all_y[h]

n = len(Y)
train_size = all_x_train[0].shape[0]
test_size = n - train_size

idx = np.array(range(train_size))
n_half = int(np.floor(train_size / 2))
idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]


#########################################################
# Prepare model
#########################################################

iteration_max = 3
alpha = 0.1
gamma = 0.04

model = init_model()

mapie_aci = MapieTimeSeriesRegressor(
    model,
    method="aci",
    agg_function="mean",
    conformity_score=AbsoluteConformityScore(sym=True),
    cv=PredefinedSplit(test_fold=[-1] * n_half + [0] * n_half),
    random_state=1,
)

#########################################################
# Reproduce experiment and results
#########################################################

# Ici, on veut pour chaque pas, entrainer puis obtenir les pred conformes.
# Partial fit ou pas? Pour moi partial fit sert quand il y a plusieurs pred.
# Ce qui n'est pas notre cas

y_pred_aci_pfit = np.zeros(((365, )))
y_pis_aci_pfit = np.zeros(((365, 2, 1)))

for i in range(min(test_size, iteration_max + 1)):
    x_train = np.array(X[i:(train_size+i), ])
    x_test = np.array(X[(train_size+i), ]).reshape(1, -1)
    y_train = np.array(Y[i:(train_size+i)])
    y_test = np.array(Y[(train_size+i)]).reshape(1, -1)

    # Fit the model with new tran/calib dataset
    mapie_aci = mapie_aci.fit(x_train, y_train)

    # Predict on test dataset
    y_pred_aci_pfit[i:i+1], y_pis_aci_pfit[i:i+1] = mapie_aci.predict(
        x_test, alpha=alpha, ensemble=False, optimize_beta=False
    )

    # Update the current_alpha_t (hidden for the user)
    mapie_aci.update(
        x_test, y_test, gamma=gamma, ensemble=False, optimize_beta=False
    )

results = y_pis_aci_pfit.copy()


#########################################################
# Get referenced result to reproduce
#########################################################


def get_pickle() -> Tuple[NDArray, NDArray]:
    """
    # TODO

    # Returns
    # -------
    # TODO"""
    website = "https://github.com/"
    page = "mzaffran/AdaptiveConformalPredictionsTimeSeries/raw/"
    folder = "131656fe4c25251bad745f52db3c2d7cb1c24bbb/results/"
    folder += "Spot_France_Hour_0_train_2019-01-01/"
    file = "ACP_0.04_RF.pkl"
    url = website + page + folder + file
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        loaded_data = pickle.load(urlopen(url))
    except FileNotFoundError:
        print(f"The file {file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return loaded_data


data_ref = get_pickle()


#########################################################
# Compare results
#########################################################

# Flatten the array to shape (n, 2)
results_ref = np.concatenate([data_ref["Y_inf"], data_ref["Y_sup"]], axis=0).T
results = np.array(results.reshape(-1, 2))

# Compare the NumPy array with the corresponding DataFrame columns
comparison_result_Y_inf = np.allclose(
    results[:iteration_max, 0], results_ref[:iteration_max, 0], rtol=1e-2
)
comparison_result_Y_sup = np.allclose(
    results[:iteration_max, 1], results_ref[:iteration_max, 1], rtol=1e-2
)
comparison_result_Y_pred = np.allclose(
    y_pred_aci_pfit[:iteration_max], np.sum(results_ref, -1)[:iteration_max]/2,
    rtol=1e-2
)

# Print the comparison results
# The results are very closed but not exactly the same because of the quantile
# calculation. In MAPIE, we use method="higher" when in the code of Zaffran,
# it use method="midpoint".
print(f"Comparison (Y_inf): {results[:iteration_max, 0]}")
print(f"Comparison (Y_inf ref): {results_ref[:iteration_max, 0]}")
print(f"Comparison (Y_sup): {results[:iteration_max, 1]}")
print(f"Comparison (Y_sup ref): {results_ref[:iteration_max, 1]}")
print(f"Predictions: {y_pred_aci_pfit[:iteration_max]}")
print(f"Predictions ref: {np.sum(results_ref, -1)[:iteration_max]/2}")
print(f"Comparison for ACP_0.04 (Y_inf): {comparison_result_Y_inf}")
print(f"Comparison for ACP_0.04 (Y_sup): {comparison_result_Y_sup}")
print(f"Comparison for ACP_0.04 (Y_pred): {comparison_result_Y_pred}")
