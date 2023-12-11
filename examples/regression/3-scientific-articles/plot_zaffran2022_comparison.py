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
repository 'https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries'
and compares the bounds of the PIs.

In order to reproduce the results of the github repository, we reuse the
``RandomForestRegressor`` regression model and follow the same conformal
prediction procedure (see 'https://github.com/mzaffran/\
AdaptiveConformalPredictionsTimeSeries/blob/\
131656fe4c25251bad745f52db3c2d7cb1c24bbb/models.py').

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
from tqdm import tqdm

import datetime
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit

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
plt.xticks(locs[0 : len(locs) : 2], labels=new_labels)

plt.xlabel("Date")
plt.ylabel("Spot price (\u20AC/MWh)")

# plt.show()

#########################################################
# Prepare data
#########################################################

limit = datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc)
id_train = data.index[pd.to_datetime(data["Date"], utc=True) < limit].tolist()
id_test = data.index[pd.to_datetime(data["Date"], utc=True) >= limit].tolist()

data_train = data.iloc[id_train, :]
data_test = data.iloc[id_test, :]

features = (
    ["hour"]
    + ["dow_%d" % i for i in range(7)]
    + ["lag_24_%d" % i for i in range(24)]
    + ["lag_168_%d" % i for i in range(24)]
    + ["conso"]
)
X_train = data_train.loc[:, features]
y_train = data_train.Spot

X_train_0 = X_train.loc[X_train.hour == 0]
y_train_0 = data_train.loc[data_train.hour == 0, "Spot"]

X_test = data_test.loc[:, features]
y_test = data_test.Spot

X_test_0 = X_test.loc[X_test.hour == 0]
y_test_0 = data_test.loc[data_test.hour == 0, "Spot"]

print(X_train_0.shape)

#########################################################
# Reproduce experiment and results
#########################################################

# all_x_train = [np.array(data_train.loc[data_train.hour == h]) for h in range(24)]
all_x_train = [np.array(X_train.loc[X_train.hour == h]) for h in range(24)]

print(X_train_0.shape)

train_size = X_train_0.shape[0]
idx = np.array(range(train_size))
n_half = int(np.floor(train_size / 2))

X_train_0 = X_train_0[:n_half]
y_train_0 = y_train_0[:n_half]


# Ici, on veut pas X_train & X_test, on veut X filtré avec vérification que la taille originale du train fait 1089 lignes

data_0 = data[data.hour == 0].reset_index(drop=True)

# Nous sert juste à obtenir la taille du train _size souhaitée
limit = datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc)
id_train_0 = data_0.index[pd.to_datetime(data_0["Date"], utc=True) < limit].tolist()
id_test_0 = data_0.index[pd.to_datetime(data_0["Date"], utc=True) >= limit].tolist()

data_train = data_0.iloc[id_train_0, :]
data_test = data_0.iloc[id_test_0, :]

train_size = data_train.shape[0]
test_size = data_test.shape[0]
print(train_size)

features = (
    ["hour"]
    + ["dow_%d" % i for i in range(7)]
    + ["lag_24_%d" % i for i in range(24)]
    + ["lag_168_%d" % i for i in range(24)]
    + ["conso"]
)

X = data_0.loc[:, features]
Y = data_0.Spot
print(X.shape)


#########################################################
# Prepare model
#########################################################

n_estimators = 1000
min_samples_leaf = 1
max_features = 6

model = RandomForestRegressor(
    n_estimators=n_estimators,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    random_state=1,
)

alpha = 0.1
gap = 1
iteration_max = 2
gamma = 0.04

# model = init_model()

test_fold = [0] * n_half + [-1] * n_half

print(n_half)
print(len(test_fold))

mapie_aci = MapieTimeSeriesRegressor(
    model,
    method="aci",
    agg_function="mean",
    n_jobs=-1,
    random_state=1,
    cv=PredefinedSplit(test_fold),
)

#########################################################
# Boucle entrainement
#########################################################

# Ici, on veut pour chaque pas, entrainer puis obtenir les pred conformes.
# Partial fit ou pas? Pour moi partial fit sert quand il y a plusieurs pred. Ce qui n'est pas notre cas

for step in tqdm(range(min(test_size, iteration_max + 1))):
    idx = np.array(range(train_size))
    n_half = int(np.floor(train_size / 2))
    idx_train, idx_cal = idx[:n_half], idx[n_half : 2 * n_half]

    x_train = X.iloc[step : (train_size + step),]
    x_test = np.array(X.iloc[(train_size + step),]).reshape((1, -1))
    y_train = Y[step : (train_size + step)]
    y_test = np.array([Y[(train_size + step)]])

    print(x_train.iloc[idx_train].shape)
    print(step)
    # print(y_test)

    print(len(idx_train))

    if step == 0:
        # D'abord on fit sur la première moitiée
        mapie_aci = mapie_aci.fit(x_train, y_train)  # Va sortir du if + prefit

        mapie_aci = mapie_aci.update(x_train, y_train)

        # Ajout d'une ligne :
        # mapie_aci.init_alpha()

        # Initialise le nombre de predictions qu'on veut faire grâce à la taille de x_test
        y_pred_aci_npfit, y_pis_aci_npfit = mapie_aci.predict(
            X.iloc[train_size:,], alpha=alpha, ensemble=True, optimize_beta=False
        )

        y_pred_aci_pfit = np.zeros(y_pred_aci_npfit.shape)
        y_pis_aci_pfit = np.zeros(y_pis_aci_npfit.shape)

        print("MAPIE estimator fitted!")

    else:
        print(step)

        mapie_aci = mapie_aci.update(x_train, y_train)
        # alpha_t = mapie_aci.current_alpha[0.1] # Va disparaitre
        # mapie_aci.fit(x_train.iloc[idx_train], y_train.iloc[idx_train]) # Va sortir du if + prefit (idem step == 0)
        # mapie_aci.current_alpha[0.1] = alpha_t # Va disparaitre

    # Ensuite, on update les pred conforme
    # mapie_aci.partial_fit(x_train.iloc[idx_cal], y_train.iloc[idx_cal])

    print("Update alpha")
    print(mapie_aci.current_alpha)
    print(y_test)

    (
        y_pred_aci_pfit[step : step + gap],
        y_pis_aci_pfit[step : step + gap, :, :],
    ) = mapie_aci.predict(
        x_test,
        alpha=alpha,
        ensemble=True,
        optimize_beta=False,
    )

    mapie_aci.adapt_conformal_inference(
        x_test,
        y_test,
        gamma=gamma,
    )

results = y_pis_aci_pfit.copy()

###

"""
mapie_aci = mapie_aci.fit(X_train_0, y_train_0)
y_pred_aci_npfit, y_pis_aci_npfit = mapie_aci.predict(
    X_test_0, alpha=alpha, ensemble=True, optimize_beta=False
)
print("MAPIE estimator fitted!")

# step 0
y_pred_aci_pfit = np.zeros(y_pred_aci_npfit.shape)
y_pis_aci_pfit = np.zeros(y_pis_aci_npfit.shape)
y_pred_aci_pfit[:gap], y_pis_aci_pfit[:gap, :, :] = mapie_aci.predict(
    X_test_0.iloc[:gap, :], alpha=alpha, ensemble=True, optimize_beta=False
)


# step t
for step in range(1, min(len(X_test_0), iteration_max + 1)):
    mapie_aci.estimator_.single_estimator_.fit(
        X_train_0.iloc[(step - gap) : step, :], y_train_0.iloc[(step - gap) : step]
    )

    mapie_aci.partial_fit(
        X_train_0.iloc[(step - gap) : step, :],
        y_train_0.iloc[(step - gap) : step],
    )

    mapie_aci.adapt_conformal_inference(
        X_train_0.iloc[(step - gap) : step, :],
        y_train_0.iloc[(step - gap) : step],
        gamma=gamma,
    )

    (
        y_pred_aci_pfit[step : step + gap],
        y_pis_aci_pfit[step : step + gap, :, :],
    ) = mapie_aci.predict(
        X_test_0.iloc[step : (step + gap), :],
        alpha=alpha,
        ensemble=True,
        optimize_beta=True,
    )

results = y_pis_aci_pfit.copy()
"""

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
print(data_ref.keys())


#########################################################
# Compare results
#########################################################

# Flatten the array to shape (n, 2)
results_ref = np.concatenate([data_ref["Y_inf"], data_ref["Y_sup"]], axis=0).T
results = np.array(results.reshape(-1, 2))

# Compare the NumPy array with the corresponding DataFrame columns
comparison_result_Y_inf = np.array_equal(
    results[:iteration_max, 0], results_ref[:iteration_max, 0]
)
comparison_result_Y_sup = np.array_equal(
    results[:iteration_max, 1], results_ref[:iteration_max, 1]
)

# Print the comparison results
print(f"Comparison (Y_inf): {results[:iteration_max, 0]}")
print(f"Comparison (Y_inf ref): {results_ref[:iteration_max, 0]}")
print(f"Comparison (Y_sup): {results[:iteration_max, 1]}")
print(f"Comparison (Y_sup ref): {results_ref[:iteration_max, 1]}")
print(f"Vraies valeures y: {list(Y[train_size:(train_size+iteration_max)])}")
print(f"Comparison for ACP_0.04 (Y_inf): {comparison_result_Y_inf}")
print(f"Comparison for ACP_0.04 (Y_sup): {comparison_result_Y_sup}")

print(f"Predictions: {y_pred_aci_pfit[:iteration_max]}")
print(
    f"Predictions ref: {[results_ref[iteration, 0]/2 + results_ref[iteration, 1]/2 for iteration in range(iteration_max)]}"
)

print()
# print(data_ref["alpha_t"][:iteration_max])
