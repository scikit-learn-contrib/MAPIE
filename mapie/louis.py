

import numpy as np
from mapie.quantile_regression import MapieQuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

np.random.seed(1)
rng = np.random.default_rng(1)
random_state = 42

def get_1d_data(
    funct, mu: float, sigma: float, n_samples: int, noise: float
):
    X_train = np.random.uniform(0, 10.0, size=n_samples).astype(np.float32)
    X_test = np.random.uniform(0, 10.0, size=n_samples).astype(np.float32)

    # X_train = rng.normal(mu, sigma, n_samples)
    # X_test = np.arange(mu - 4 * sigma, mu + 4 * sigma, sigma / 20.0)

    y_train, y_test = funct(X_train), funct(X_test)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    y_train += rng.normal(0, noise, y_train.shape[0])
    y_test += rng.normal(0, noise, y_test.shape[0])

    X_train = np.reshape(X_train, (n_train, 1))
    X_test = np.reshape(X_test, (n_test, 1))
    return (
        X_train.reshape(-1, 1),
        y_train,
        X_test.reshape(-1, 1),
        y_test
    )


def f(x):
    ax = 0*x
    for i in range(len(x)):
        ax[i] = rng.poisson(np.sin(x[i])**2+0.1) + 0.03*x[i]*rng.random()
        ax[i] += 25*(rng.uniform(0, 1, 1) < 0.01)*rng.random()
    return ax.astype(np.float32)


# number of training examples
n_train = 3000

# number of test examples (to evaluate average coverage and length)
n_test = 1000


# training features
X_train = rng.uniform(0, 5.0, size=n_train).astype(np.float32)
X_test = rng.uniform(0, 5.0, size=n_test).astype(np.float32)

# generate labels
y_train = f(X_train)
y_test = f(X_test)

# reshape the features
X_train = np.reshape(X_train, (n_train, 1))
X_test = np.reshape(X_test, (n_test, 1))

mapie_reg = MapieQuantileRegressor(
    GradientBoostingRegressor(loss="quantile"),
    cv="simple",
    alpha=0.1
)
# mapie_reg = MapieRegressor(
#     cv=2
# )
mapie_reg.fit(X_train, y_train)

# print(mapie_reg.list_estimators)
# print(mapie_reg.list_y_preds_calib)
# print(mapie_reg.list_y_preds_calib)

# print()
# print(len(mapie_reg.list_y_preds))
# print(np.round(mapie_reg.list_y_preds[0], 2))
# print(np.round(mapie_reg.list_y_preds[1], 2))
# print(np.round(mapie_reg.list_y_preds[2], 2))

y_pred = mapie_reg.predict(X_test, symmetry=False)
# print(y_pred)
