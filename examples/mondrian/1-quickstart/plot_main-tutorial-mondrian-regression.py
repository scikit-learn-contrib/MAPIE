r"""
=============================================
Tutorial for tabular regression with Mondrian
=============================================

In this tutorial, we compare the prediction intervals estimated by MAPIE on a
simple, one-dimensional, ground truth function with classical conformal
prediction intervals versus Mondrian conformal prediction intervals.
The function is a sinusoidal function with added noise, and the data is
grouped in 10 groups. The goal is to estimate the prediction intervals
for new data points, and to compare the coverage of the prediction intervals
by groups.
Throughout this tutorial, we will answer the following questions:


- How to use MAPIE to estimate prediction intervals for a regression problem?
- How to use Mondrian conformal prediction intervals for regression?
- How to compare the coverage of the prediction intervals by groups?
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from mapie.metrics import regression_coverage_score_v2
from mapie.mondrian import MondrianCP
from mapie.regression import MapieRegressor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")


##############################################################################
# 1. Create the noisy dataset
# ----------------------------------------------------------------------------
# We create a dataset with 10 groups, each of those groups having a different
# level of noise.


n_points = 100000
np.random.seed(0)
X = np.linspace(0, 10, n_points).reshape(-1, 1)
group_size = n_points // 10
partition_list = []
for i in range(10):
    partition_list.append(np.array([i] * group_size))
partition = np.concatenate(partition_list)

noise_0_1 = np.random.normal(0, 0.1, group_size)
noise_1_2 = np.random.normal(0, 0.5, group_size)
noise_2_3 = np.random.normal(0, 1, group_size)
noise_3_4 = np.random.normal(0, .4, group_size)
noise_4_5 = np.random.normal(0, .2, group_size)
noise_5_6 = np.random.normal(0, .3, group_size)
noise_6_7 = np.random.normal(0, .6, group_size)
noise_7_8 = np.random.normal(0, .7, group_size)
noise_8_9 = np.random.normal(0, .8, group_size)
noise_9_10 = np.random.normal(0, .9, group_size)

y = np.concatenate(
    [
        np.sin(X[partition == 0, 0] * 2) + noise_0_1,
        np.sin(X[partition == 1, 0] * 2) + noise_1_2,
        np.sin(X[partition == 2, 0] * 2) + noise_2_3,
        np.sin(X[partition == 3, 0] * 2) + noise_3_4,
        np.sin(X[partition == 4, 0] * 2) + noise_4_5,
        np.sin(X[partition == 5, 0] * 2) + noise_5_6,
        np.sin(X[partition == 6, 0] * 2) + noise_6_7,
        np.sin(X[partition == 7, 0] * 2) + noise_7_8,
        np.sin(X[partition == 8, 0] * 2) + noise_8_9,
        np.sin(X[partition == 9, 0] * 2) + noise_9_10,
    ], axis=0
)


##############################################################################
# We plot the dataset with the partition as colors.


plt.scatter(X, y, c=partition)
plt.show()


##############################################################################
# 2. Split the dataset into a training set, a calibration set, and a test set.
# ----------------------------------------------------------------------------

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
partition_train_temp, partition_test, _, _ = train_test_split(
    partition, y, test_size=0.2, random_state=0
)
X_cal, X_train, y_cal, y_train = train_test_split(
    X_train_temp, y_train_temp, test_size=0.5, random_state=0
)
partition_cal, partition_train, _, _ = train_test_split(
    partition_train_temp, y_train_temp, test_size=0.5, random_state=0
)


##############################################################################
# We plot the training set, the calibration set, and the test set.


f, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].scatter(X_train, y_train, c=partition_train)
ax[0].set_title("Train set")
ax[1].scatter(X_cal, y_cal, c=partition_cal)
ax[1].set_title("Calibration set")
ax[2].scatter(X_test, y_test, c=partition_test)
ax[2].set_title("Test set")
plt.show()


##############################################################################
# 3. Fit a random forest regressor on the training set.
# ----------------------------------------------------------------------------

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)


##############################################################################
# 4. Fit a MapieRegressor and a MondrianCP on the calibration set.
# ----------------------------------------------------------------------------

mapie_regressor = MapieRegressor(rf, cv="prefit")
mondrian_regressor = MondrianCP(MapieRegressor(rf, cv="prefit"))
mapie_regressor.fit(X_cal, y_cal)
mondrian_regressor.fit(X_cal, y_cal, partition=partition_cal)


##############################################################################
# 5. Predict the prediction intervals on the test set with both methods.
# ----------------------------------------------------------------------------

_, y_pss_split = mapie_regressor.predict(X_test, alpha=.1)
_, y_pss_mondrian = mondrian_regressor.predict(
    X_test, partition=partition_test, alpha=.1
)


##############################################################################
# 6. Compare the coverage by partition, plot both methods side by side.
# ----------------------------------------------------------------------------

coverages = {}
for group in np.unique(partition_test):
    coverages[group] = {}
    coverages[group]["split"] = regression_coverage_score_v2(
        y_test[partition_test == group], y_pss_split[partition_test == group]
    )
    coverages[group]["mondrian"] = regression_coverage_score_v2(
        y_test[partition_test == group],
        y_pss_mondrian[partition_test == group]
    )


# Plot the coverage by groups, plot both methods side by side
plt.figure(figsize=(10, 5))
plt.bar(
    np.arange(len(coverages)) * 2,
    [float(coverages[group]["split"]) for group in coverages],
    label="Split"
)
plt.bar(
    np.arange(len(coverages)) * 2 + 1,
    [float(coverages[group]["mondrian"]) for group in coverages],
    label="Mondrian"
)
plt.xticks(
    np.arange(len(coverages)) * 2 + .5,
    [f"Group {group}" for group in coverages],
    rotation=45
)
plt.hlines(0.9, -1, 21, label="90% coverage", color="black", linestyle="--")
plt.ylabel("Coverage")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
