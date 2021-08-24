"""
=============================================================
Plotting MAPIE score conformal predictions with a toy dataset
=============================================================

An example plot of :class:`mapie.classification.MapieClassifier`.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from mapie.classification import MapieClassifier

# Create training, and calibration datasets from blobs
centers = [(-5, -5), (5, -5), (-5, 5), (5, 5)]
x_min, x_max, y_min, y_max, step = -15, 15, -15, 15, 0.1
X_train_val, y_train_val = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=centers,
    cluster_std=2.5,
    random_state=59
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)
# Create test from (x, y) coordinates
X_test = np.stack([
    [x, y]
    for x in np.arange(x_min, x_max, step)
    for y in np.arange(x_min, x_max, step)
])

# Fit a Logistic Regression model on the training set
clf = LogisticRegression(multi_class="multinomial")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)

# Apply MapieClassifier on the calibration set to get prediction sets
mapie = MapieClassifier(estimator=clf, cv="prefit")
mapie.fit(X_val, y_val)
y_pred_mapie, y_pi_mapie = mapie.predict(X_test, alpha=0.01)
y_pi_sums = y_pi_mapie.sum(axis=1)

# Plot the results
tab10 = plt.cm.get_cmap('Purples', 4)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
y_pred_col = [colors[i] for _, i in enumerate(y_pred)]
y_train_col = [colors[i] for _, i in enumerate(y_train)]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
ax1.scatter(
    X_test[:, 0],
    X_test[:, 1],
    color=y_pred_col,
    marker='.',
    s=1, alpha=0.4
)
ax1.scatter(
    X_train[:, 0],
    X_train[:, 1],
    color=y_train_col,
    marker='o',
    s=10,
    edgecolor='k'
)
ax1.set_title("Predicted labels")
max_probs = ax2.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pred_proba_max,
    marker='.',
    s=2,
    alpha=0.4,
    cmap="Purples"
)
ax2.scatter(
    X_train[:, 0],
    X_train[:, 1],
    marker='o',
    color=y_train_col,
    s=10,
    edgecolor='k'
)
cbar = plt.colorbar(max_probs, ax=ax2)
ax2.set_title("Maximum probabilities")
num_labels = ax3.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_pi_sums,
    marker='.',
    s=2,
    alpha=1,
    cmap=tab10
)
ax3.scatter(
    X_train[:, 0],
    X_train[:, 1],
    marker='o',
    color=y_train_col,
    s=10,
    edgecolor='k'
)
cbar = plt.colorbar(num_labels, ax=ax3)
ax3.set_title("Number of labels in prediction sets")
plt.show()
