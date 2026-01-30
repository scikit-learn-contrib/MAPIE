"""
====================
Plot prediction sets
====================

In this example, we explain how to use MAPIE on a basic classification setting.
"""

##################################################################################
# We will use MAPIE to estimate prediction sets on a two-dimensional dataset with
# three labels.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score
from mapie.utils import train_conformalize_test_split

np.random.seed(42)

##############################################################################
# Firstly, let us create our dataset:

X, y = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=3.4)

(X_train, X_conformalize, X_test, y_train, y_conformalize, y_test) = (
    train_conformalize_test_split(
        X, y, train_size=0.4, conformalize_size=0.4, test_size=0.2
    )
)

##############################################################################
# We fit our training data with a KNN estimator.
# Then, we initialize a :class:`~mapie.classification.SplitConformalClassifier`
# using our estimator, indicating that it has already been fitted with
# `prefit=True`.
# Lastly, we compute the prediction sets with the desired confidence level using the
# ``conformalize`` and ``predict_set`` methods.

classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train, y_train)

confidence_level = 0.95
mapie_classifier = SplitConformalClassifier(
    estimator=classifier, confidence_level=confidence_level, prefit=True
)
mapie_classifier.conformalize(X_conformalize, y_conformalize)
y_pred, y_pred_set = mapie_classifier.predict_set(X_test)

##############################################################################
# ``y_pred`` represents the point predictions as a ``np.ndarray`` of shape
# ``(n_samples)``.
# ``y_pred_set`` corresponds to the prediction sets as a ``np.ndarray`` of shape
# ``(n_samples, 3, 1)``. This array contains only boolean values: ``True`` if the label
# is included in the prediction set, and ``False`` if not.

##############################################################################
# Finally, we can easily compute the coverage score (i.e., the proportion of times the
# true labels fall within the predicted sets).

coverage_score = classification_coverage_score(y_test, y_pred_set)
print(
    f"For a confidence level of {confidence_level:.2f}, "
    f"the target coverage is {confidence_level:.3f}, "
    f"and the effective coverage is {coverage_score[0]:.3f}."
)

##############################################################################
# In this example, the effective coverage is slightly above the target coverage
# (i.e., 0.95), indicating that the confidence level we set has been reached.
# Therefore, we can confirm that the prediction sets effectively contain the
# true label more than 95% of the time.

##############################################################################
# Now, let us plot the confidence regions across the plane.
# This plot will give us insights about what the prediction set looks like for each
# point.

x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
step = 0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
X_test_mesh = np.stack([xx.ravel(), yy.ravel()], axis=1)

y_pred_set = mapie_classifier.predict_set(X_test_mesh)[1][:, :, 0]

cmap_back = ListedColormap(
    [
        (0.7803921568627451, 0.9137254901960784, 0.7529411764705882),
        (0.9921568627450981, 0.8156862745098039, 0.6352941176470588),
        (0.6196078431372549, 0.6039215686274509, 0.7843137254901961),
        (0.7764705882352941, 0.8588235294117647, 0.9372549019607843),
        (0.6196078431372549, 0.6039215686274509, 0.7843137254901961),
        (0.6196078431372549, 0.6039215686274509, 0.7843137254901961),
    ]
)
cmap_dots = ListedColormap(
    [
        (0.19215686274509805, 0.5098039215686274, 0.7411764705882353),
        (0.9019607843137255, 0.3333333333333333, 0.050980392156862744),
        (0.19215686274509805, 0.6392156862745098, 0.32941176470588235),
    ]
)

plt.scatter(
    X_test_mesh[:, 0],
    X_test_mesh[:, 1],
    c=np.ravel_multi_index(y_pred_set.T, (2, 2, 2)),
    cmap=cmap_back,
    marker=".",
    s=10,
)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_dots)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Confidence regions with KNN")
plt.show()

##############################################################################
# On the plot above, the dots represent the samples from our dataset, with their
# color indicating their respective label.
# The blue, orange and green zones correspond to prediction sets
# containing only the blue label, orange label and green label respectively.
# The purple zone represents areas where the prediction sets contain more than one
# label, indicating that the model is uncertain.
