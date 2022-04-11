"""
====================================================================
Capturing the aleatoric uncertainties on corrupted images with MAPIE
====================================================================

In this tutorial, we use the MNIST famous hand-written digits dataset from
scikit-learn
to assess how MAPIE captures the addition of some noise in training
images on the estimate of prediction sets.
"""

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from mapie.classification import MapieClassifier
from mapie.metrics import (
    classification_coverage_score,
    classification_mean_width_score
)
from mapie._typing import NDArray


##############################################################################
# 1. Loading and digits datasets
# ------------------------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


##############################################################################
# Now let's create a corrupted dataset with `n_splatters` splatters of
# `size_splatters` pixels on each image.

def generate_mnist_corrupted(
    n_splatters: int,
    size_splatters: int,
    datasets: Any
) -> Any:
    digits_corrupted = datasets.load_digits()
    n_samples = digits_corrupted["data"].shape[0]
    digits_corrupted["data"] = (
        digits_corrupted["data"].reshape(n_samples, 8, 8)
    )
    rnds = np.random.randint(
        0, 8 - (size_splatters - 1), (n_samples, n_splatters, 2)
    )
    for i in range(n_samples):
        for isplat in range(n_splatters):
            rnd = [rnds[i, isplat, 0], rnds[i, isplat, 1]]
            digits_corrupted["data"][
                i, rnd[0]:rnd[0] + size_splatters, rnd[1]:rnd[1]+size_splatters
            ] = 8
    digits_corrupted["data"] = digits_corrupted["data"].reshape(n_samples, 64)
    return digits_corrupted


digits_corrupted = generate_mnist_corrupted(8, 2, datasets)

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
zip_imgs = zip(axes, digits_corrupted.images, digits_corrupted.target)
for ax, image, label in zip_imgs:
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


##############################################################################
# 2. Training the base classifiers
# --------------------------------
# We follow the procedure of the scikit-learn tutorial for training a base
# regressor on our two datasets:
#
# - The images are flatten to turn each 2-D array of grayscale values from
#   shape ``(8, 8)`` into shape ``(64,)``. The entire dataset will be of shape
#   ``(n_samples, n_features)``, where ``n_samples`` is the number of images
#   and ``n_features`` is the total number of pixels in each image.
#
# - We can then split the data into train, calibration, and test subsets and
#   fit a support vector classifier on the train samples. The fitted classifier
#   can subsequently be used to predict the value of the digit for the samples
#   in the calibration and test subsets.

def get_datasets(dataset: Any) -> Tuple[
    NDArray, NDArray, NDArray, NDArray, NDArray, NDArray
]:
    n_samples = len(digits.images)
    data = dataset.images.reshape((n_samples, -1))
    X_train_calib, X_test, y_train_calib, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=True, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_calib,
        y_train_calib,
        test_size=0.25,
        shuffle=True,
        random_state=42
    )
    return X_train, X_calib, X_test, y_train, y_calib, y_test


X_train1, X_calib1, X_test1, y_train1, y_calib1, y_test1 = (
    get_datasets(digits)
)
X_train2, X_calib2, X_test2, y_train2, y_calib2, y_test2 = (
    get_datasets(digits_corrupted)
)

clf1 = svm.SVC(gamma=0.001, probability=True, random_state=42)
clf1.fit(X_train1, y_train1)
y_pred1 = clf1.predict(X_test1)
clf2 = svm.SVC(gamma=0.001, probability=True, random_state=42)
clf2.fit(X_train2, y_train2)
y_pred2 = clf2.predict(X_test2)

##############################################################################
# Below we visualize the first 5 test samples with their predicted digit value
# for both the original and corrupted datasets.

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test1, y_pred1):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')
_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test2, y_pred2):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Prediction: {prediction}')

##############################################################################
# We now compare some metrics as function of the image digits for
# both datasets.
# As expected, it can be noticed that the classifier performs worse
# on the corrupted
# dataset with an overall accuracy of 85 \%.

classif_report = pd.concat({
    "Original dataset": pd.DataFrame(
        metrics.classification_report(y_test1, y_pred1, output_dict=True)
    ).T,
    "Corrupted dataset": pd.DataFrame(
        metrics.classification_report(y_test2, y_pred2, output_dict=True)
    ).T
}, axis=1).round(3)
print(classif_report)

##############################################################################
# We can also plot a `confusion matrix` of the true digit values and the
# predicted digit values for both datasets.

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
disp1 = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test1, y_pred1, ax=axes[0]
)
disp2 = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test2, y_pred2, ax=axes[1]
)
disp1.figure_.suptitle("Confusion matrix - Original vs Corrupted datasets")


##############################################################################
# 3. Estimating prediction sets with MAPIE
# ----------------------------------------
# We now use :class:`mapie.classification.MapieClassifier` to estimate
# prediction sets for both datasets using the "cumulated_score" `method` and
# for `alpha` values ranging from 0.01 to 0.99.

alpha = np.arange(0.01, 1, 0.01)

mapie_clf1 = MapieClassifier(
    clf1, method="cumulated_score", cv="prefit", random_state=42
    )
mapie_clf1.fit(X_calib1, y_calib1)
y_pred1, y_ps1 = mapie_clf1.predict(
    X_test1, alpha=alpha, include_last_label="randomized"
)

mapie_clf2 = MapieClassifier(
    clf2, method="cumulated_score", cv="prefit", random_state=42
    )
mapie_clf2.fit(X_calib2, y_calib2)
y_pred2, y_ps2 = mapie_clf2.predict(
    X_test2, alpha=alpha, include_last_label="randomized"
)

##############################################################################
# We can then estimate the marginal coverage for all alpha values in order
# to produce a so-called calibration plot, comparing the target coverage with
# the "real" coverage obtained on the test set.

coverages1 = [
    classification_coverage_score(y_test1, y_ps1[:, :, i])
    for i, _ in enumerate(alpha)
]
coverages2 = [
    classification_coverage_score(y_test2, y_ps2[:, :, i])
    for i, _ in enumerate(alpha)
]
widths1 = [
    classification_mean_width_score(y_ps1[:, :, i])
    for i, _ in enumerate(alpha)
]
widths2 = [
    classification_mean_width_score(y_ps2[:, :, i])
    for i, _ in enumerate(alpha)
]

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
axes[0].set_xlabel("1 - alpha")
axes[0].set_ylabel("Effective coverage")
axes[0].scatter(1-alpha, coverages1)
axes[0].scatter(1-alpha, coverages2)
axes[0].plot([0, 1], [0, 1], ls="--", color="k")
axes[0].legend(["", "Original dataset", "Corrupted dataset"])
axes[1].set_xlabel("1 - alpha")
axes[1].set_ylabel("Average of prediction set sizes")
axes[1].scatter(1-alpha, widths1)
axes[1].scatter(1-alpha, widths2)
axes[1].legend(["Original dataset", "Corrupted dataset"])

cond_coverages1 = [
    classification_coverage_score(
        y_test1[y_test1 == inum], y_ps1[y_test1 == inum, :, 9]
    ) for inum in range(10)
]
cond_coverages2 = [
    classification_coverage_score(
        y_test2[y_test2 == inum], y_ps2[y_test2 == inum, :, 9]
    ) for inum in range(10)
]

##############################################################################
# We can also estimate the conditional coverages obtained by MAPIE on the
# different image numbers. One can notice for instance that the coverage is
# worse for numbers 3 or 9 but is close to 1 for number 7.

cond_coverages_df = pd.DataFrame(
    [cond_coverages1, cond_coverages2],
    index=["Original dataset", "Corrupted dataset"]
).T.round(3)
print(cond_coverages_df)


##############################################################################
# We now finish by visualizing some predictions for five images
# picked randomly.

y_ps_label1 = np.array(
    [
        [
            np.argwhere(y_ps1[isample, :, ialpha])
            for ialpha, _ in enumerate(alpha)
        ] for isample in range(len(y_test1))
    ], dtype="object"
)
y_ps_label2 = np.array(
    [
        [
            np.argwhere(y_ps2[isample, :, ialpha])
            for ialpha, _ in enumerate(alpha)
        ] for isample in range(len(y_test2))
    ], dtype="object"
)

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
num_images = np.random.randint(0, len(y_test2), 5)
for i, ax in enumerate(axes):
    ax.set_axis_off()
    image = X_test1[num_images[i], :].reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(
        f'Prediction:\n {np.array(y_ps_label1[num_images[i], 9]).ravel()}'
    )
plt.suptitle("Original dataset")
_, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.set_axis_off()
    image = X_test2[num_images[i], :].reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(
        f'Prediction:\n {np.array(y_ps_label2[num_images[i], 9]).ravel()}'
    )
plt.suptitle("Corrupted dataset")
plt.show()
