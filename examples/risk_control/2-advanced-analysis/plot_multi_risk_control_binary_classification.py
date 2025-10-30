"""
========================================================================
Use MAPIE to control multiple performance metrics of a binary classifier
========================================================================

In this example, we explain how to do multi-risk control for binary classification with MAPIE.

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from mapie.risk_control import BinaryClassificationController
from mapie.utils import train_conformalize_test_split

RANDOM_STATE = 1

##############################################################################
# Fist, load the dataset and then split it into training, calibration
# (for conformalization), and test sets.

X, y = make_circles(n_samples=5000, noise=0.3, factor=0.3, random_state=RANDOM_STATE)
(X_train, X_calib, X_test, y_train, y_calib, y_test) = train_conformalize_test_split(
    X,
    y,
    train_size=0.8,
    conformalize_size=0.1,
    test_size=0.1,
    random_state=RANDOM_STATE,
)

# Plot the three datasets to visualize the distribution of the two classes.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["Training Data", "Calibration Data", "Test Data"]
datasets = [(X_train, y_train), (X_calib, y_calib), (X_test, y_test)]

for i, (ax, (X_data, y_data), title) in enumerate(zip(axes, datasets, titles)):
    ax.scatter(
        X_data[y_data == 0, 0],
        X_data[y_data == 0, 1],
        edgecolors="k",
        c="tab:blue",
        label='"negative" class',
        alpha=0.5,
    )
    ax.scatter(
        X_data[y_data == 1, 0],
        X_data[y_data == 1, 1],
        edgecolors="k",
        c="tab:red",
        label='"positive" class',
        alpha=0.5,
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Feature 1", fontsize=16)
    ax.tick_params(labelsize=14)

    if i == 0:
        ax.set_ylabel("Feature 2", fontsize=16)
    else:
        ax.set_ylabel("")
        ax.set_yticks([])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    ncol=2,
    fontsize=16,
)

plt.suptitle("Visualization of Train, Calibration, and Test Sets", fontsize=22)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()

##############################################################################
# Second, fit a Multi-layer Perceptron classifier on the training data.

clf = MLPClassifier(max_iter=150, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)

##############################################################################
# Next, we initialize a :class:`~mapie.risk_control.BinaryClassificationController`
# using the probability estimation function from the fitted estimator:
# ``clf.predict_proba``, a list risk or performance metric (here, ["precision", "recall"]),
# a list target risk level, and a single confidence level. Then we use the calibration data
# to compute statistically guaranteed thresholds using a risk control method.
#
# Different risks or performance metrics have been implemented, such as precision,
# recall and accuracy, but you can also implement your own custom function using
# :class:`~mapie.risk_control.BinaryClassificationRisk` and choose your own
# secondary objective.

target_precision = 0.7
target_recall = 0.75
confidence_level = 0.9

bcc = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk=["precision", "recall"],
    target_level=[target_precision, target_recall],
    confidence_level=confidence_level,
    best_predict_param_choice="precision",
)
bcc.calibrate(X_calib, y_calib)

print(
    f"{len(bcc.valid_predict_params)} thresholds found that guarantee a precision of "
    f"at least {target_precision} with a confidence of {confidence_level}.\n"
    "Among those, the one that maximizes the precision (passed in `best_predict_param_choice`) is: "
    f"{bcc.best_predict_param:.3f}."
)


##############################################################################
# In the plot below, we visualize how the threshold values impact precision and recall,
# and what thresholds have been computed as statistically guaranteed.

proba_positive_class = clf.predict_proba(X_calib)[:, 1]

tested_thresholds = bcc._predict_params
precisions = np.full(len(tested_thresholds), np.inf)
recalls = np.full(len(tested_thresholds), np.inf)
for i, threshold in enumerate(tested_thresholds):
    y_pred = (proba_positive_class >= threshold).astype(int)
    precisions[i] = precision_score(y_calib, y_pred)
    recalls[i] = recall_score(y_calib, y_pred)

valid_thresholds_indices = np.array(
    [t in bcc.valid_predict_params for t in tested_thresholds]
)
best_threshold_index = np.where(tested_thresholds == bcc.best_predict_param)[0][0]

plt.figure(figsize=(8, 6))
plt.scatter(
    tested_thresholds[valid_thresholds_indices],
    precisions[valid_thresholds_indices],
    c="tab:green", marker="o", label="Precision at Valid Thresholds"
)
plt.scatter(
    tested_thresholds[valid_thresholds_indices],
    recalls[valid_thresholds_indices],
    marker="p", facecolors="none", edgecolors="tab:green",
    label="Recall at Valid Thresholds"
)

plt.scatter(
    tested_thresholds[~valid_thresholds_indices],
    precisions[~valid_thresholds_indices],
    c="tab:red", marker="o", label="Precision at Invalid Thresholds"
)
plt.scatter(
    tested_thresholds[~valid_thresholds_indices],
    recalls[~valid_thresholds_indices],
    marker="p",
    facecolors="none",
    edgecolors="tab:blue",
    label="Recall at Invalid Thresholds",
)
plt.scatter(
    tested_thresholds[best_threshold_index],
    precisions[best_threshold_index],
    c="tab:green", marker="*", edgecolors="k", s=300, label="Best threshold"
)
plt.axhline(target_precision, color="tab:gray", linestyle="--")
plt.text(
    0.8,
    target_precision + 0.02,
    "Target precision",
    color="tab:gray",
    fontstyle="italic",
)
plt.axhline(target_recall, color="magenta", linestyle=":")
plt.text(
    0.0, target_recall + 0.02, "Target recall", color="magenta", fontstyle="italic"
)
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.title("Precision and Recall by Threshold")
plt.legend()
plt.tight_layout()
plt.show()
