"""
==========================================================
Use MAPIE to control multiple risks of a binary classifier
==========================================================

In this example, we explain how to do multi-risk control for binary classification with MAPIE.

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import FixedThresholdClassifier
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
# to compute statistically guaranteed thresholds using a multi-risk control method.
#
# Different risks or performance metrics have been implemented, such as precision,
# recall and accuracy, but you can also implement your own custom functions using
# :class:`~mapie.risk_control.BinaryClassificationRisk` and choose your own
# secondary objective (passed in ``best_predict_param_choice``)
#
# Note that if the secondary objective is not specified, the first risk in the list is used
# as the secondary objective by default. Here, we choose "recall" as the secondary objective.
#
# Here we consider the list of risks ["precision", "recall"] and choose "recall" as the secondary
# objective. Furthermore, we consider two scenarios according to different target levels
# for precision and recall.

##############################################################################
# The following table summarizes the configuration of both scenarios:
#
# +-------------------------------+------------------------+------------------------+
# | **Parameter**                 | **Scenario 1**         | **Scenario 2**         |
# +-------------------------------+------------------------+------------------------+
# | **List of lisks**             | ["precision", "recall"]| ["precision", "recall"]|
# +-------------------------------+------------------------+------------------------+
# | **List of target levels**     | [0.75, 0.70]           | [0.85, 0.80]           |
# +-------------------------------+------------------------+------------------------+
# | **Confidence level**          | 0.9                    | 0.9                    |
# +-------------------------------+------------------------+------------------------+
# | **Best predict param choice** | "recall"               | "recall"               |
# +-------------------------------+------------------------+------------------------+
#
# Both scenarios use the same list of risks and best parameter choice,
# but with different target levels for precision and recall.
#
# For each scenario, we first fit two mono-risk controllers, followed by a multi-risk controller.
# The objective is to illustrate that, even when mono-risk controllers find valid thresholds for both risks,
# the multi-risk controller may not find any threshold that satisfies both simultaneously
# with statistical guarantees.
#
# Note that in the mono-risk case, the best predict parameter is left as "auto".
# See :class:`~mapie.risk_control.BinaryClassificationController` documentation for more details.


##############################################################################

# Scenario 1:
target_levels_1 = [0.75, 0.70]
confidence_level_1 = 0.9

# Cas mono risk
bcc_precision_1 = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk="precision",
    target_level=target_levels_1[0],
    confidence_level=confidence_level_1,
    best_predict_param_choice="auto",
)
bcc_precision_1.calibrate(X_calib, y_calib)

bcc_recall_1 = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk="recall",
    target_level=target_levels_1[1],
    confidence_level=confidence_level_1,
    best_predict_param_choice="auto",
)
bcc_recall_1.calibrate(X_calib, y_calib)

# Cas multi risk
bcc_1 = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk=["precision", "recall"],
    target_level=target_levels_1,
    confidence_level=confidence_level_1,
    best_predict_param_choice="recall",
)
bcc_1.calibrate(X_calib, y_calib)

print(
    f"Scenario 1 - Multiple risks : {len(bcc_1.valid_predict_params)} "
    "thresholds found that guarantee a precision of "
    f"at least {target_levels_1[0]} and a recall of at least {target_levels_1[1]} "
    f"with a confidence of {confidence_level_1}.\n"
    "Among those, the one that maximizes the secondary objective "
    "(here, recall, passed in `best_predict_param_choice`) is: "
    f"{bcc_1.best_predict_param:.3f}.\n"
)

##############################################################################

# Scenario 2:
target_levels_2 = [0.85, 0.8]
confidence_level_2 = 0.9

# Cas mono risk
bcc_precision_2 = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk="precision",
    target_level=target_levels_2[0],
    confidence_level=confidence_level_2,
    best_predict_param_choice="auto",
)
bcc_precision_2.calibrate(X_calib, y_calib)

bcc_recall_2 = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk="recall",
    target_level=target_levels_2[1],
    confidence_level=confidence_level_2,
    best_predict_param_choice="auto",
)
bcc_recall_2.calibrate(X_calib, y_calib)

# Cas multi risk
bcc_2 = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk=["precision", "recall"],
    target_level=target_levels_2,
    confidence_level=confidence_level_2,
    best_predict_param_choice="recall",
)
bcc_2.calibrate(X_calib, y_calib)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")  # Capture all warnings
    bcc_2.calibrate(X_calib, y_calib)

    if w:  # If any warnings were raised
        print(f"Scenario 2 - Multiple risks : {w[0].message}")

##############################################################################
# In the plot below, we visualize how the threshold values impact precision
# and recall, and what thresholds have been computed as statistically guaranteed.

proba_positive_class = clf.predict_proba(X_calib)[:, 1]
scenarios = [
    {
        "name": "Scenario 1 - Mono Risk",
        "bcc": [bcc_precision_1, bcc_recall_1],
        "target_levels": [target_levels_1[0], target_levels_1[1]],
    },
    {"name": "Scenario 1 - Multi Risk", "bcc": bcc_1, "target_levels": target_levels_1},
    {
        "name": "Scenario 2 - Mono Risk",
        "bcc": [bcc_precision_2, bcc_recall_2],
        "target_levels": [target_levels_2[0], target_levels_2[1]],
    },
    {"name": "Scenario 2 - Multi Risk", "bcc": bcc_2, "target_levels": target_levels_2},
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
axes = axes.flatten()

for ax, scenario in zip(axes, scenarios):
    if isinstance(scenario["bcc"], list):
        bcc_precision, bcc_recall = scenario["bcc"]
        target_precision, target_recall = scenario["target_levels"]
        tested_thresholds = bcc_precision._predict_params
        bccs = {"precision": bcc_precision, "recall": bcc_recall}
    else:
        bcc = scenario["bcc"]
        target_precision, target_recall = scenario["target_levels"]
        tested_thresholds = bcc._predict_params
        bccs = {"precision": bcc, "recall": bcc}

    metrics = {
        "precision": np.array(
            [
                precision_score(y_calib, (proba_positive_class >= t).astype(int))
                for t in tested_thresholds
            ]
        ),
        "recall": np.array(
            [
                recall_score(y_calib, (proba_positive_class >= t).astype(int))
                for t in tested_thresholds
            ]
        ),
    }

    valid_indices = {}
    best_indices = {}
    for key, controller in bccs.items():
        valid = controller.valid_predict_params
        if valid is None:
            valid = []
        valid = np.array(valid).tolist()
        valid_indices[key] = np.array([t in valid for t in tested_thresholds])
        best_indices[key] = (
            np.where(tested_thresholds == controller.best_predict_param)[0][0]
            if controller.best_predict_param in tested_thresholds
            else None
        )

    ax.scatter(
        tested_thresholds[valid_indices["precision"]],
        metrics["precision"][valid_indices["precision"]],
        color="tab:green",
        marker="o",
        label="Precision at valid thresholds",
    )
    ax.scatter(
        tested_thresholds[valid_indices["recall"]],
        metrics["recall"][valid_indices["recall"]],
        marker="p",
        facecolors="none",
        edgecolors="tab:green",
        label="Recall at valid thresholds",
    )
    ax.scatter(
        tested_thresholds[~valid_indices["precision"]],
        metrics["precision"][~valid_indices["precision"]],
        color="tab:red",
        marker="o",
        label="Precision at invalid thresholds",
    )
    ax.scatter(
        tested_thresholds[~valid_indices["recall"]],
        metrics["recall"][~valid_indices["recall"]],
        marker="p",
        facecolors="none",
        edgecolors="tab:orange",
        label="Recall at invalid thresholds",
    )

    if best_indices["precision"] is not None:
        ax.scatter(
            tested_thresholds[best_indices["precision"]],
            metrics["precision"][best_indices["precision"]],
            color="tab:green",
            marker="*",
            edgecolors="k",
            s=200,
            label="Precision best threshold",
        )
    if best_indices["recall"] is not None:
        ax.scatter(
            tested_thresholds[best_indices["recall"]],
            metrics["recall"][best_indices["recall"]],
            color="tab:blue",
            marker="*",
            edgecolors="k",
            s=200,
            label="Recall best threshold",
        )

    ax.axhline(target_precision, color="tab:gray", linestyle="--")
    ax.text(
        0.8,
        target_precision + 0.02,
        "Target precision",
        color="tab:gray",
        fontstyle="italic",
        fontsize=14,
    )
    ax.axhline(target_recall, color="tab:blue", linestyle=":")
    ax.text(
        0.0,
        target_recall + 0.02,
        "Target recall",
        color="tab:blue",
        fontstyle="italic",
        fontsize=14,
    )

    ax.set_xlabel("Threshold", fontsize=14)
    ax.set_ylabel("Performance metric Value", fontsize=14)
    ax.set_title(scenario["name"], fontsize=16)
    ax.legend(fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.suptitle("Precision and recall by threshold for all scenarios", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


##############################################################################
# Contrary to the naive way of computing a threshold to satisfy a precision and
# a recall targets on calibration data, risk control provides statistical guarantees
# on unseen data. In the plot above, we can see that not all thresholds corresponding
# to a precision (resp. recall) higher (resp. lower) than the target are valid.
# This is due to the uncertainty inherent to the finite size of the calibration set,
# which risk control takes into account.
#
# In particular, for instance, for precision, the highest threshold values are considered
# invalid due to the small number of observations used to compute the precision,
# following the Learn Then Test procedure. In the most extreme case, no observation
# is available, which causes the precision value to be ill-defined and set to 0.
#
# In scenario 1, both the mono-risk controllers and the multi-risk controller found
# valid thresholds that satisfy the precision and recall targets individually and jointly.
# The jointly valid thresholds found by the multi-risk controller are shown as green markers in the plot.
# In scenario 2, although valid thresholds are found individually for precision and recall
# by the mono-risk controllers, the multi-risk controller cannot find any threshold
# that satisfies both targets simultaneously.

# For Scenario 1 - Multi-risk only:
# Besides computing a set of valid thresholds,
# :class:`~mapie.risk_control.BinaryClassificationController` also outputs the "best"
# one, which is the valid threshold that maximizes a secondary objective
# (recall here).
#
# After obtaining the best threshold, we can use the ``predict`` function of
# :class:`~mapie.risk_control.BinaryClassificationController` for future predictions,
# or use scikit-learn's ``FixedThresholdClassifier`` as a wrapper to benefit
# from functionalities like easily plotting the decision boundary as seen below.

y_pred = bcc_1.predict(X_test)

clf_threshold = FixedThresholdClassifier(clf, threshold=bcc_1.best_predict_param)
clf_threshold.fit(X_train, y_train)
# .fit necessary for plotting, alternatively you can use sklearn.frozen.FrozenEstimator

disp = DecisionBoundaryDisplay.from_estimator(
    clf_threshold, X_test, response_method="predict", cmap=plt.cm.coolwarm
)

plt.scatter(
    X_test[y_test == 0, 0],
    X_test[y_test == 0, 1],
    edgecolors="k",
    c="tab:blue",
    alpha=0.5,
    label='"negative" class',
)
plt.scatter(
    X_test[y_test == 1, 0],
    X_test[y_test == 1, 1],
    edgecolors="k",
    c="tab:red",
    alpha=0.5,
    label='"positive" class',
)
plt.title(
    "Decision Boundary of FixedThresholdClassifier for the Scenario 1 - Multi Risk",
    fontsize=10,
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
