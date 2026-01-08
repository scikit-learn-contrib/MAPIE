"""
=========================================================
Use MAPIE to control the risk of a multi-label classifier
=========================================================

In this example, we explain how to perform risk control for multi-label
classification using the Learn-Then-Test (LTT) procedure implemented in MAPIE.

"""

# sphinx_gallery_thumbnail_number = 4

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB

from mapie.risk_control import MultiLabelClassificationController

RANDOM_STATE = 42

##############################################################################
# First, we generate a two-dimensional toy dataset with three possible labels.
# The idea is to create a triangle where the observations on the edges have only one
# label, those on the vertices have two labels (those of the two edges) and the
# center have all the labels.

# Generate synthetic dataset
np.random.seed(RANDOM_STATE)

centers = [(0, 10), (-5, 0), (5, 0), (0, 5), (0, 0), (-4, 5), (5, 5)]
covs = [
    np.eye(2),
    np.eye(2),
    np.eye(2),
    np.diag([5, 5]),
    np.diag([3, 1]),
    np.array([[4, 3], [3, 4]]),
    np.array([[3, -2], [-2, 3]]),
]

x_min, x_max, y_min, y_max, step = -15, 15, -5, 15, 0.1
n_samples = 800
X = np.vstack(
    [
        np.random.multivariate_normal(center, cov, n_samples)
        for center, cov in zip(centers, covs)
    ]
)
classes = [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
y = np.vstack([np.full((n_samples, 3), row) for row in classes])

# Split the dataset into training, calibration and test sets.
X_train_cal, X_test, y_train_cal, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_cal, y_train_cal, test_size=0.25
)

# Plot the three datasets to visualize the distribution of the labels.
colors = {
    (0, 0, 1): {"color": "#1f77b4", "lac": "0-0-1"},
    (0, 1, 1): {"color": "#ff7f0e", "lac": "0-1-1"},
    (1, 0, 1): {"color": "#2ca02c", "lac": "1-0-1"},
    (0, 1, 0): {"color": "#d62728", "lac": "0-1-0"},
    (1, 1, 0): {"color": "#ffd700", "lac": "1-1-0"},
    (1, 0, 0): {"color": "#c20078", "lac": "1-0-0"},
    (1, 1, 1): {"color": "#06C2AC", "lac": "1-1-1"},
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ["Training Data", "Calibration Data", "Test Data"]
datasets = [(X_train, y_train), (X_calib, y_calib), (X_test, y_test)]

for i, (ax, (X_data, y_data), title) in enumerate(zip(axes, datasets, titles)):
    for label, props in colors.items():
        label = np.array(label)
        mask = np.all(y_data == label, axis=1)

        ax.scatter(
            X_data[mask, 0],
            X_data[mask, 1],
            color=props["color"],
            edgecolors="k",
            s=60,
            alpha=1,
            label=props["lac"] if i == 0 else None,
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
    bbox_to_anchor=(0.5, 0),
    ncol=7,
    fontsize=18,
)

plt.suptitle("Visualization of Train, Calibration, and Test Sets", fontsize=22)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()

##############################################################################
# Second, we fit a MultiOutputClassifier by training one Gaussian Naive Bayes
# classifier per label. Using MultiOutputClassifier allows us to extend
# classifiers that do not natively support multi-label classification.
#

clf = MultiOutputClassifier(GaussianNB())
clf.fit(X_train, y_train)

##############################################################################
# Next, we initialize a :class:`~mapie.risk_control.MultiLabelClassificationController`
# using the probability estimation function from the fitted estimator:
# ``clf.predict_proba``, a chosen risk ("precision" in this example),
# a target risk level, and a confidence level. Then we use the calibration data
# to compute statistically valid thresholds using a risk control procedure.
#
# When ``risk="precision"``, the controller relies on the LTT procedure,
# which is designed to handle non-monotonic risks.
#
# Alternatively, ``risk="recall"`` can also be used.
# In that case, the controller relies on monotonicity and uses either:
#
# - RCPS (Risk-Controlling Prediction Sets), which provides a probabilistic guarantee,
# - CRC (Conformal Risk Control), which provides a guarantee in expectation.
#
# Please refer to the theoretical description of risk control in the MAPIE
# documentation for more details.

target_precision = 0.8
confidence_level = 0.9
mcc = MultiLabelClassificationController(
    predict_function=clf.predict_proba,
    risk="precision",
    method="ltt",
    predict_params=np.arange(0.01, 1, 0.01),
    target_level=target_precision,
    confidence_level=confidence_level,
)
mcc.calibrate(X_calib, y_calib)

print(
    f"{len(mcc.valid_predict_params[0])} thresholds found that guarantee a precision of "
    f"at least {target_precision} with a confidence of {confidence_level}. "
    f"The best threshold is: {mcc.best_predict_param[0]:.3f}."
)

##############################################################################
# In the plot below, we visualize how the threshold values impact precision, and what
# thresholds have been computed as statistically guaranteed.

tested_thresholds = mcc.predict_params
precisions = 1 - mcc.r_hat  # risk is defined as 1 - precision

naive_threshold_index = np.argmin(
    np.where(precisions >= target_precision, precisions - target_precision, np.inf)
)

valid_thresholds_indices = mcc.valid_index[0]  # valid_index is a list of lists
mask_invalid_threshold = np.ones(len(tested_thresholds), dtype=bool)
mask_invalid_threshold[valid_thresholds_indices] = False

best_threshold_index = np.where(tested_thresholds == mcc.best_predict_param[0])[0][0]

probas_test = clf.predict_proba(X_test)
proba_positive = np.column_stack([p[:, 1] for p in probas_test])

y_pred_naive = (proba_positive >= tested_thresholds[naive_threshold_index]).astype(int)
precision_naive_threshold = precision_score(
    y_test, y_pred_naive, average=None, zero_division=0
).mean()

y_pred_ltt_best_threshold = (
    proba_positive >= tested_thresholds[best_threshold_index]
).astype(int)
precision_best_ltt_threshold = precision_score(
    y_test, y_pred_ltt_best_threshold, average=None, zero_division=0
).mean()

plt.figure()
plt.scatter(
    tested_thresholds[valid_thresholds_indices],
    precisions[valid_thresholds_indices],
    c="tab:green",
    label="Valid thresholds",
)
plt.scatter(
    tested_thresholds[mask_invalid_threshold],
    precisions[mask_invalid_threshold],
    c="tab:red",
    label="Invalid thresholds",
)
plt.scatter(
    tested_thresholds[best_threshold_index],
    precisions[best_threshold_index],
    c="tab:green",
    label="Best threshold",
    marker="*",
    edgecolors="k",
    s=300,
)
plt.scatter(
    tested_thresholds[naive_threshold_index],
    precisions[naive_threshold_index],
    c="tab:red",
    label="Naive threshold",
    marker="*",
    edgecolors="k",
    s=300,
)
plt.axhline(target_precision, color="tab:gray", linestyle="--")
plt.text(
    0.65,
    target_precision + 0.02,
    "Target precision",
    color="tab:gray",
    fontstyle="italic",
)
plt.xlabel("Threshold")
plt.ylabel("Precision")
plt.legend()
plt.show()


print(
    "With the naive threshold, the precision is:\n "
    f"- {precisions[naive_threshold_index]:.3f} on the calibration set\n "
    f"- {precision_naive_threshold:.3f} on the test set."
)

print(
    "\n\nWith risk control, the precision is:\n "
    f"- {precisions[best_threshold_index]:.3f} on the calibration set\n "
    f"- {precision_best_ltt_threshold:.3f} on the test set."
)

##############################################################################
# The naive threshold is selected on the calibration set to match the target
# precision, but it does not provide any statistical guarantee on unseen data.
#
# In contrast, the threshold selected by risk control takes into account the
# uncertainty due to the finite calibration sample size and guarantees that
# the target precision is met on unseen data with high probability.
#
# As illustrated above, not all thresholds achieving a precision higher than
# the target are statistically valid. This highlights the importance of risk
# control when deploying multi-label classifiers in practice.
#
