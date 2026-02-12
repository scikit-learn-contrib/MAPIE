"""
============================================================================
Comparison of FWER control methods for risk control in binary classification
============================================================================

This example compares how different family-wise error rate (FWER) control
strategies affect the set of statistically valid thresholds when controlling
a risk in binary classification.

Risk control consists in selecting a threshold on predicted probabilities
so that a chosen risk (e.g., 1-recall) is guaranteed to stay below a target
level with high probability on unseen data. The guarantee is obtained using
a calibration dataset and a multiple testing correction across candidate
thresholds.

We compare three FWER procedures:

- ``"bonferroni"``: a classical correction valid under any dependence
  structure but often conservative.
- ``"fst_ascending"``: Fixed-Sequence Testing (FST), which exploits monotonicity
  of the risk to gain power but requires a single monotonic risk.
- ``"bonferroni_holm"``: a sequentially rejective procedure that is less
  conservative than Bonferroni while remaining generally valid.

The applicability of each method depends on the problem structure:

+----------------------------+------------------+---------------+------------------+
| **Method**                 | Monotonic risk   | Multi-risk    | Multi-parameter  |
+----------------------------+------------------+---------------+------------------+
| Bonferroni                 | ✅               | ✅             | ✅              |
| FST                        | required         | ❌             | ❌              |
| Holm                       | ✅               | ✅             | ✅              |
+----------------------------+------------------+---------------+------------------+

Here we control **1-recall**, which is monotonic with respect to the decision
threshold. We therefore expect FST to be the least conservative, Bonferroni
the most conservative, and Bonferroni-Holm to lie in between.

Using the same classifier, calibration set, and target recall, we:

- compute valid thresholds for each method,
- compare their selected best thresholds,
- visualize agreement and differences between procedures.
"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import recall_score
from sklearn.neural_network import MLPClassifier

from mapie.risk_control import BinaryClassificationController
from mapie.utils import train_conformalize_test_split

RANDOM_STATE = 42

##############################################################################
# First, load the dataset and then split it into training, calibration
# (for conformalization), and test sets.

X, y = make_circles(n_samples=5000, noise=0.3, factor=0.3, random_state=RANDOM_STATE)
(X_train, X_calib, X_test, y_train, y_calib, y_test) = train_conformalize_test_split(
    X,
    y,
    train_size=0.7,
    conformalize_size=0.1,
    test_size=0.2,
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
# Next, we initialize :class:`~mapie.risk_control.BinaryClassificationController`
# with the estimator probability function (``clf.predict_proba``), a risk metric
# (here ``"recall"``), a target level, and a confidence level. We then calibrate
# it to compute thresholds that are statistically guaranteed to satisfy the
# target risk on unseen data.
#
# We compare three FWER control strategies via the ``fwer_method`` parameter:
#
# - ``"bonferroni"``: universally valid but conservative,
# - ``"fst_ascending"``: more powerful when the risk is monotonic,
# - ``"bonferroni_holm"``: sequential method balancing validity and power.
#
# The FST procedure requires the risk to be monotonic with respect to the
# threshold. This holds for recall but not for precision, which is generally
# non-monotonic; therefore FST cannot be used for controlling 1-precision.
##############################################################################

target_recall = 0.8
confidence_level = 0.7
bcc_bonferroni = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk="recall",
    target_level=target_recall,
    confidence_level=confidence_level,
    list_predict_params=np.linspace(0.01, 0.99, 100),
    fwer_method="bonferroni",
)

bcc_fst = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk="recall",
    target_level=target_recall,
    confidence_level=confidence_level,
    list_predict_params=np.linspace(0.01, 0.99, 100),
    fwer_method="fst_ascending",
)

bcc_bonferroni_holm = BinaryClassificationController(
    predict_function=clf.predict_proba,
    risk="recall",
    target_level=target_recall,
    confidence_level=confidence_level,
    list_predict_params=np.linspace(0.01, 0.99, 100),
    fwer_method="bonferroni_holm",
)

bcc_bonferroni.calibrate(X_calib, y_calib)
bcc_fst.calibrate(X_calib, y_calib)
bcc_bonferroni_holm.calibrate(X_calib, y_calib)

print(
    f"Thresholds found that guarantee a recall of at least {target_recall} with a confidence of {confidence_level}:\n"
    f"- Bonferroni correction: {len(bcc_bonferroni.valid_predict_params)} valid thresholds. The best threshold maximizing precision is: {bcc_bonferroni.best_predict_param:.3f}\n"
    f"- FST procedure: {len(bcc_fst.valid_predict_params)} valid thresholds. The best threshold maximizing precision is: {bcc_fst.best_predict_param:.3f}\n"
    f"- Bonferroni-Holm method: {len(bcc_bonferroni_holm.valid_predict_params)} valid thresholds. The best threshold maximizing precision is: {bcc_bonferroni_holm.best_predict_param:.3f}\n"
)


#################################################################################
# The plot below shows how recall varies with the decision threshold and which
# thresholds are statistically valid under each FWER control method. Colors
# indicate agreement or disagreement between methods, and stars mark the best
# valid threshold selected by each procedure as well as the naive threshold.
#################################################################################

proba_positive_class = clf.predict_proba(X_calib)[:, 1]

tested_thresholds = bcc_bonferroni._predict_params
recalls = np.full(len(tested_thresholds), np.inf)
for i, threshold in enumerate(tested_thresholds):
    y_pred = (proba_positive_class >= threshold).astype(int)
    recalls[i] = recall_score(y_calib, y_pred)

naive_threshold_index = np.argmin(
    np.where(recalls >= target_recall, recalls - target_recall, np.inf)
)

valid_index_bonferroni = np.array(
    [t in bcc_bonferroni.valid_predict_params for t in tested_thresholds]
)
valid_index_fst = np.array(
    [t in bcc_fst.valid_predict_params for t in tested_thresholds]
)
valid_index_bonferroni_holm = np.array(
    [t in bcc_bonferroni_holm.valid_predict_params for t in tested_thresholds]
)
best_thr_index_bonferroni = np.where(
    tested_thresholds == bcc_bonferroni.best_predict_param
)[0][0]
best_thr_index_fst = np.where(tested_thresholds == bcc_fst.best_predict_param)[0][0]
best_thr_index_bonferroni_holm = np.where(
    tested_thresholds == bcc_bonferroni_holm.best_predict_param
)[0][0]

# plotting the valid and invalid thresholds for each method, and the best valid threshold for each method
valid_all = valid_index_bonferroni & valid_index_fst & valid_index_bonferroni_holm
invalid_all = ~valid_index_bonferroni & ~valid_index_fst & ~valid_index_bonferroni_holm

only_bonf = valid_index_bonferroni & ~valid_all
only_fst = valid_index_fst & ~valid_all
only_holm = valid_index_bonferroni_holm & ~valid_all

plt.figure()

plt.scatter(
    tested_thresholds[invalid_all],
    recalls[invalid_all],
    c="tab:red",
    label="Invalid all methods",
)
plt.scatter(
    tested_thresholds[valid_all],
    recalls[valid_all],
    c="tab:green",
    label="Valid all methods",
)
plt.scatter(
    tested_thresholds[only_bonf],
    recalls[only_bonf],
    c="lime",
    label="Valid Bonferroni only",
)
plt.scatter(
    tested_thresholds[only_fst],
    recalls[only_fst],
    c="teal",
    label="Valid FST only",
)
plt.scatter(
    tested_thresholds[only_holm],
    recalls[only_holm],
    c="olive",
    label="Valid Bonferroni-Holm only",
)
plt.scatter(
    tested_thresholds[best_thr_index_bonferroni],
    recalls[best_thr_index_bonferroni],
    c="lime",
    marker="*",
    edgecolors="k",
    s=300,
    label="Best Bonferroni",
)
plt.scatter(
    tested_thresholds[best_thr_index_fst],
    recalls[best_thr_index_fst],
    c="teal",
    marker="*",
    edgecolors="k",
    s=300,
    label="Best FST",
)
plt.scatter(
    tested_thresholds[best_thr_index_bonferroni_holm],
    recalls[best_thr_index_bonferroni_holm],
    c="olive",
    marker="*",
    edgecolors="k",
    s=300,
    label="Best Bonferroni-Holm",
)
plt.scatter(
    tested_thresholds[naive_threshold_index],
    recalls[naive_threshold_index],
    c="tab:red",
    marker="*",
    edgecolors="k",
    s=300,
    label="Naive threshold",
)
plt.axhline(target_recall, color="gray", linestyle="--")
plt.text(
    0.7,
    target_recall + 0.02,
    "Target recall",
    color="gray",
    fontstyle="italic",
)

plt.xlabel("Threshold")
plt.ylabel("Recall")
plt.legend()
plt.show()


#################################################################################
# Finally, we compare test-set recall obtained with the naive threshold and with
# the best valid threshold selected by each FWER method, alongside their
# calibration-set recalls.
#################################################################################


proba_positive_class_test = clf.predict_proba(X_test)[:, 1]
y_pred_naive = (
    proba_positive_class_test >= tested_thresholds[naive_threshold_index]
).astype(int)
print(
    "With the naive threshold, the recall is:\n "
    f"- {recalls[naive_threshold_index]:.3f} on the calibration set\n "
    f"- {recall_score(y_test, y_pred_naive):.3f} on the test set."
)

print(
    "\n\nWith Bonferroni correction, the recall is:\n "
    f"- {recalls[best_thr_index_bonferroni]:.3f} on the calibration set\n "
    f"- {recall_score(y_test, bcc_bonferroni.predict(X_test)):.3f} on the test set."
)

print(
    "\n\nWith FST procedure, the recall is:\n "
    f"- {recalls[best_thr_index_fst]:.3f} on the calibration set\n "
    f"- {recall_score(y_test, bcc_fst.predict(X_test)):.3f} on the test set."
)

print(
    "\n\nWith Bonferroni-Holm, the recall is:\n "
    f"- {recalls[best_thr_index_bonferroni_holm]:.3f} on the calibration set\n "
    f"- {recall_score(y_test, bcc_bonferroni_holm.predict(X_test)):.3f} on the test set."
)

################################################################################
# Risk control provides statistical guarantees on unseen data, unlike naive
# threshold selection. Although the naive threshold may meet the target recall
# on this test set, it does not come with a confidence guarantee. In contrast,
# thresholds selected via risk control are valid with the prescribed confidence
# level.
#
# As expected, Bonferroni is the most conservative (fewest valid thresholds),
# FST is the least conservative when its assumptions hold (largest valid set),
# and Bonferroni-Holm lies in between, offering a compromise between power and generality.
################################################################################
