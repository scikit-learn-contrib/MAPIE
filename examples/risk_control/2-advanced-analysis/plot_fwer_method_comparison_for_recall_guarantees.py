"""
============================================================================
Comparison of FWER control methods for risk control in binary classification
============================================================================

This example illustrates how different family-wise error rate (FWER) control strategies impact
the set of statistically valid thresholds when controlling a risk in binary classification.

The risk control involves chosing a threshold on the predicted probabilities of the positive class
to ensure that acertain risk (e.g., 1-precision, 1-recall) is controlled with high probability
on unseen data. The procedure relies on a calibration set to estimate the risk at different
thresholds and to select thresholds that are valid with high probability on unseen data.
The FWER control strategy determines how the multiple comparisons problem is handled
when testing multiple thresholds.

In this example, we compare the Bonferroni correction, the Fixed-Sequence Testing (FST) procedure,
and the Holm-Bonferroni method.

- Bonferroni, a classical simultaneous correction that is valid under any dependence structure but can be conservative.
- Fixed-Sequence Testing, which exploits monotonicity when available to gain power but requires a pre-specified order of testing.
- Bonferroni-Holm, a sequentially rejective method that is more powerful than Bonferroni while still controlling the FWER under any dependence structure.

Note that the Fixed-Sequence Testing procedure can only be applied when the risk
is monotonic with respect to the threshold, which is the case for instance for 1-recall but not for 1-precision.
The table below summarizes the applicability of each method depending on the properties of the risks and parameters.

+-------------------------------+---------------------------+--------------------+-------------------------+
| **FWER method **              | **Only monotonic risks**  | **Multiple risks** | **Multiple parameters** |
+-------------------------------+---------------------------+--------------------+-------------------------+
| **Bonferroni correction**     |             ❌            |         ✅         |            ✅           |
+-------------------------------+---------------------------+--------------------+-------------------------+
| **Fixed-Sequence Testing**    |             ✅            |         ❌         |            ❌           |
+-------------------------------+---------------------------+--------------------+-------------------------+
| **Bonferroni-Holm**           |             ❌            |         ✅         |            ✅           |
+-------------------------------+---------------------------+--------------------+-------------------------+


In this example, we compare the three methods for controlling the FWER when controlling the 1-recall.
Using the same classifier, calibration set, test set, and target recall, we:

- Compute the set of valid thresholds for each method.
- Compare the valid thresholds and the best valid threshold for each method
- Visualize the valid and invalid thresholds for each method and the best valid threshold for each method.

"""

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
# Next, we initialize a :class:`~mapie.risk_control.BinaryClassificationController`
# using the probability estimation function from the fitted estimator:
# ``clf.predict_proba``, a risk or performance metric (here, "recall"),
# a target risk level, and a confidence level. Then we use the calibration data
# to compute statistically guaranteed thresholds using a risk control method,
# and a FWER control strategy passed through the ``fwer_method`` parameter.
#
# We compare three FWER control strategies: the Bonferroni correction,
# the FST procedure, and the Holm-Bonferroni method.
#
# Note that the FTS procedure can only be applied when the risk or performance metric
# is monotonic with respect to the threshold. Here "recall" is monotonic with respect
# to the threshold. However, "precision" is not monotonic with respect to the threshold,
# therefore the FST procedure cannot be applied when controlling the 1-precision.
#

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
    f"- Holm-Bonferroni method: {len(bcc_bonferroni_holm.valid_predict_params)} valid thresholds. The best threshold maximizing precision is: {bcc_bonferroni_holm.best_predict_param:.3f}\n"
)


#################################################################################
# In the plot below, we visualize how the threshold values impact recall, and what
# thresholds have been computed as statistically guaranteed for each method.

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
    label="Valid Holm only",
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
    label="Best Holm",
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
# Finally, we compare the recall on the test set when using the naive threshold,
# the best valid threshold for each method, and the corresponding recall on the calibration set.

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
# The risk control procedure provides guarantees on unseen data, which is not the
# case for the naive threshold selection. In this example, the naive threshold results
# in a recall on the test set that satisfies the target recall but does not satisfy the
# confidence level, which means that the guarantee is not satisfied. On the other hand,
# the thresholds selected by risk control for each method satisfy the target recall on
# the test set with a confidence level of 0.7, which means that the guarantee is satisfied.
# The Bonferroni correction is the most conservative method, resulting in the smallest set
# of valid thresholds and the lowest best threshold, while the FST procedure is the least
# conservative, resulting in the largest set of valid thresholds and the highest best threshold.
# The Holm-Bonferroni method is in between, providing a larger set of valid thresholds than
# Bonferroni but smaller than FST, and a best threshold that is higher than Bonferroni but
# lower than FST.
