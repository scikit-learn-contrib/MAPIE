"""
====================================================
Calibrating multi-class classifier with Venn-ABERS
====================================================
This example shows how to calibrate a multi-class classifier with
:class:`~mapie.calibration.VennAbersCalibrator` and visualize the
impact on predicted probabilities. We compare an uncalibrated model
against its Venn-ABERS calibrated version using reliability diagrams
and multi-class Brier scores.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from mapie.calibration import VennAbersCalibrator

####################################################################
# 1. Build a miscalibrated multi-class classifier
# -----------------------------------------------
# We generate a 3-class dataset and fit a gradient boosting model,
# which is often miscalibrated out of the box.

X, y = make_classification(
    n_samples=2500,
    n_features=20,
    n_informative=12,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=7,
)

classes = np.unique(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7, stratify=y
)

base_model = GradientBoostingClassifier(random_state=7)
base_model.fit(X_train, y_train)
probs_raw = base_model.predict_proba(X_test)

####################################################################
# 2. Calibrate with Venn-ABERS
# ----------------------------
# The calibrator refits the base model internally and learns a mapping
# from a held-out calibration subset. Venn-ABERS natively supports
# multi-class problems.

va_calibrator = VennAbersCalibrator(
    estimator=GradientBoostingClassifier(random_state=7),
    inductive=True,
    random_state=7,
)
va_calibrator.fit(X_train, y_train)
probs_va = va_calibrator.predict_proba(X_test)

####################################################################
# 3. Multi-class Brier score helper
# ---------------------------------
# We compute the mean squared error between predicted probabilities and
# one-hot encoded labels.


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_onehot = label_binarize(y_true, classes=classes)
    return float(np.mean(np.sum((y_onehot - proba) ** 2, axis=1)))


brier_raw = multiclass_brier(y_test, probs_raw)
brier_va = multiclass_brier(y_test, probs_va)

####################################################################
# 4. Reliability diagrams and Brier scores
# ----------------------------------------
# We plot one-vs-rest reliability curves for each class before and after
# calibration. Lower Brier score indicates better calibration.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for cls in classes:
    y_true_cls = (y_test == cls).astype(int)
    prob_raw_cls = probs_raw[:, cls]
    prob_va_cls = probs_va[:, cls]

    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_true_cls, prob_raw_cls, n_bins=10, strategy="uniform"
    )
    frac_pos_va, mean_pred_va = calibration_curve(
        y_true_cls, prob_va_cls, n_bins=10, strategy="uniform"
    )

    axes[0].plot(mean_pred_raw, frac_pos_raw, marker="o", label=f"class {cls}")
    axes[1].plot(mean_pred_va, frac_pos_va, marker="o", label=f"class {cls}")

for ax, title in zip(
    axes,
    [
        f"Before calibration (Brier={brier_raw:.3f})",
        f"After Venn-ABERS (Brier={brier_va:.3f})",
    ],
):
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
