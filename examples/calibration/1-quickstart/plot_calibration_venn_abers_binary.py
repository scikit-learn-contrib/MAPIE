"""
=================================================
Calibrating binary classifier with Venn-ABERS
=================================================
This example shows how to calibrate a binary classifier with
:class:`~mapie.calibration.VennAbersCalibrator` and visualize the
impact on predicted probabilities.

We compare an uncalibrated model to its Venn-ABERS calibrated version
using reliability diagrams and Brier scores.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from mapie.calibration import VennAbersCalibrator

####################################################################
# 1. Build a miscalibrated binary classifier
# ---------------------------------------------------
# We generate a toy binary dataset and fit a random forest model
# which is known to be miscalibrated out of the box (produces
# probabilities too close to 0 or 1). We use a larger dataset to
# ensure sufficient data for proper calibration.

from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    class_sep=0.8,
    random_state=42,
)

# Split into train, calibration, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
)

# Use Random Forest which tends to be miscalibrated
base_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42
)
base_model.fit(X_train, y_train)
probs_raw = base_model.predict_proba(X_test)[:, 1]
raw_brier = brier_score_loss(y_test, probs_raw)

####################################################################
# 2. Calibrate with Venn-ABERS
# ----------------------------
# We wrap the same base model in :class:`~mapie.calibration.VennAbersCalibrator`
# using the inductive mode (default). The calibrator uses the calibration set
# to learn a calibration mapping that will improve probability estimates.

va_calibrator = VennAbersCalibrator(
    estimator=RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    inductive=True,
    random_state=42,
)
va_calibrator.fit(X_train, y_train, X_calib=X_calib, y_calib=y_calib)
probs_va = va_calibrator.predict_proba(X_test)[:, 1]
va_brier = brier_score_loss(y_test, probs_va)

####################################################################
# 3. Reliability diagrams and Brier scores
# ----------------------------------------
# Reliability diagrams show how predicted probabilities compare to
# observed frequencies. Perfect calibration lies on the diagonal.
# We also display Brier scores to quantify the improvement.

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
CalibrationDisplay.from_predictions(
    y_test,
    probs_raw,
    name=f"Uncalibrated (Brier={raw_brier:.3f})",
    n_bins=10,
    ax=axes[0],
)
CalibrationDisplay.from_predictions(
    y_test,
    probs_va,
    name=f"Venn-ABERS (Brier={va_brier:.3f})",
    n_bins=10,
    ax=axes[1],
)
axes[0].set_title("Before calibration")
axes[1].set_title("After Venn-ABERS calibration")
plt.tight_layout()
plt.show()
