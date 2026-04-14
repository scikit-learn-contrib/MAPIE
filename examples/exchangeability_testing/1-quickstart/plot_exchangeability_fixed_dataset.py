"""
# Quickstart: exchangeability testing on a fixed dataset

This quickstart uses `RiskMonitoring` in an **offline setting** where data is
already available as a fixed dataset (not received batch-by-batch online).

It compares exchangeability testing methods on two fixed-dataset cases:

1. **Exchangeable fixed dataset** (no shift),
2. **Non-exchangeable fixed dataset** with an abrupt shift.

Currently, only `RiskMonitoring` is shown.
"""

from sklearn.linear_model import LogisticRegression

from mapie.exchangeability_testing import RiskMonitoring
from utils import generate_gaussian_stream, sample_two_gaussians

##############################################################################
# We first fit a classifier on reference training data. Then, in the same
# workflow, we estimate the monitoring threshold on a reference test set and
# update the monitor on a stable online stream. Here, `risk="accuracy"` means
# that `RiskMonitoring` tracks the misclassification risk `1 - accuracy`.

random_state = 42
prop_shift = 0.5
method_name = "RiskMonitoring"

X_train, y_train = sample_two_gaussians(random_state=random_state)
X_reference, y_reference = sample_two_gaussians(random_state=random_state + 1)

clf = LogisticRegression(random_state=random_state)
clf.fit(X_train, y_train)

X_fixed_no_shift, y_fixed_no_shift = generate_gaussian_stream(
    shift_type="stable",
    prop_shift=prop_shift,
    random_state=random_state + 2,
)

monitor_no_shift = RiskMonitoring(risk="accuracy")
monitor_no_shift.compute_threshold(y_reference, clf.predict(X_reference))
threshold = monitor_no_shift.threshold

print(
    f"Reference upper bound on the misclassification risk: {monitor_no_shift.reference_risk_upper_bound:.3f}"
)
print(f"Monitoring threshold: {threshold:.3f}")

# Offline/fixed-dataset usage: update in one call with the full dataset.
y_pred_no_shift = clf.predict(X_fixed_no_shift)
monitor_no_shift.update_online_risk(y_fixed_no_shift, y_pred_no_shift)

is_exchangeable_no_shift = not monitor_no_shift.harmful_shift_detected

##############################################################################
# Non-exchangeable fixed dataset: abrupt shift in the second part.

X_fixed_abrupt, y_fixed_abrupt = generate_gaussian_stream(
    shift_type="abrupt",
    prop_shift=prop_shift,
    random_state=random_state + 3,
)

monitor_abrupt = RiskMonitoring(risk="accuracy", threshold=threshold)
y_pred_abrupt = clf.predict(X_fixed_abrupt)
monitor_abrupt.update_online_risk(y_fixed_abrupt, y_pred_abrupt)

is_exchangeable_abrupt = not monitor_abrupt.harmful_shift_detected

print("\nExchangeability summary (fixed dataset setting):")
print(f"- Method: {method_name}")
print(f"- Exchangeable fixed dataset: is_exchangeable={is_exchangeable_no_shift}")
print(f"- Abrupt-shift fixed dataset: is_exchangeable={is_exchangeable_abrupt}")
