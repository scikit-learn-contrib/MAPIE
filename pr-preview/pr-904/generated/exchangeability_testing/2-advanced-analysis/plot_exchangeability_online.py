"""
# Quickstart: online exchangeability testing with RiskMonitoring

This quickstart compares exchangeability testing methods on two online cases:

1. **Exchangeable stream** (no harmful shift expected),
2. **Non-exchangeable stream** with an **abrupt shift** (harmful shift expected).

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
batch_size = 25
prop_shift = 0.5
method_name = "RiskMonitoring"

X_train, y_train = sample_two_gaussians(random_state=random_state)
X_reference, y_reference = sample_two_gaussians(random_state=random_state + 1)

clf = LogisticRegression(random_state=random_state)
clf.fit(X_train, y_train)

X_online_no_shift, y_online_no_shift = generate_gaussian_stream(
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

for start in range(0, len(X_online_no_shift), batch_size):
    stop = start + batch_size
    y_pred_batch = clf.predict(X_online_no_shift[start:stop])
    monitor_no_shift.update_online_risk(y_online_no_shift[start:stop], y_pred_batch)

is_exchangeable_no_shift = not monitor_no_shift.harmful_shift_detected

##############################################################################
# Non-exchangeable case: abrupt distribution shift in the stream.

X_online_abrupt, y_online_abrupt = generate_gaussian_stream(
    shift_type="abrupt",
    prop_shift=prop_shift,
    random_state=random_state + 3,
)

monitor_abrupt = RiskMonitoring(risk="accuracy", threshold=threshold)
for start in range(0, len(X_online_abrupt), batch_size):
    stop = start + batch_size
    y_pred_batch = clf.predict(X_online_abrupt[start:stop])
    monitor_abrupt.update_online_risk(y_online_abrupt[start:stop], y_pred_batch)

is_exchangeable_abrupt = not monitor_abrupt.harmful_shift_detected

print("\nExchangeability summary (online setting):")
print(f"- Method: {method_name}")
print(f"- Exchangeable stream: is_exchangeable={is_exchangeable_no_shift}")
print(f"- Abrupt-shift stream: is_exchangeable={is_exchangeable_abrupt}")
