"""
# Online exchangeability testing with RiskMonitoring

This quickstart compares exchangeability testing methods on two online cases:

1. **Exchangeable stream** (no harmful shift expected),
2. **Non-exchangeable stream** with an **abrupt shift** (harmful shift expected).

Currently, only `RiskMonitoring` is shown.
"""

from sklearn.linear_model import LogisticRegression
from utils import generate_gaussian_stream

from mapie.exchangeability_testing import OnlineExchangeabilityTest

##############################################################################
# We first fit a classifier on reference training data. Then, in the same
# workflow, we estimate the monitoring threshold on a reference test set and
# update the monitor on a stable online stream. Here, `risk="accuracy"` means
# that `RiskMonitoring` tracks the misclassification risk `1 - accuracy`.

random_state = 42
batch_size = 25
prop_shift = 0.5

X_train, y_train = generate_gaussian_stream(
    shift_type="stable",
    random_state=random_state,
)
X_reference, y_reference = generate_gaussian_stream(
    shift_type="stable",
    random_state=random_state + 1,
)

clf = LogisticRegression(random_state=random_state)
clf.fit(X_train, y_train)
y_pred_reference = clf.predict(X_reference)

X_online_no_shift, y_online_no_shift = generate_gaussian_stream(
    shift_type="stable",
    prop_shift=prop_shift,
    random_state=random_state + 2,
)

online_test_no_shift = OnlineExchangeabilityTest(
    method_names="all",
    method_params={
        "risk_monitoring": {
            "risk": "accuracy",
            "reference_data": (y_reference, y_pred_reference),
        }
    },
)
threshold = online_test_no_shift.test_methods[0].threshold

print(
    "Reference upper bound on the misclassification risk: "
    f"{online_test_no_shift.test_methods[0].reference_risk_upper_bound:.3f}"
)
print(f"Monitoring threshold: {threshold:.3f}")

for start in range(0, len(X_online_no_shift), batch_size):
    stop = start + batch_size
    y_pred_batch = clf.predict(X_online_no_shift[start:stop])
    online_test_no_shift.update(y_online_no_shift[start:stop], y_pred_batch)

is_exchangeable_no_shift = online_test_no_shift.is_exchangeable["risk_monitoring"]

##############################################################################
# Non-exchangeable case: abrupt distribution shift in the stream.

X_online_abrupt, y_online_abrupt = generate_gaussian_stream(
    shift_type="abrupt",
    prop_shift=prop_shift,
    random_state=random_state + 3,
)

online_test_abrupt = OnlineExchangeabilityTest(
    method_names="all",
    method_params={
        "risk_monitoring": {
            "risk": "accuracy",
            "reference_data": (y_reference, y_pred_reference),
        }
    },
)
for start in range(0, len(X_online_abrupt), batch_size):
    stop = start + batch_size
    y_pred_batch = clf.predict(X_online_abrupt[start:stop])
    online_test_abrupt.update(y_online_abrupt[start:stop], y_pred_batch)

is_exchangeable_abrupt = online_test_abrupt.is_exchangeable["risk_monitoring"]

print("\nExchangeability summary (online setting):")
print(f"- Exchangeable stream: is_exchangeable={is_exchangeable_no_shift}")
print(f"- Abrupt-shift stream: is_exchangeable={is_exchangeable_abrupt}")
