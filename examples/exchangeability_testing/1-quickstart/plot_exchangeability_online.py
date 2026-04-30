"""
Exchangeability testing on an online stream
===========================================

This quickstart demonstrates how to test exchangeability on a labeled online
stream, i.e. data arriving sequentially in batches.

Guarantees provided by conformal prediction and risk control depend on the
hypothesis that future data is exchangeable with the data used for calibration
and monitoring. Verifying exchangeability before and during deployment is
therefore important.

Note that only labeled samples can be used to test exchangeability. In
practice, a fraction of the online stream can be labeled from time to time to
run the test and assess performance.
"""

##############################################################################
# Prepare the online stream
# -------------------------
#
# We first prepare an exchangeable online stream. The stream is processed
# batch by batch, as new labeled data would arrive in deployment.

from mapie._example_utils import generate_gaussian_stream, plot_dataset
from mapie.exchangeability_testing import OnlineExchangeabilityTest

random_state = 42
batch_size = 20

X_online, y_online = generate_gaussian_stream(
    shift_type="stable",
    random_state=random_state,
)

plot_dataset(
    X_online,
    y_online,
    title="Exchangeable online stream",
)

##############################################################################
# Run the exchangeability test
# ----------------------------
#
# Now we can test exchangeability on the online stream.
# The test is updated batch by batch as new labels become available.

online_test = OnlineExchangeabilityTest()
for start in range(0, len(X_online), batch_size):
    stop = start + batch_size
    online_test.update(X_online[start:stop], y_online[start:stop])

print("Is the online stream exchangeable?")
for test_name, is_exchangeable in online_test.is_exchangeable.items():
    print(f"{test_name}: {is_exchangeable}")

##############################################################################
# Interpret the result
# --------------------
#
# The online stream is exchangeable. We can confidently continue monitoring
# future data with MAPIE's online methods.

##############################################################################
# Create a non-exchangeable stream
# --------------------------------
#
# Now let us see what happens for a non-exchangeable online stream.
# Here, an abrupt shift happens in the second part of the stream.

prop_shift = 0.5
X_online_abrupt, y_online_abrupt = generate_gaussian_stream(
    shift_type="abrupt",
    prop_shift=prop_shift,
    random_state=random_state + 1,
)
shift_start_abrupt = int(len(y_online_abrupt) * (1 - prop_shift))
plot_dataset(
    X_online_abrupt,
    y_online_abrupt,
    title="Non-exchangeable online stream",
    shift_start=shift_start_abrupt,
)

online_test_abrupt = OnlineExchangeabilityTest()
for start in range(0, len(X_online_abrupt), batch_size):
    stop = start + batch_size
    online_test_abrupt.update(
        X_online_abrupt[start:stop],
        y_online_abrupt[start:stop],
    )

print("Is the shifted online stream exchangeable?")
for test_name, is_exchangeable in online_test_abrupt.is_exchangeable.items():
    print(f"{test_name}: {is_exchangeable}")

##############################################################################
# Interpret the shifted stream
# ----------------------------
#
# The shifted online stream is not exchangeable: MAPIE cannot provide
# statistical guarantees on future data from this stream, and the underlying
# predictive model should not be trusted without further investigation.
#
# Note that the jumper martingale fails to detect the non-exchangeability in
# this case. Itmostly reacts to *one-sided* p-value distortions
# (many consistently small p-values, or many consistently large ones).
# The shift creates both many very low and many very high p-values,
# the effects cancel out for the jumper martingale.
# More generally, this illustrates that no single test is perfect,
# and that it is important to use multiple tests to get a complete picture.
