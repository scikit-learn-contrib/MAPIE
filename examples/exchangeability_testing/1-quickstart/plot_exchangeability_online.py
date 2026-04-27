"""
# Exchangeability testing on an online stream

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
# We first prepare an exchangeable online stream. The stream is processed
# batch by batch, as new labeled data would arrive in deployment.

from examples.exchangeability_testing.utils import generate_gaussian_stream, plot_dataset

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
# Now we can test exchangeability on the online stream.
# The test is updated batch by batch as new labels become available.

online_test = OnlineExchangeabilityTest()
for start in range(0, len(X_online), batch_size):
    stop = start + batch_size
    online_test.update(X_online[start:stop], y_online[start:stop])

print(f"Is the online stream exchangeable? {online_test.is_exchangeable}")

##############################################################################
# The online stream is exchangeable. We can confidently continue monitoring
# future data with MAPIE's online methods.

##############################################################################
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

print(
    f"Is the shifted online stream exchangeable? {online_test_abrupt.is_exchangeable}"
)

##############################################################################
# The shifted online stream is not exchangeable: MAPIE cannot provide
# statistical guarantees on future data from this stream, and the underlying
# predictive model should not be trusted without further investigation.
