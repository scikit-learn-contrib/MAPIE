"""
# Online exchangeability testing

This quickstart demonstrates how to test exchangeability on a labeled online
stream. Note that only labeled samples can be used to test exchangeability.
In practice, a sample of online data can be labeled from time to time in
order to test exchangeability and assess performance.

Guarantees provided by conformal prediction and risk control depend on the
hypothesis that future data is exchangeable with the data used for calibration
and monitoring. This is why verifying exchangeability before applying methods
from MAPIE is important.

Here, we process the stream batch by batch and update the online
exchangeability test as new labeled data arrives.
"""

##############################################################################
# We first prepare an exchangeable online stream.

from utils import generate_gaussian_stream, plot_dataset

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
# The test is updated batch by batch as labels become available.

online_test = OnlineExchangeabilityTest()
for start in range(0, len(X_online), batch_size):
    stop = start + batch_size
    online_test.update(X_online[start:stop], y_online[start:stop])

print(online_test.is_exchangeable)

##############################################################################
# The online stream is exchangeable. We can continue monitoring future data
# with MAPIE's online methods.

##############################################################################
# Non-exchangeable online stream: abrupt shift in the second part.

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

print(online_test_abrupt.is_exchangeable)

##############################################################################
# The online stream is not exchangeable anymore. MAPIE cannot provide
# statistical guarantees on future data from this shifted stream.
