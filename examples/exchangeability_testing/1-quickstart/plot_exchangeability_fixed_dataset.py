"""
# Exchangeability testing on a fixed dataset

This quickstart demonstrates how to test exchangeability on a fixed dataset.

Guarantees provided by conformal prediction and risk control depend on the
hypothesis that data is exchangeable. Verifying exchangeability before
applying methods from MAPIE is therefore important. Typically, (split)
conformal prediction, risk control, and calibration require data not seen
during training, such as a split of the test data.

Note that for the exchangeability test to be valid, the order of samples in
the fixed dataset must be representative of what will happen after deployment.
Shuffling the data beforehand would trivially render the dataset exchangeable
and hide any potential distribution shift.
"""

##############################################################################
# We first prepare the data and fit a classifier on the training data.

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import generate_gaussian_stream, plot_dataset

from mapie.classification import SplitConformalClassifier
from mapie.exchangeability_testing import FixedDatasetExchangeabilityTest

random_state = 42

X, y = generate_gaussian_stream(
    shift_type="stable",
    random_state=random_state,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state, shuffle=False
)

plot_dataset(
    X_test,
    y_test,
    title="Exchangeable fixed dataset",
)

classifier = LogisticRegression(random_state=random_state)
classifier.fit(X_train, y_train)

##############################################################################
# Now we can test the exchangeability of the test dataset.
# By default, we use all available test methods.
# Method-specific parameters can be passed as a dictionary.

exchangeability_test = FixedDatasetExchangeabilityTest()
exchangeability_test.run(X_test, y_test)

print(f"Is the test dataset exchangeable? {exchangeability_test.is_exchangeable}")

##############################################################################
# The test dataset is exchangeable. We can continue with MAPIE.
# Conformalization with a split of the test dataset will provide
# coverage guarantees on the remaining test data.

X_conformalize, X_test_new, y_conformalize, y_test_new = train_test_split(
    X_test, y_test, test_size=0.5, random_state=random_state
)

confidence_level = 0.95
mapie_classifier = SplitConformalClassifier(
    estimator=classifier, confidence_level=confidence_level, prefit=True
)
mapie_classifier.conformalize(X_conformalize, y_conformalize)
y_pred, y_pred_set = mapie_classifier.predict_set(X_test_new)

##############################################################################
# Now let us see what happens for a non-exchangeable fixed dataset.
# Here, an abrupt shift happens in the second part of the dataset.

X_test_abrupt, y_test_abrupt = generate_gaussian_stream(
    n_samples=len(X_test),
    shift_type="abrupt",
    prop_shift=0.5,
    random_state=random_state + 1,
)
shift_start_abrupt = int(len(y_test_abrupt) * 0.5)
plot_dataset(
    X_test_abrupt,
    y_test_abrupt,
    title="Non-exchangeable fixed dataset",
    shift_start=shift_start_abrupt,
)

exchangeability_test = FixedDatasetExchangeabilityTest()
exchangeability_test.run(X_test_abrupt, y_test_abrupt)

print(f"Is the shifted dataset exchangeable? {exchangeability_test.is_exchangeable}")

##############################################################################
# The shifted test dataset is not exchangeable: MAPIE cannot provide
# statistical guarantees on it, and more generally the classifier itself
# should not be trusted without further investigation.
