from numpy.typing import NDArray
from pyparsing import Optional
from mapie.conformity_scores import BaseConformityScores

# Test level
alpha = 0.05

# The class could be adapted for Martingale test and Risk monitoring test.
# The following framing is more close to the Martingale test.
class ExchangeabilityTestOnlineDataset:
    def __init__(self, test_method, confidence_level, base_conformity_scores=None, warn=True):
        self.test_method = test_method
        self.confidence_level = confidence_level
        self.base_conformity_scores = base_conformity_scores,
        self.warn = warn
        
        # Initialize other necessary attributes for the test
    
    @property
    def is_exchangeable(self):
        # Return True if exchangeable : case last_stat_value < alpha
        # Return False if not exchangeable : case last_stat_value > 1/alpha
        # Return None if not enough data to determine exchangeability : case less data 200 data points and last_stat_value between alpha and 1/alpha
        pass

    def update(self, y_true: NDArray, y_pred: NDArray, X: Optional[NDArray] = None):
        # Update the test with the new point (y_true, y_pred, X)
        # Compute conformity scores and update the test state

        # 1. compute conformity score for the new point
        conformity_score = self.base_conformity_scores(y_true, y_pred, X)
        # 2. Save the conformity score for the new point for future updates
        self.base_conformity_scores.append(conformity_score)
        
        # 3. update the test state based on the new conformity score and the previous scores
        # This step include martingale update for the martingale test method, or other updates for other test methods
        # Update the "is_exchangeable" property

        # 4. Check if the test has rejected the null hypothesis of exchangeability and update the test state accordingly
        # Raise warning if the null hypothesis of exchangeability has been rejected and warn is True

        # 5. Return self
        return self

    def summary(self):
        # Return a summary table of the test statistic
        # min, quantile 0.025, quantile 0.25, quantile 0.5, mean, quantile 0.75, quantile 0.975, max
        # Test level alpha
        # Number of data points used in the test
        # Stoping time of the test
        # The point at which the test rejected the null hypothesis of exchangeability (if applicable)
        # The depthest point in the online dataset (if applicable)
        # Any other relevant information about the test state
        pass

# Initialize the test object
etod = ExchangeabilityTestOnlineDataset(
    test_method="martingale",
    confidence_level=1 - alpha,
    base_conformity_scores=BaseConformityScores
)

# Initialize the test through update method before online
etod.update(y_test, y_pred, X_test) # X_test is optional, only used for some test methods

X_online_all = []
y_pred_all = []

def get_next_features():
    # Get the next features for the online data point
    # This function should return the features of the next online data point as a numpy array
    pass

while True:
    # Get the next test point (y_true, y_pred, X)
    x_online = get_next_features()
    y_pred = clf.predict_proba(x_online.reshape(1, -1))[0, 1]  # Get the predicted probability for the positive class
    X_online_all.append(x_online)
    y_pred_all.append(y_pred)

# get some labels
X_online_test = [x_online_1, x_online_8, ...]
y_pred_test = [y_pred_1, y_pred_8, ...]  # Get the predicted probabilities for the online data point(s)
y_true = get_next_label(X_online_test)  # Get the true label for the online data point(s)
# Update the test with the new point
etod.update(y_true, y_pred_test, X_online_test)  # X_online_test is optional, only used for some test methods
# Check if the test has rejected the null hypothesis


class OnlineExchangeabilityTesting:

    def __init__(self,):


    
    def