# Fixed dataset test
# Three test methods families : Permutation test, Martingale test, Risk monitoring test
# - Risk Monotoring: only one way testing
#
# - Permutation test:
#  * Permutation test, see Section 2.2. of "Theoretical Foundations of Conformal Prediction" https://arxiv.org/abs/2411.11824v5
#  * Permutation test using Binomial strategy of "Sequential Monte-Carlo testing by betting" https://arxiv.org/abs/2401.07365v5
#  * Permutation test using Mixture Binomial strategy of "Sequential Monte-Carlo testing by betting" https://arxiv.org/abs/2401.07365v5
#
# - Martingale test:
#  * Jumper Martingale test
#  * Plug-in Martingale test
#

from abc import ABC


class EXchangeabilityTestFixed(ABC):
    def __init__(self, test_method, significance_level=0.05):
        self.test_method = test_method
        self.significance_level = significance_level
        # Initialize other necessary attributes for the test


class ExchangeabilityTestFixedDataset(EXchangeabilityTestFixed):
    def run(self, X_test: NDArray, y_test: NDArray):
        # Run the test on the fixed dataset (X_test, y_test)

        # 1. In the case of Risk Monitoring test and Martingale test,
        # Implement wrapper of ExchangeabilityTestOnlineDataset to run the test on the fixed dataset (X_test, y_test)
        ## Update the test state using the update method of ExchangeabilityTestOnlineDataset
        ## Check if the test has rejected the null hypothesis of exchangeability at each step and update the test state accordingly
        ## Return the summary of the test results using the summary method of ExchangeabilityTestOnlineDataset

        # 2. In the case of Permutation test, implement the test directly on the fixed dataset (X_test, y_test)
        ## Transform the dataset (X_test, y_test) into a suitable non-conformity score preserving the exchangeability property
        ## and preservig the non-exchangeability anomaly if the dataset is not exchangeable (for example, the change point, covariate shilft, time dependency, etc.)
        ##   * Need a model for the non-conformity score:
        ##   ** It is either given: is not self.estimator
        ##   ** Or it is trained on a separate training set, e.g. 20% first data points of the dataset,
        ##   ** and then applied to the remaining 80% data points to compute the non-conformity scores for the test
        ## Compute the test statistic for the non-conformity scores of the original dataset (X_test, y_test)
        ## Generate a number of permuted datasets by shuffling non-conformity scores
        ## Compute the p-value as the proportion of permuted non-conformity score test statistics that are as extreme as or more extreme than the test statistic for the original dataset
        ## Compare the p-value to the significance level to determine if the null hypothesis of exchangeability is rejected or not
        ## Return the summary of the test results
        pass


class ExchangeabilityTestFixedNonConformityScore(EXchangeabilityTestFixed):
    def run(self, score_values: NDArray):
        # Compute test on non-conformity scores directly, without the need of the original dataset (X_test, y_test)
        # 1. In the case of Risk Monitoring test and Martingale test,
        # Implement wrapper of ExchangeabilityTestOnlineDataset to run the test
        ## Update the test state using the update method of ExchangeabilityTestOnlineDataset
        ## Check if the test has rejected the null hypothesis of exchangeability at each step and update the test state accordingly
        ## Return the summary of the test results using the summary method of ExchangeabilityTestOnlineDataset

        # 2. In the case of Permutation test, implement the test directly on the non-conformity scores (score_values)
        ## /!\ Is the score preserving the exchangeability property and preservig the non-exchangeability anomaly
        ## if the dataset is not exchangeable (for example, the change point, covariate shilft, time dependency, etc.)
        ## Generate a number of permuted datasets by shuffling non-conformity scores
        ## Compute the p-value as the proportion of permuted non-conformity score test statistics that are as extreme as or more extreme than the test statistic for the original dataset
        ## Compare the p-value to the significance level to determine if the null hypothesis of exchangeability is rejected or not
        ## Return the summary of the test results
        pass


# Init for fixed dataset test
etfd = ExchangeabilityTestFixedDataset(
    test_method="all",  # or "Permutation", "Martingale", "RiskMonitoring",
    # Call all the tests and return the results in a kind of summary table,
    significance_level=0.05,  # Significance level for the tests (default: 0.05)
)

etfd.run(X_test, y_test)
# Output a summary table of the test results for each test method,
# including the test statistic, p-value, and conclusion about exchangeability based on the significance level

# Init for fixed non-conformity score test
## 1. The test object
etfd_ncs = ExchangeabilityTestFixedNonConformityScore(
    test_method="all",  # or "Permutation", "Martingale", "RiskMonitoring",
    # Call all the tests and return the results in a kind of summary table,
    significance_level=0.05,  # Significance level for the tests (default: 0.05)
)
## 2. Compute the non-conformity scores for the dataset (X_test, y_test) using a suitable non-conformity score function
## that preserves the exchangeability property and preservig the non-exchangeability anomaly
## if the dataset is not exchangeable (for example, the change point, covariate shilft, time dependency, etc.)
## This step might not be easy with MAPIE (may be can be done using the MAPIE conformilizer)
score_values = compute_non_conformity_scores(X_test, y_test)

## 3. Run the test on the non-conformity scores
etfd_ncs.run(score_values)
