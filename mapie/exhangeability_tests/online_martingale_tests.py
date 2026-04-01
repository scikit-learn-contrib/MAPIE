import warnings
from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

# Test level
alpha = 0.05


# The class could be adapted for Martingale test and Risk monitoring test.
# The following framing is more close to the Martingale test.
class OnlineMartingaleTest:
    def __init__(
        self,
        non_conformity_score_function: Callable[
            [NDArray, NDArray, Optional[NDArray]], NDArray
        ],
        test_method: Literal[
            "jumper_martingale", "plugin_martingale"
        ] = "jumper_martingale",
        confidence_level: float = 0.95,
        raise_warning: bool = True,
    ):
        self.non_conformity_score_function = non_conformity_score_function
        self.test_method = test_method
        self.confidence_level = confidence_level
        self.raise_warning = raise_warning

        # General test state
        self.pvalue_history = []
        self.non_conformity_score_history = []
        self.martingale_value_history = []
        self.current_martingale_value = 1.0
        self.is_exchangeable = None

        # Simple Jumper state
        self.jump_size = 0.01
        self._jumper_initialized = False
        self._jumper_expert_grid = np.array([-1.0, 0.0, 1.0], dtype=float)
        self._jumper_wealth_by_expert = np.full(3, 1 / 3, dtype=float)

    @property
    def is_exchangeable(self):
        # Return True if exchangeable : case last_stat_value < alpha
        # Return False if not exchangeable : case last_stat_value > 1/alpha
        # Return None if not enough data to determine exchangeability : case less data 200 data points and last_stat_value between alpha and 1/alpha
        pass

    def compute_p_value(self, current_non_conformity_score: float, non_conformity_score_history: NDArray) -> float:
        """
        Estimate the p-value for the current non-conformity score based on the history of non-conformity scores.

        pvalue_i = (1 + #{non_conformity_score_j >= non_conformity_score_i}) / (i + 1)

        Parameters
        ----------
        current_non_conformity_score : float
        non_conformity_score_history : list

        Returns
        -------
        float

        References:
        ----------
        Theoretical Foundations of Conformal Prediction, Angelopoulos, Barber, Bates (2026), Definition 3.8, page 35.
        """
        n = len(non_conformity_score_history)
        if n < 50:
            return np.random.uniform(0, 1)

        non_conformity_score_history = np.asarray(non_conformity_score_history)
        score_greater_or_equal_to_current = np.sum(non_conformity_score_history >= current_non_conformity_score)
        pvalue = (1 + score_greater_or_equal_to_current) / (n + 1)
        return pvalue

    def _estimate_pvalues_density(self, pvalue: float) -> float:
        """
        Estimate density of p-values using reflected Kernel Density Estimation and normalize
        over the unit interval.

        Parameters
        ----------
        pvalue : float
            Evaluation point.

        Returns
        -------
        float
            Estimated density value.
        """

        # When insufficient data, return uniform density
        # Under the null hypothesis of exchangeability, p-values are uniformly distributed,
        # so this is a neutral choice that does not bias the test in either direction.
        if len(self.pvalue_history) < 50:
            return np.random.uniform(0, 1)

        p_array = np.asarray(self.pvalue_history)

        # reflection to reduce boundary bias
        reflected = np.concatenate([-p_array, p_array, 2 - p_array])

        kde = gaussian_kde(reflected, bw_method="silverman")

        # compute normalization constant over [0,1]
        grid = np.linspace(0, 1, 200)
        density_vals = kde(grid)
        normalization_val = np.trapezoid(density_vals, grid)

        # control against numerical failure
        if normalization_val <= 0 or not np.isfinite(normalization_val):
            return 1.0

        # density value at p
        density = kde([pvalue])[0] / normalization_val

        # enforce support restriction
        if pvalue < 0 or pvalue > 1:
            density = 0.0

        return max(float(density), 1e-12)

    def update_simple_jumper_martingale(
        self,
        pvalue: float,
        jump_size: float = 0.01,
    ) -> float:
        if not (0.0 <= pvalue <= 1.0):
            raise ValueError("pvalue must lie in [0,1].")

        if not (0.0 <= jump_size <= 1.0):
            raise ValueError("jump_size must lie in [0,1].")

        # Initialize jumper state if needed
        jumper_expert_grid = np.array([-1.0, 0.0, 1.0], dtype=float)
        jumper_wealth_by_expert = np.full(3, 1 / 3, dtype=float)

        # Store p-value history
        self.pvalue_history.append(pvalue)

        # Previous total wealth M_{t-1}
        M_prev = self.current_martingale_value

        # Step 1: simultaneous jump / mixing update
        mixed_wealth = (1.0 - jump_size) * jumper_wealth_by_expert + (
            jump_size / 3.0
        ) * M_prev

        # Step 2: betting update using h_e(p) = 1 + e(p - 1/2)
        betting_multipliers = 1.0 + jumper_expert_grid * (pvalue - 0.5)
        jumper_wealth_by_expert = mixed_wealth * betting_multipliers

        # Step 3: aggregate martingale value
        self.current_martingale_value = float(np.sum(jumper_wealth_by_expert))
        self.martingale_value_history.append(self.current_martingale_value)

        return self.current_martingale_value

    def update_plugin_martingale(self, pvalue: float) -> float:
        rho_hat = self._estimate_pvalues_density(pvalue)
        self.current_martingale_value *= rho_hat
        self.martingale_value_history.append(self.current_martingale_value)
        return self.current_martingale_value

    def update(self, y_true: NDArray, y_pred: NDArray, X: Optional[NDArray] = None):
        # Update the test with the new point (y_true, y_pred, X)
        # Compute conformity scores and update the test state

        # 1. compute conformity score for the new point
        non_conformity_score = self.non_conformity_score_function(y_true, y_pred, X)

        for current_score in non_conformity_score:
            # 2. Save the conformity score for the new point for future updates
            self.non_conformity_score_history.append(current_score)
            pvalue = self.compute_p_value(current_score, self.non_conformity_score_history)
            self.pvalue_history.append(pvalue)
            if self.test_method == "jumper_martingale":
                self.update_simple_jumper_martingale(pvalue, self.jump_size)
            elif self.test_method == "plugin_martingale":
                self.update_plugin_martingale(pvalue)
            else:
                raise ValueError(f"Unsupported test method: {self.test_method}")

        # 4. Check if the test has rejected the null hypothesis of exchangeability and update the test state accordingly
        alpha_level = 1.0 - self.confidence_level
        if not 0.0 < alpha_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")

        reject_threshold = 1.0 / alpha_level

        if self.current_martingale_value < alpha_level:
            self._is_exchangeable = True
        elif self.current_martingale_value > reject_threshold:
            self._is_exchangeable = False
        elif len(self.pvalue_history) < 200:
            self._is_exchangeable = None
        else:
            self._is_exchangeable = None

        if self._is_exchangeable is False and self.raise_warning:
            warnings.warn(
            "The online martingale test has rejected exchangeability. "
            f"Martingale value = {self.current_martingale_value:.3g} "
            f"exceeds threshold = {reject_threshold:.3g}.",
            UserWarning,
            )

        # 5. Return self
        return self

    def summary(self):
        martingale_values = np.asarray(self.martingale_value_history, dtype=float)
        alpha_level = 1.0 - self.confidence_level
        reject_threshold = 1.0 / alpha_level

        if martingale_values.size == 0:
            return {
            "test_method": self.test_method,
            "confidence_level": self.confidence_level,
            "alpha_level": alpha_level,
            "reject_threshold": reject_threshold,
            "current_martingale_value": float(self.current_martingale_value),
            "is_exchangeable": getattr(self, "_is_exchangeable", None),
            "n_non_conformity_scores": len(self.non_conformity_score_history),
            "n_pvalues": len(self.pvalue_history),
            "stopping_time": None,
            "first_rejection_index": None,
            "first_acceptance_index": None,
            }

        quantiles = np.quantile(
            martingale_values, [0.0, 0.025, 0.25, 0.5, 0.75, 0.975, 1.0]
        )
        first_rejection_index = next(
            (i + 1 for i, value in enumerate(martingale_values) if value > reject_threshold),
            None,
        )
        first_acceptance_index = next(
            (i + 1 for i, value in enumerate(martingale_values) if value < alpha_level),
            None,
        )
        if first_rejection_index is not None and first_acceptance_index is not None:
            stopping_time = min(first_rejection_index, first_acceptance_index)
        else:
            stopping_time = first_rejection_index or first_acceptance_index

        return {
            "test_method": self.test_method,
            "confidence_level": self.confidence_level,
            "alpha_level": alpha_level,
            "reject_threshold": reject_threshold,
            "current_martingale_value": float(self.current_martingale_value),
            "is_exchangeable": getattr(self, "_is_exchangeable", None),
            "n_non_conformity_scores": len(self.non_conformity_score_history),
            "n_pvalues": len(self.pvalue_history),
            "martingale_statistics": {
            "min": float(quantiles[0]),
            "q025": float(quantiles[1]),
            "q25": float(quantiles[2]),
            "median": float(quantiles[3]),
            "mean": float(np.mean(martingale_values)),
            "q75": float(quantiles[4]),
            "q975": float(quantiles[5]),
            "max": float(quantiles[6]),
            },
            "stopping_time": stopping_time,
            "first_rejection_index": first_rejection_index,
            "first_acceptance_index": first_acceptance_index,
            "last_observation_index": int(martingale_values.size),
        }

# # Initialize the test object
# etod = OnlineMartingaleTest(
#     test_method="jumper_martingale",
#     confidence_level=0.95,
#     non_conformity_score_function=None,  # ... function that takes (y_true, y_pred, X) and returns a non-conformity score as a float
# )


# # Initialize the test through update method before online
# # Assume you have already some data points (X_test, y_test) and predictions (y_pred) to initialize the test state before starting the online updates
# etod.update(y_test, y_pred, X_test) # X_test is optional, only used for some test methods

# X_online_all = []
# y_pred_all = []

# def get_next_features():
#     # Get the next features for the online data point
#     # This function should return the features of the next online data point as a numpy array
#     pass

# while True: # Update the stopping rule
#     # Get the next test point (y_true, y_pred, X)
#     x_online = get_next_features()
#     y_pred = clf.predict_proba(x_online.reshape(1, -1))[0, 1]  # Get the predicted probability for the positive class
#     X_online_all.append(x_online)
#     y_pred_all.append(y_pred)

# # get some labels
# X_online_test = [x_online_1, x_online_8, ...]
# y_pred_test = [y_pred_1, y_pred_8, ...]  # Get the predicted probabilities for the online data point(s)
# y_true = get_next_label(X_online_test)  # Get the true label for the online data point(s)
# # Update the test with the new point
# etod.update(y_true, y_pred_test, X_online_test)  # X_online_test is optional, only used for some test methods
# # Check if the test has rejected the null hypothesis
