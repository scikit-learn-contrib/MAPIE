from typing import Literal, Union

from numpy.typing import NDArray

from mapie.exchangeability_testing.risk_monitoring import RiskMonitoring

OnlineTestMethods = Literal[
    "all",
    "Plug-in Martingale",
    "Jumper Martingale",
    "Risk Monitoring",
]

FixedDatasetTestMethods = Union[
    OnlineTestMethods,
    Literal[
        "p-value Permutation",
        "Monte-Carlo Permutation Binomial",
        "Monte-Carlo Permutation Mixture Binomial",
        "Monte-Carlo Permutation Aggressive",
    ],
]

test_method_choice_map = {
    "all": None,
    "Plug-in Martingale": None,
    "Jumper Martingale": None,
    "Risk Monitoring": RiskMonitoring,
    "p-value Permutation": None,
    "Monte-Carlo Permutation Binomial": None,
    "Monte-Carlo Permutation Mixture Binomial": None,
    "Monte-Carlo Permutation Aggressive": None,
}


class FixedDatasetExchangeabilityTest:
    def __init__(
        self, method_name: FixedDatasetTestMethods, test_level=0.05, warn=True
    ):
        self.method_name = method_name
        self.test_level = test_level
        self.warn = warn

        self.test_method = test_method_choice_map[method_name]

    @property
    def is_exchangeable(self):
        if self.method_name == "Risk Monitoring":
            return not self.test_method.harmful_shift_detected

    def run(self, X_test: NDArray, y_test: NDArray):
        if self.method_name == "Risk Monitoring":
            return self.test_method.update_online_risk(y_test, y_pred)
        pass


class OnlineExchangeabilityTest:
    def __init__(self, method_name: OnlineTestMethods, test_level=0.05, warn=True):
        self.method_name = method_name
        self.test_level = test_level
        self.warn = warn

    @property
    def is_exchangeable(self):
        if self.method_name == "Risk Monitoring":
            return not self.test_method.harmful_shift_detected

    def update(self, X_test: NDArray, y_test: NDArray):
        if self.method_name == "Risk Monitoring":
            return self.test_method.update_online_risk(y_test, y_pred)
