from typing import List, Literal, Union

from numpy.typing import NDArray

from mapie.exchangeability_testing.risk_monitoring import RiskMonitoring

online_test_method_choice_map = {
    "Plug-in Martingale": None,
    "Jumper Martingale": None,
    "Risk Monitoring": RiskMonitoring,
}

fixed_dataset_test_method_choice_map = {
    "p-value Permutation": None,
    "Monte-Carlo Permutation Binomial": None,
    "Monte-Carlo Permutation Mixture Binomial": None,
    "Monte-Carlo Permutation Aggressive": None,
    **online_test_method_choice_map,
}

OnlineTestMethods = Literal[tuple(online_test_method_choice_map.keys())]
FixedDatasetTestMethods = Literal[tuple(fixed_dataset_test_method_choice_map.keys())]


class FixedDatasetExchangeabilityTest:
    def __init__(
        self,
        method_names: Union[
            FixedDatasetTestMethods, Literal["all"], List[FixedDatasetTestMethods]
        ] = "all",
        test_level=0.05,
        warn=True,
    ):
        if method_names == "all":
            self.method_names = list(fixed_dataset_test_method_choice_map.keys())
        elif isinstance(method_names, str):
            self.method_names = [method_names]
        elif isinstance(method_names, list):
            self.method_names = method_names
        else:
            raise ValueError(
                f"Invalid method_names type: {type(method_names)}. Must be a string, list, or 'all'."
            )

        for method_name in self.method_names:
            if method_name not in fixed_dataset_test_method_choice_map:
                raise ValueError(
                    f"Invalid method name: {method_name}. Valid methods are: {list(fixed_dataset_test_method_choice_map.keys())}"
                )

        self.test_methods = [
            fixed_dataset_test_method_choice_map[method_name]
            for method_name in self.method_names
        ]

        self.test_level = test_level
        self.warn = warn

    @property
    def is_exchangeable(self):
        if self.method_name == "Risk Monitoring":
            return not self.test_method.harmful_shift_detected

    def run(self, X_test: NDArray, y_test: NDArray):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if method_name == "Risk Monitoring":
                results[method_name] = test_method.update_online_risk(y_test, y_pred)
            else:
                results[method_name] = test_method.run(X_test, y_test)
        return results


class OnlineExchangeabilityTest:
    def __init__(
        self,
        method_names: Union[
            OnlineTestMethods, Literal["all"], List[OnlineTestMethods]
        ] = "all",
        test_level=0.05,
        warn=True,
    ):
        if method_names == "all":
            self.method_names = list(online_test_method_choice_map.keys())
        elif isinstance(method_names, str):
            self.method_names = [method_names]
        elif isinstance(method_names, list):
            self.method_names = method_names
        else:
            raise ValueError(
                f"Invalid method_names type: {type(method_names)}. Must be a string, list, or 'all'."
            )

        for method_name in self.method_names:
            if method_name not in online_test_method_choice_map:
                raise ValueError(
                    f"Invalid method name: {method_name}. Valid methods are: {list(online_test_method_choice_map.keys())}"
                )

        self.test_methods = [
            online_test_method_choice_map[method_name]
            for method_name in self.method_names
        ]

        self.test_level = test_level
        self.warn = warn

    @property
    def is_exchangeable(self):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if method_name == "Risk Monitoring":
                results[method_name] = not test_method.harmful_shift_detected
        return results

    def update(self, X_test: NDArray, y_test: NDArray, y_pred: NDArray):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if method_name == "Risk Monitoring":
                results[method_name] = test_method.update_online_risk(y_test, y_pred)
        return results
