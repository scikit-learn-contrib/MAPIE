from typing import Any, Dict, List, Literal, Optional, Union

from numpy.typing import NDArray

from mapie.exchangeability_testing.risk_monitoring import RiskMonitoring

online_test_method_choice_map = {
    # "Plug-in Martingale": None,
    # "Jumper Martingale": None,
    "Risk Monitoring": RiskMonitoring,
}

fixed_dataset_test_method_choice_map = {
    # "p-value Permutation": None,
    # "Monte-Carlo Permutation Binomial": None,
    # "Monte-Carlo Permutation Mixture Binomial": None,
    # "Monte-Carlo Permutation Aggressive": None,
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
        method_params: Optional[Dict[str, Dict[str, Any]]] = None,
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

        self.test_level = test_level
        self.warn = warn
        self.method_params = method_params or {}
        self.test_methods = [
            self._init_test_method(method_name) for method_name in self.method_names
        ]

    def _init_test_method(self, method_name: str):
        method_class = fixed_dataset_test_method_choice_map[method_name]
        if method_name == "Risk Monitoring":
            params = {"risk": "accuracy", **self.method_params.get(method_name, {})}
            return method_class(test_level=self.test_level, warn=self.warn, **params)
        return None

    @property
    def is_exchangeable(self):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if method_name == "Risk Monitoring":
                results[method_name] = not test_method.harmful_shift_detected
        return results

    def run(self, y_test: NDArray, y_pred: NDArray, X_test: Optional[NDArray] = None):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            results[method_name] = test_method.update(y_test, y_pred)

        return results


class OnlineExchangeabilityTest:
    def __init__(
        self,
        method_names: Union[
            OnlineTestMethods, Literal["all"], List[OnlineTestMethods]
        ] = "all",
        method_params: Optional[Dict[str, Dict[str, Any]]] = None,
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

        self.test_level = test_level
        self.warn = warn
        self.method_params = method_params or {}
        self.test_methods = [
            self._init_test_method(method_name) for method_name in self.method_names
        ]

    def _init_test_method(self, method_name: str):
        method_class = online_test_method_choice_map[method_name]
        if method_name == "Risk Monitoring":
            params = {"risk": "accuracy", **self.method_params.get(method_name, {})}
            return method_class(test_level=self.test_level, warn=self.warn, **params)
        return None

    @property
    def is_exchangeable(self):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if method_name == "Risk Monitoring":
                results[method_name] = not test_method.harmful_shift_detected
        return results

    def update(
        self, y_test: NDArray, y_pred: NDArray, X_test: Optional[NDArray] = None
    ):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            results[method_name] = test_method.update(y_test, y_pred)
        return results
