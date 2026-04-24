from typing import Any, Dict, List, Literal, Optional, Union, cast

from numpy.typing import NDArray

from mapie.exchangeability_testing.martingales import OnlineMartingaleTest
from mapie.exchangeability_testing.permutations import (
    PValuePermutationTest,
    SequentialMonteCarloTest,
)

online_test_method_choice_map = {
    "plugin_martingale": OnlineMartingaleTest,
    "jumper_martingale": OnlineMartingaleTest,
}

fixed_dataset_test_method_choice_map = {
    "pvalue_permutation": PValuePermutationTest,
    "permutation_binomial": SequentialMonteCarloTest,
    "permutation_binomial_mixture": SequentialMonteCarloTest,
    "permutation_aggressive": SequentialMonteCarloTest,
    **online_test_method_choice_map,
}

OnlineTestMethods = Literal[
    "plugin_martingale",
    "jumper_martingale",
]
FixedDatasetTestMethods = Literal[
    "pvalue_permutation",
    "permutation_binomial",
    "permutation_binomial_mixture",
    "permutation_aggressive",
    "plugin_martingale",
    "jumper_martingale",
]


class FixedDatasetExchangeabilityTest:
    def __init__(
        self,
        method_names: Union[
            FixedDatasetTestMethods, Literal["all"], List[FixedDatasetTestMethods]
        ] = "all",
        method_params: Optional[Dict[str, Dict[str, Any]]] = None,
        test_level=0.05,
        warn=False,
    ):
        if method_names == "all":
            self.method_names = list(fixed_dataset_test_method_choice_map.keys())
        elif isinstance(method_names, str):
            self.method_names = [method_names]
        elif isinstance(method_names, list):
            self.method_names = cast(List[str], method_names)
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
        params = {**self.method_params.get(method_name, {})}
        if method_class is OnlineMartingaleTest:
            params = {"test_method": method_name, **params}
        if method_class is SequentialMonteCarloTest:
            strategy = method_name.removeprefix("permutation_")
            params = {"strategy": strategy, **params}
        return method_class(
            test_level=self.test_level,
            warn=self.warn,
            **params,
        )

    @property
    def is_exchangeable(self):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            results[method_name] = test_method.is_exchangeable
        return results

    def run(
        self,
        X_test: NDArray,
        y_test: NDArray,
    ):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if callable(getattr(test_method, "update", None)):
                results[method_name] = test_method.update(X_test, y_test)
            elif callable(getattr(test_method, "run", None)):
                results[method_name] = test_method.run(X_test, y_test)
            else:
                raise AttributeError(
                    f"Test method '{method_name}' must define either 'update' or 'run'."
                )

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
            self.method_names = cast(List[str], method_names)
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
        params = {**self.method_params.get(method_name, {})}
        if method_class is OnlineMartingaleTest:
            params = {"test_method": method_name, **params}
        return method_class(
            test_level=self.test_level,
            warn=self.warn,
            **params,
        )

    @property
    def is_exchangeable(self):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            results[method_name] = test_method.is_exchangeable
        return results

    def update(self, X_test: NDArray, y_test: NDArray):
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if callable(getattr(test_method, "update", None)):
                results[method_name] = test_method.update(X_test, y_test)
            else:
                raise AttributeError(
                    f"Test method '{method_name}' must define an 'update' method."
                )
        return results
