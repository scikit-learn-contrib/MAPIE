from typing import Any, Dict, List, Literal, Optional, Protocol, Union, cast

from numpy.typing import NDArray

from mapie.exchangeability_testing.martingales import OnlineMartingaleTest
from mapie.exchangeability_testing.permutations import (
    PValuePermutationTest,
    SequentialMonteCarloTest,
)

FixedDatasetTestMethod = Union[
    PValuePermutationTest,
    SequentialMonteCarloTest,
    OnlineMartingaleTest,
]
OnlineTestMethod = OnlineMartingaleTest
ExchangeabilityDecision = Optional[bool]
MethodParams = Dict[str, Dict[str, Any]]


class ExchangeabilityTestProtocol(Protocol):
    @property
    def is_exchangeable(self) -> ExchangeabilityDecision: ...


class RunnableExchangeabilityTestProtocol(ExchangeabilityTestProtocol, Protocol):
    def run(self, X: NDArray, y: NDArray) -> FixedDatasetTestMethod: ...


class UpdatableExchangeabilityTestProtocol(ExchangeabilityTestProtocol, Protocol):
    def update(self, X: NDArray, y: NDArray) -> FixedDatasetTestMethod: ...


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
    """
    Run one or several exchangeability tests on a labeled dataset.

    This wrapper provides a high-level interface around the exchangeability
    testing methods implemented in MAPIE. It can instantiate permutation-based
    tests as well as online martingale tests and run them through a shared API.

    Parameters
    ----------
    method_names : Union[FixedDatasetTestMethods, Literal["all"], \
List[FixedDatasetTestMethods]], default="all"
        Name of the test method to run, a list of method names, or ``"all"``
        to run every available fixed-dataset method.

    method_params : Optional[MethodParams], default=None
        Additional keyword arguments passed to each method constructor. Keys are
        method names and values are dictionaries of keyword arguments.

    test_level : float, default=0.05
        Significance level passed to each underlying test.

    warn : bool, default=False
        Whether underlying methods should raise warnings when they reject
        exchangeability.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(20, dtype=float).reshape(-1, 1)
    >>> y = 2 * X.ravel() + np.linspace(0.0, 0.1, X.shape[0])
    >>> test = FixedDatasetExchangeabilityTest(
    ...     method_names="pvalue_permutation", warn=False
    ... )
    >>> _ = test.run(X, y)
    """

    def __init__(
        self,
        method_names: Union[
            FixedDatasetTestMethods, Literal["all"], List[FixedDatasetTestMethods]
        ] = "all",
        method_params: Optional[MethodParams] = None,
        test_level: float = 0.05,
        warn: bool = False,
    ) -> None:
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

    def _init_test_method(self, method_name: str) -> FixedDatasetTestMethod:
        """Instantiate one fixed-dataset exchangeability test."""
        method_class = fixed_dataset_test_method_choice_map[method_name]
        params = {**self.method_params.get(method_name, {})}
        if method_class is OnlineMartingaleTest:
            params = {"test_method": method_name, **params}
        if method_class is SequentialMonteCarloTest:
            strategy = method_name.removeprefix("permutation_")
            params = {"strategy": strategy, **params}
        return cast(
            FixedDatasetTestMethod,
            method_class(
                test_level=self.test_level,
                warn=self.warn,
                **params,
            ),
        )

    @property
    def is_exchangeable(self) -> Dict[str, ExchangeabilityDecision]:
        """
        Return the current exchangeability decision for each configured method.

        Returns
        -------
        Dict[str, Optional[bool]]
            A dictionary mapping each method name to its current decision.
            Values are typically ``True``, ``False``, or ``None`` when the
            underlying test is still inconclusive.
        """
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            results[method_name] = test_method.is_exchangeable
        return results

    def run(
        self,
        X_test: NDArray,
        y_test: NDArray,
    ) -> Dict[str, FixedDatasetTestMethod]:
        """
        Run all configured exchangeability tests on the provided dataset.

        Parameters
        ----------
        X_test : NDArray
            Feature matrix of the labeled dataset.

        y_test : NDArray
            Labels or targets associated with ``X_test``.

        Returns
        -------
        Dict[str, FixedDatasetTestMethod]
            A dictionary mapping each method name to the updated underlying
            test instance.

        Raises
        ------
        AttributeError
            If one of the configured test methods defines neither ``update``
            nor ``run``.
        """
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if callable(getattr(test_method, "update", None)):
                results[method_name] = cast(
                    UpdatableExchangeabilityTestProtocol, test_method
                ).update(X_test, y_test)
            elif callable(getattr(test_method, "run", None)):
                results[method_name] = cast(
                    RunnableExchangeabilityTestProtocol, test_method
                ).run(X_test, y_test)
            else:
                raise AttributeError(
                    f"Test method '{method_name}' must define either 'update' or 'run'."
                )

        return results


class OnlineExchangeabilityTest:
    """
    Monitor exchangeability online with one or several martingale tests.

    This wrapper exposes a shared interface for the online exchangeability
    testing methods available in MAPIE. Each configured method is updated on
    the same labeled stream, allowing side-by-side monitoring of different
    martingale constructions.

    Parameters
    ----------
    method_names : Union[OnlineTestMethods, Literal["all"], \
List[OnlineTestMethods]], default="all"
        Name of the online method to use, a list of method names, or ``"all"``
        to instantiate every available online method.

    method_params : Optional[MethodParams], default=None
        Additional keyword arguments passed to each method constructor. Keys are
        method names and values are dictionaries of keyword arguments.

    test_level : float, default=0.05
        Significance level passed to each underlying online test.

    warn : bool, default=True
        Whether underlying methods should raise warnings when they reject
        exchangeability.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(120, dtype=float).reshape(-1, 1)
    >>> y = 2 * X.ravel() + np.linspace(0.0, 0.1, X.shape[0])
    >>> online_test = OnlineExchangeabilityTest(
    ...     method_names="plugin_martingale", warn=False
    ... )
    >>> _ = online_test.update(X, y)
    """

    def __init__(
        self,
        method_names: Union[
            OnlineTestMethods, Literal["all"], List[OnlineTestMethods]
        ] = "all",
        method_params: Optional[MethodParams] = None,
        test_level: float = 0.05,
        warn: bool = True,
    ) -> None:
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

    def _init_test_method(self, method_name: str) -> OnlineTestMethod:
        """Instantiate one online exchangeability test."""
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
    def is_exchangeable(self) -> Dict[str, ExchangeabilityDecision]:
        """
        Return the current exchangeability decision for each online method.

        Returns
        -------
        Dict[str, Optional[bool]]
            A dictionary mapping each method name to its current online
            decision. Values may be ``None`` during the burn-in phase.
        """
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            results[method_name] = test_method.is_exchangeable
        return results

    def update(
        self,
        X_test: NDArray,
        y_test: NDArray,
    ) -> Dict[str, OnlineTestMethod]:
        """
        Update all configured online tests with newly labeled observations.

        Parameters
        ----------
        X_test : NDArray
            Feature matrix for the newly observed batch.

        y_test : NDArray
            Labels or targets associated with ``X_test``.

        Returns
        -------
        Dict[str, OnlineMartingaleTest]
            A dictionary mapping each method name to the updated underlying
            test instance.

        Raises
        ------
        AttributeError
            If one of the configured methods does not define ``update``.
        """
        results = {}
        for test_method, method_name in zip(self.test_methods, self.method_names):
            if callable(getattr(test_method, "update", None)):
                results[method_name] = test_method.update(X_test, y_test)
            else:
                raise AttributeError(
                    f"Test method '{method_name}' must define an 'update' method."
                )
        return results
