from typing import Callable, Dict, Any, Optional
import inspect


def filter_params(
    function: Callable,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Filters the given parameters to only include keys that match the arguments
    of the provided function.

    Args:
        function (Callable): The function whose argument names are used to
        filter params.
        params (Optional[Dict[str, Any]]): The dictionary of parameters
        to be filtered. Defaults to an empty dictionary if None.

    Returns:
        Dict[str, Any]: A dictionary containing only the key-value pairs
        from `params` that match the arguments of `function`.
    """
    if params is None:
        return {}

    model_params = inspect.signature(function).parameters
    return {k: v for k, v in params.items() if k in model_params}
