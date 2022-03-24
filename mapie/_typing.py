from typing import Union, List, Any

import numpy as np


try:
    from numpy.typing import ArrayLike, NDArray
except (AttributeError, ModuleNotFoundError):
    ArrayLike = Union[np.ndarray, List[Any]]  # type: ignore
    NDArray = np.ndarray  # type: ignore

__all__ = ["ArrayLike", "NDArray"]
