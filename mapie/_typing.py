import numpy as np
from typing import Union, List


try:
    from numpy.typing import ArrayLike, NDArray
except (AttributeError, ModuleNotFoundError):
    ArrayLike = Union[np.ndarray, List[List[float]]]
    NDArray = np.ndarray

__all__ = ["ArrayLike", "NDArray"]
