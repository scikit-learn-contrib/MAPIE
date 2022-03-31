import numpy as np
from typing import Union, List


try:
    from numpy.typing import ArrayLike, NDArray
except (AttributeError, ModuleNotFoundError, ImportError):
    ArrayLike = Union[np.ndarray, List[List[float]]]  # type: ignore
    NDArray = np.ndarray  # type: ignore

__all__ = ["ArrayLike", "NDArray"]
