import numpy as np
from typing import Union, List

try:
    from numpy.typing import ArrayLike
except (AttributeError, ModuleNotFoundError):
    ArrayLike = Union[np.ndarray, List[List[float]]]

__all__ = ["ArrayLike"]
