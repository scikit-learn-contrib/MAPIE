from typing import Any

import numpy as np
from packaging.version import parse as parse_version

from ._typing import ArrayLike, NDArray


def np_quantile_version_below_122(
    a: ArrayLike,
    q: ArrayLike,
    method: str = "linear",
    **kwargs: Any
) -> NDArray:
    """Wrapper of np.quantile function for numpy version < 1.22."""
    return np.quantile(a, q, interpolation=method, **kwargs)  # type: ignore


def np_quantile_version_above_122(
    a: ArrayLike,
    q: ArrayLike,
    method: str = "linear",
    **kwargs: Any
) -> NDArray:
    """Wrapper of np.quantile function for numpy version >= 1.22."""
    return np.quantile(a, q, method=method, **kwargs)  # type: ignore


def np_nanquantile_version_below_122(
    a: ArrayLike,
    q: ArrayLike,
    method: str = "linear",
    **kwargs: Any
) -> NDArray:
    """Wrapper of np.quantile function for numpy version < 1.22."""
    return np.nanquantile(a, q, interpolation=method, **kwargs)  # type: ignore


def np_nanquantile_version_above_122(
    a: ArrayLike,
    q: ArrayLike,
    method: str = "linear",
    **kwargs: Any
) -> NDArray:
    """Wrapper of np.quantile function for numpy version >= 1.22."""
    return np.nanquantile(a, q, method=method, **kwargs)  # type: ignore


numpy_version = parse_version(np.__version__)
if numpy_version < parse_version("1.22"):
    np_quantile = np_quantile_version_below_122
    np_nanquantile = np_nanquantile_version_below_122

else:
    np_quantile = np_quantile_version_above_122
    np_nanquantile = np_nanquantile_version_above_122
