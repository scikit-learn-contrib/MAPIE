from typing import Callable, Optional

import numpy as np

from ._typing import NDArray


def phi1D(
    x: NDArray,
    B: NDArray,
    fun: Callable[[NDArray], NDArray],
) -> NDArray:
    """
    The function phi1D is called by phi2D.
    It aims at applying a function ``fun`` after multiplying each row
    of B by x.

    Parameters
    ----------
    x : NDArray of shape (n, )
        1D vector.
    B : NDArray of shape (k, n)
        2D vector whose number of columns is the number of rows of x.
    fun : function
        Vectorized function applying to NDArray.

    Returns
    -------
    NDArray
        The function fun is applied to the product of ``x`` and ``B``.
        Typically, ``fun`` is a numpy function, ignoring nan,
        with argument ``axis=1``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> B = np.array([[1, 1, 1, np.nan, np.nan],
    ...               [np.nan, np.nan, 1, 1, 1]])
    >>> fun = lambda x: np.nanmean(x, axis=1)
    >>> res = phi1D(x, B, fun)
    >>> print(res)
    [2. 4.]
    """
    return fun(x * B)


def phi2D(
    A: NDArray,
    B: NDArray,
    fun: Callable[[NDArray], NDArray],
) -> NDArray:
    """
    The function phi2D is a loop applying phi1D on each row of A.

    Parameters
    ----------
    A : NDArray of shape (n_rowsA, n_columns)
    B : NDArray of shape (n_rowsB, n_columns)
        A and B must have the same number of columns.

    fun : function
        Vectorized function applying to NDArray, and that should ignore nan.

    Returns
    -------
    NDArray of shape (n_rowsA, n_rowsB)
        Applies phi1D(x, B, fun) to each row x of A.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]])
    >>> B = np.array([[1, 1, 1, np.nan, np.nan],
    ...               [np.nan, np.nan, 1, 1, 1]])
    >>> fun = lambda x: np.nanmean(x, axis=1)
    >>> res = phi2D(A, B, fun)
    >>> print(res.ravel())
    [ 2.  4.  7.  9. 12. 14.]
    """
    return np.apply_along_axis(phi1D, axis=1, arr=A, B=B, fun=fun)


def aggregate_all(agg_function: Optional[str], X: NDArray) -> NDArray:
    """
    Applies np.nanmean(, axis=1) or np.nanmedian(, axis=1) according
    to the string ``agg_function``.

    Parameters
    -----------
    X : NDArray of shape (n, p)
        Array of floats and nans

    Returns
    --------
    NDArray of shape (n, 1):
        Array of the means or medians of each row of X

    Raises
    ------
    ValueError
        If agg_function is ``None``

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.aggregation_functions import aggregate_all
    >>> agg_function = "mean"
    >>> aggregate_all(agg_function,
    ...     np.array([list(range(30)),
    ...     list(range(30))]))
    array([14.5, 14.5])

    """
    if agg_function == "median":
        return np.nanmedian(X, axis=1)
    elif agg_function == "mean":
        return np.nanmean(X, axis=1)
    raise ValueError("Aggregation function called but not defined.")
