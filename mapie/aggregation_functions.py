from typing import Callable, Optional

import numpy as np

from ._typing import ArrayLike


def phi1D(
    x: ArrayLike,
    B: ArrayLike,
    fun: Callable[[ArrayLike], ArrayLike],
) -> ArrayLike:
    """
    The function phi1D is called by phi2D.
    It aims at applying a function ``fun`` after multiplying each row
    of B by x.

    Parameters
    ----------
    x : ArrayLike of shape (n, )
        1D vector.
    B : ArrayLike of shape (k, n)
        2D vector whose number of columns is the number of rows of x.
    fun : function
        Vectorized function applying to Arraylike.

    Returns
    -------
    ArrayLike
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
    A: ArrayLike,
    B: ArrayLike,
    fun: Callable[[ArrayLike], ArrayLike],
) -> ArrayLike:
    """
    The function phi2D is a loop applying phi1D on each row of A.

    Parameters
    ----------
    A : ArrayLike of shape (n_rowsA, n_columns)
    B : ArrayLike of shape (n_rowsB, n_columns)
        A and B must have the same number of columns.

    fun : function
        Vectorized function applying to Arraylike, and that should ignore nan.

    Returns
    -------
    ArrayLike of shape (n_rowsA, n_rowsB)
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


def aggregate_all(agg_function: Optional[str], X: ArrayLike) -> ArrayLike:
    """
    Applies np.nanmean(, axis=1) or np.nanmedian(, axis=1) according
    to the string ``agg_function``.

    Parameters
    -----------
    X : ArrayLike of shape (n, p)
        Array of floats and nans

    Returns
    --------
    ArrayLike of shape (n, 1):
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
