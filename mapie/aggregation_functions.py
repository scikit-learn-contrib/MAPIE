from typing import Callable, Optional, List


from sklearn.base import RegressorMixin


import numpy as np

from ._typing import ArrayLike, NDArray


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


def pred_multi(
    X: ArrayLike,
    estimators_: List[RegressorMixin],
    agg_function: Optional[str],
    k_: NDArray,
) -> NDArray:
    """
    Return a prediction per train sample for each test sample, by
    aggregation with matrix  ``k_``.

    Parameters
    ----------
        X : NDArray of shape (n_samples_test, n_features)
            Input data

        estimators_ : List[RegressorMixin]
            List of out-of-folds estimators.

        agg_function : str
            Determines how to aggregate predictions from perturbed models, both
            at training and prediction time.

        k_ : NDArray of shape (n_samples_training, n_estimators)
            1-or-nan array: indicates whether to integrate the prediction
            of a given estimator into the aggregation, for each training
            sample.

    Returns
    -------
        NDArray of shape (n_samples_test, n_samples_train)
    """
    y_pred_multi = np.column_stack(
        [e.predict(X) for e in estimators_]
    )
    # At this point, y_pred_multi is of shape
    # (n_samples_test, n_estimators_). The method
    # ``_aggregate_with_mask`` fits it to the right size
    # thanks to the shape of k_.

    y_pred_multi = aggregate_with_mask(y_pred_multi, agg_function, k_)
    return y_pred_multi


def aggregate_with_mask(
    x: NDArray,
    agg_function: Optional[str],
    k: NDArray,
) -> NDArray:
    """
    Take the array of predictions, made by the refitted estimators,
    on the testing set, and the 1-or-nan array indicating for each training
    sample which one to integrate, and aggregate to produce phi-{t}(x_t)
    for each training sample x_t.


    Parameters:
    -----------
    x : ArrayLike of shape (n_samples_test, n_estimators)
        Array of predictions, made by the refitted estimators,
        for each sample of the testing set.

    agg_function : str
        Determines how to aggregate predictions from perturbed models, both
        at training and prediction time.

    k : ArrayLike of shape (n_samples_training, n_estimators)
        1-or-nan array: indicates whether to integrate the prediction
        of a given estimator into the aggregation, for each training
        sample.

    Returns:
    --------
    ArrayLike of shape (n_samples_test,)
        Array of aggregated predictions for each testing  sample.
    """
    if agg_function == "median":
        return phi2D(A=x, B=k, fun=lambda x: np.nanmedian(x, axis=1))

    # To aggregate with mean() the aggregation coud be done
    # with phi2D(A=x, B=k, fun=lambda x: np.nanmean(x, axis=1).
    # However, phi2D contains a np.apply_along_axis loop which
    # is much slower than the matrices multiplication that can
    # be used to compute the means.
    if agg_function in ["mean", None]:
        K = np.nan_to_num(k, nan=0.0)
        return np.matmul(x, (K / (K.sum(axis=1, keepdims=True))).T)
    raise ValueError("The value of agg_function is not correct")


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
