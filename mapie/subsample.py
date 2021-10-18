from __future__ import annotations

from typing import Any, Callable, Generator, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state, resample

from ._typing import ArrayLike


class Subsample(BaseCrossValidator):  # type: ignore
    """
    Generate a sampling method, that resamples the training set with
    possible bootstraps. It can replace KFold or  LeaveOneOut as cv argument
    in the MAPIE class.

    Parameters
    ----------
    n_resamplings : int
        Number of resamplings.
    n_samples: int
        Number of samples in each resampling. By default None,
        the size of the training set.
    replace: bool
        Whether to replace samples in resamplings or not.
    random_state: Optional
        int or RandomState instance.


    Examples
    --------
    >>> import pandas as pd
    >>> from mapie.subsample import Subsample
    >>> cv = Subsample(n_resamplings=2,random_state=0)
    >>> X = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9,10]))
    >>> for train_index, test_index in cv.split(X):
    ...    print(f"train index is {train_index}, test index is {test_index}")
    train index is [5 0 3 3 7 9 3 5 2 4], test index is [8 1 6]
    train index is [7 6 8 8 1 6 7 7 8 1], test index is [0 2 3 4 5 9]
    """

    def __init__(
        self,
        n_resamplings: int,
        n_samples: Optional[int] = None,
        replace: bool = True,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        self.n_resamplings = n_resamplings
        self.n_samples = n_samples
        self.replace = replace
        self.random_state = random_state

    def split(
        self, X: ArrayLike
    ) -> Generator[Tuple[Any, ArrayLike], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : ArrayLike of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ArrayLike of shape (n_indices_training,)
            The training set indices for that split.
        test : ArrayLike of shape (n_indices_test,)
            The testing set indices for that split.
        """
        indices = np.arange(len(X))
        n_samples = (
            self.n_samples if self.n_samples is not None else len(indices)
        )
        random_state = check_random_state(self.random_state)
        for k in range(self.n_resamplings):
            train_index = resample(
                indices,
                replace=self.replace,
                n_samples=n_samples,
                random_state=random_state,
                stratify=None,
            )
            test_index = np.array(
                list(set(indices) - set(train_index)), dtype=np.int64
            )
            yield train_index, test_index

    def get_n_splits(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:

        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : ArrayLike
            Always ignored, exists for compatibility with BaseCrossValidator
            object. By default, None.
        y : ArrayLike
            Always ignored, exists for compatibility with BaseCrossValidator
            object. By default, None.
        groups : ArrayLike
            Always ignored, exists for compatibility with BaseCrossValidator
            object. By default, None.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_resamplings


def phi1D(
    x: ArrayLike,
    B: ArrayLike,
    fun: Callable[[ArrayLike], ArrayLike],
) -> ArrayLike:
    """
    The function phi1D is called by phi2D. It aims at multiplying two matrices
    and apply a function to the product.

    Parameters
    ----------
    x : ArrayLike
        1D vector.
    B : ArrayLike
        2D vector whose number of columns is the length of x.
    fun : function
        Vectorized function applying to Arraylike, and that should ignore nan.

    Returns
    -------
    ArrayLike
        Each row of ``B`` is multiply by ``x`` and then the function fun is
        applied. Typically, ``fun`` is a numpy function, with argument
        ``axis = 1``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> B = np.array([[1, 1, 1, np.nan, np.nan], [np.nan, np.nan, 1, 1, 1]])
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
    The function phi2D is a loop applying phi1D.

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
        Apply phi1D(x, B, fun) to each row x of A.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10],[11, 12, 13, 14, 15]])
    >>> B = np.array([[1, 1, 1, np.nan, np.nan],[np.nan, np.nan, 1, 1, 1]])
    >>> fun = lambda x: np.nanmean(x, axis=1)
    >>> res = phi2D(A, B, fun)
    >>> print(res)
    [[ 2.  4.]
     [ 7.  9.]
     [12. 14.]]
    """
    return np.apply_along_axis(phi1D, axis=1, arr=A, B=B, fun=fun)
