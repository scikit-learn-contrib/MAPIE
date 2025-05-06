from __future__ import annotations

from typing import Any, Generator, Optional, Tuple, Union, cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.random import RandomState
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state, resample
from sklearn.utils.validation import _num_samples

from numpy.typing import NDArray
from .utils import _check_n_samples


class Subsample(BaseCrossValidator):
    """
    Generate a sampling method, that resamples the training set with
    possible bootstraps. It can be used as cv argument in
    :class:`~mapie.regression.JackknifeAfterBootstrapRegressor`.

    Parameters
    ----------
    n_resamplings : int
        Number of resamplings. By default ``30``.
    n_samples: Union[int, float]
        Number of samples in each resampling. By default ``None``,
        the size of the training set. If it is between 0 and 1,
        it becomes the fraction of samples
    replace: bool
        Whether to replace samples in resamplings or not. By default ``True``.
    random_state: Optional[Union[int, RandomState]]
        int or RandomState instance. By default ``None``


    Examples
    --------
    >>> import numpy as np
    >>> from mapie.subsample import Subsample
    >>> cv = Subsample(n_resamplings=2,random_state=0)
    >>> X = np.array([1,2,3,4,5,6,7,8,9,10])
    >>> for train_index, test_index in cv.split(X):
    ...    print(f"train index is {train_index}, test index is {test_index}")
    train index is [5 0 3 3 7 9 3 5 2 4], test index is [1 6 8]
    train index is [7 6 8 8 1 6 7 7 8 1], test index is [0 2 3 4 5 9]
    """

    def __init__(
        self,
        n_resamplings: int = 30,
        n_samples: Optional[Union[int, float]] = None,
        replace: bool = True,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        self.n_resamplings = n_resamplings
        self.n_samples = n_samples
        self.replace = replace
        self.random_state = random_state

    def split(
        self, X: NDArray, *args: Any, **kargs: Any
    ) -> Generator[Tuple[NDArray, NDArray], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Training data.

        Yields
        ------
        train : NDArray of shape (n_indices_training,)
            The training set indices for that split.
        test : NDArray of shape (n_indices_test,)
            The testing set indices for that split.
        """
        indices = np.arange(_num_samples(X))
        n_samples = _check_n_samples(X, self.n_samples, indices)
        random_state = check_random_state(self.random_state)
        for k in range(self.n_resamplings):
            train_index = resample(
                indices,
                replace=self.replace,
                n_samples=n_samples,
                random_state=random_state,
                stratify=None,
            )
            test_index = np.setdiff1d(indices, train_index)
            yield train_index, test_index

    def get_n_splits(self, *args: Any, **kargs: Any) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.

        Returns
        -------
        int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_resamplings


class BlockBootstrap(BaseCrossValidator):  # type: ignore
    """
    Generate a sampling method, that block bootstraps the training set.
    It can replace KFold, LeaveOneOut or SubSample as cv argument in the
    TimeSeriesRegressor class.

    Parameters
    ----------
    n_resamplings : int
        Number of resamplings. By default ``30``.
    length: int
        Length of the blocks. By default ``None``,
        the length of the training set divided by ``n_blocks``.
    overlapping: bool
        Whether the blocks can overlap or not. By default ``False``.
    n_blocks: int
        Number of blocks in each resampling. By default ``None``,
        the size of the training set divided by ``length``.
    random_state: Optional
        int or RandomState instance.

    Raises
    ------
    ValueError
        If both ``length`` and ``n_blocks`` are ``None``.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.subsample import BlockBootstrap
    >>> cv = BlockBootstrap(n_resamplings=2, length=3, random_state=0)
    >>> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> for train_index, test_index in cv.split(X):
    ...    print(f"train index is {train_index}, test index is {test_index}")
    train index is [1 2 3 4 5 6 1 2 3 4 5 6], test index is [8 9 7]
    train index is [4 5 6 7 8 9 1 2 3 7 8 9], test index is []
    """

    def __init__(
        self,
        n_resamplings: int = 30,
        length: Optional[int] = None,
        n_blocks: Optional[int] = None,
        overlapping: bool = False,
        random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        self.n_resamplings = n_resamplings
        self.length = length
        self.n_blocks = n_blocks
        self.overlapping = overlapping
        self.random_state = random_state

    def split(
        self, X: NDArray, *args: Any, **kargs: Any
    ) -> Generator[Tuple[NDArray, NDArray], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            Training data.

        Yields
        ------
        train : NDArray of shape (n_indices_training,)
            The training set indices for that split.
        test : NDArray of shape (n_indices_test,)
            The testing set indices for that split.

        Raises
        ------
        ValueError
            If ``length`` is not positive or greater than the train set size.
        """
        if (self.n_blocks is not None) + (self.length is not None) != 1:
            raise ValueError(
                "Exactly one argument between ``length`` or "
                "``n_blocks`` has to be not None"
            )

        n = len(X)

        if self.n_blocks is not None:
            length = (
                self.length if self.length is not None else n // self.n_blocks
            )
            n_blocks = self.n_blocks
        else:
            length = cast(int, self.length)
            n_blocks = (n // length) + 1

        indices = np.arange(n)
        if (length <= 0) or (length > n):
            raise ValueError(
                "The length of blocks is <= 0 or greater than the length"
                "of training set."
            )

        if self.overlapping:
            blocks = sliding_window_view(indices, window_shape=length)
        else:
            indices = indices[(n % length):]
            blocks_number = n // length
            blocks = np.asarray(
                np.array_split(indices, indices_or_sections=blocks_number)
            )

        random_state = check_random_state(self.random_state)

        for k in range(self.n_resamplings):
            block_indices = resample(
                range(len(blocks)),
                replace=True,
                n_samples=n_blocks,
                random_state=random_state,
                stratify=None,
            )
            train_index = np.concatenate(
                [blocks[k] for k in block_indices], axis=0
            )
            test_index = np.array(
                list(set(indices) - set(train_index)), dtype=np.int64
            )
            yield train_index, test_index

    def get_n_splits(self, *args: Any, **kargs: Any) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.

        Returns
        -------
        int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_resamplings
