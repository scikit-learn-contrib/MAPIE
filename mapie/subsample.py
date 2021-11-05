from __future__ import annotations

from typing import Any, Generator, Optional, Tuple, Union

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
        Number of samples in each resampling. By default ``None``,
        the size of the training set.
    replace: bool
        Whether to replace samples in resamplings or not.
    random_state: Optional
        int or RandomState instance.


    Examples
    --------
    >>> import numpy as np
    >>> from mapie.subsample import Subsample
    >>> cv = Subsample(n_resamplings=2,random_state=0)
    >>> X = np.array([1,2,3,4,5,6,7,8,9,10])
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

    def get_n_splits(self, *args: Any, **kargs: Any) -> int:
        """
        Returns the number of splitting iterations in the cross-validator.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_resamplings
