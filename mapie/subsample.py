from __future__ import annotations
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import resample
from sklearn.utils.validation import _num_samples

from ._typing import ArrayLike


class Subsample(BaseCrossValidator):  # type: ignore
    """
    Generate a sampling method, that resamples the training set with
    possible bootstraps. It can replace KFold or  LeaveOneOut as cv argument
    in the MAPIE class

    Parameters
    ----------
    n_resamplings : int
        Number of resamplings
    n_samples: int
        Number of samples in each resampling. By default None,
        the size of the training set
    replace: bool
        Whether to replace samples in resamplings or not
    random_states: Optional
        List to fix random states


    Examples
    --------
    >>> import pandas as pd
    >>> from mapie.subsample import Subsample
    >>> cv = Subsample(n_resamplings=2,random_states=[0,1])
    >>> X = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9,10]))
    >>> for train_index, test_index in cv.split(X):
    ...    print(f"train index is {train_index}, test index is {test_index}")
    train index is [5 0 3 3 7 9 3 5 2 4], test index is [8 1 6]
    train index is [5 8 9 5 0 0 1 7 6 9], test index is [2 3 4]
    """

    def __init__(
        self,
        n_resamplings: int,
        n_samples: Optional[int] = None,
        replace: bool = True,
        random_states: Optional[List[int]] = None,
    ) -> None:

        self.check_parameters_Subsample(
            random_states=random_states, n_resamplings=n_resamplings,
        )
        self.n_resamplings = n_resamplings
        self.n_samples = n_samples
        self.replace = replace
        self.random_states = random_states

    def check_parameters_Subsample(
        self, random_states: Optional[List[int]], n_resamplings: int,
    ) -> None:
        if (random_states is not None) and (
            len(random_states) != n_resamplings
        ):
            raise ValueError("Incoherent number of random states")

    def split(
        self, X: ArrayLike
    ) -> Generator[Tuple[Any, ArrayLike], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : ArrayLike of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : ArrayLike of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ArrayLike
            The training set indices for that split.
        test : ArrayLike
            The testing set indices for that split.
        """
        indices = np.arange(_num_samples(X))
        n_samples = (
            self.n_samples if self.n_samples is not None else len(indices)
        )

        for k in range(self.n_resamplings):
            if self.random_states is None:
                rnd_state = None
            else:
                rnd_state = self.random_states[k]
            train_index = resample(
                indices,
                replace=self.replace,
                n_samples=n_samples,
                random_state=rnd_state,
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

        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility with BaseCrossValidator
            object.
        y : object
            Always ignored, exists for compatibility with BaseCrossValidator
            object.
        groups : object
            Always ignored, exists for compatibility with BaseCrossValidator
            object.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_resamplings
