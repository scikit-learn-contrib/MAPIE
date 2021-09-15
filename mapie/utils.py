from os import replace
from typing import (
    Tuple,
    List,
    TypeVar,
    Optional,
    Union,
    Iterable,
    Any,
    cast,
    Type,
    Generator,
    Sequence,
)
from inspect import signature

import numpy as np
from scipy.sparse.construct import rand, random

from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import resample
from sklearn.utils.validation import _num_samples
from sklearn.base import RegressorMixin, ClassifierMixin

from ._typing import ArrayLike


class ReSampling:
    def __init__(
        self,
        n_resamplings: int,
        n_samples: Optional[Union[Type[None], int]] = None,
        bootstrap: bool = True,
        random_states: Optional[List[int]] = None,
    ) -> None:
        self.n_resamplings = n_resamplings
        self.n_samples = n_samples
        self.boostrap = bootstrap
        if (random_states is not None) and (len(random_states) != n_samples):
            raise ValueError("Incoherent number of random states")
        else:
            self.random_states = random_states

    def split(self, X: ArrayLike) -> Generator[Tuple[Any, ArrayLike], None, None]:
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        indices = np.arange(_num_samples(X))
        n_samples = self.n_samples if self.n_samples is not None else len(indices)

        for k in range(self.n_resamplings):
            if self.random_states is None:
                rnd_state = None
            else:
                rnd_state = self.random_states[k]
            train_index = resample(
                indices,
                replace=self.boostrap,
                n_samples=n_samples,
                random_state=rnd_state,
                stratify=None,
            )
            test_index = np.array(list(set(indices) - set(train_index)), dtype=np.int64)
            yield train_index, test_index


def check_null_weight(
    sample_weight: ArrayLike, X: ArrayLike, y: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Check sample weights and remove samples with null sample weights.

    Parameters
    ----------
    sample_weight : ArrayLike
        Sample weights.
    X : ArrayLike of shape (n_samples, n_features)
        Training samples.
    y : ArrayLike of shape (n_samples,)
        Training labels.

    Returns
    -------
    sample_weight : ArrayLike of shape (n_samples,)
        Non-null sample weights.

    X : ArrayLike of shape (n_samples, n_features)
        Training samples with non-null weights.

    y : ArrayLike of shape (n_samples,)
        Training labels with non-null weights.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.utils import check_null_weight
    >>> X = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    >>> y = np.array([5, 7, 9, 11, 13, 15])
    >>> sample_weight = np.array([0, 1, 1, 1, 1, 1])
    >>> sample_weight, X, y = check_null_weight(sample_weight, X, y)
    >>> print(sample_weight)
    [1. 1. 1. 1. 1.]
    >>> print(X)
    [[1]
     [2]
     [3]
     [4]
     [5]]
    >>> print(y)
    [ 7  9 11 13 15]
    """
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        non_null_weight = sample_weight != 0
        X, y = X[non_null_weight, :], y[non_null_weight]
        sample_weight = sample_weight[non_null_weight]
    return sample_weight, X, y


def fit_estimator(
    estimator: RegressorMixin,
    X: ArrayLike,
    y: ArrayLike,
    sample_weight: Optional[ArrayLike] = None,
) -> RegressorMixin:
    """
    Fit an estimator on training data by distinguishing two cases:
    - the estimator supports sample weights and sample weights are provided.
    - the estimator does not support samples weights or
      samples weights are not provided

    Parameters
    ----------
    estimator : RegressorMixin
        Estimator to train.

    X : ArrayLike of shape (n_samples, n_features)
        Input data.

    y : ArrayLike of shape (n_samples,)
        Input labels.

    sample_weight : Optional[ArrayLike] of shape (n_samples,)
        Sample weights. If None, then samples are equally weighted.
        By default None.

    Returns
    -------
    RegressorMixin
        Fitted estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.utils.validation import check_is_fitted
    >>> X = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    >>> y = np.array([5, 7, 9, 11, 13, 15])
    >>> estimator = LinearRegression()
    >>> estimator = fit_estimator(estimator, X, y)
    >>> check_is_fitted(estimator)
    """
    fit_parameters = signature(estimator.fit).parameters
    supports_sw = "sample_weight" in fit_parameters
    if supports_sw and sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


def check_alpha(
    alpha: Optional[Union[float, Iterable[float]]] = None
) -> Optional[np.ndarray]:
    """
    Check alpha and prepare it as a np.ndarray

    Parameters
    ----------
    alpha : Union[float, Iterable[float]]
        Can be a float, a list of floats, or a np.ndarray of floats.
        Between 0 and 1, represent the uncertainty of the confidence interval.
        Lower alpha produce larger (more conservative) prediction intervals.
        alpha is the complement of the target coverage level.
        Only used at prediction time. By default 0.1.

    Returns
    -------
    np.ndarray
        Prepared alpha.

    Raises
    ------
    ValueError
        If alpha is not a float or an Iterable of floats between 0 and 1.
    """
    if alpha is None:
        return alpha
    if isinstance(alpha, float):
        alpha_np = np.array([alpha])
    elif isinstance(alpha, Iterable):
        alpha_np = np.array(alpha)
    else:
        raise ValueError("Invalid alpha. Allowed values are float or Iterable.")
    if len(alpha_np.shape) != 1:
        raise ValueError(
            "Invalid alpha. " "Please provide a one-dimensional list of values."
        )
    if alpha_np.dtype.type not in [np.float64, np.float32]:
        raise ValueError("Invalid alpha. Allowed values are Iterable of floats.")
    if np.any((alpha_np <= 0) | (alpha_np >= 1)):
        raise ValueError("Invalid alpha. Allowed values are between 0 and 1.")
    return alpha_np


def check_n_features_in(
    X: ArrayLike,
    cv: Optional[Union[float, str, ReSampling]] = None,
    estimator: Optional[Union[RegressorMixin, ClassifierMixin]] = None,
) -> int:
    """
    Check the expected number of training features.
    In general it is simply the number of columns in the data.
    If ``cv=="prefit"`` however,
    it can be deduced from the estimator's ``n_features_in_`` attribute.
    These two values absolutely must coincide.

    Parameters
    ----------
    cv : Optional[Union[float, str]]
        The cross-validation strategy for computing scores,
        by default ``None``.

    X : ArrayLike of shape (n_samples, n_features)
        Data passed into the ``fit`` method.

    estimator : RegressorMixin
        Backend estimator of MAPIE.

    Returns
    -------
    int
        Expected number of training features.

    Raises
    ------
    ValueError
        If there is an inconsistency between the shape of the dataset
        and the one expected by the estimator.
    """
    n_features_in: int = X.shape[1]
    if cv == "prefit" and hasattr(estimator, "n_features_in_"):
        if cast(Any, estimator).n_features_in_ != n_features_in:
            raise ValueError(
                "Invalid mismatch between " "X.shape and estimator.n_features_in_."
            )
    return n_features_in


def check_alpha_and_n_samples(alphas: Iterable[float], n: int) -> None:
    """
    Check if the quantile can be computed based
    on the number of samples and the alpha value.

    Parameters
    ----------
    alphas : Iterable[float]
        Iterable of floats.

    n : int
        number of samples.

    Raises
    ------
    ValueError
        If the number of samples of the score is too low,
        1/alpha (or 1/(1-alpha)) must be lower than the number of samples.
    """
    for alpha in alphas:
        if n < 1 / alpha or n < 1 / (1 - alpha):
            raise ValueError(
                "Number of samples of the score is too low,"
                " 1/alpha (or 1/(1 - alpha)) must be lower "
                "than the number of samples."
            )
