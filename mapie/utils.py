from typing import Tuple, Optional, Union, Iterable, Any, cast
from inspect import signature

import numpy as np
from sklearn.utils.validation import _check_sample_weight
from sklearn.base import RegressorMixin, ClassifierMixin

from ._typing import ArrayLike


def check_null_weight(
    sample_weight: ArrayLike,
    X: ArrayLike,
    y: ArrayLike
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
    sample_weight: Optional[ArrayLike] = None
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
        raise ValueError(
            "Invalid alpha. Allowed values are float or Iterable."
        )
    if len(alpha_np.shape) != 1:
        raise ValueError(
            "Invalid alpha. "
            "Please provide a one-dimensional list of values."
        )
    if alpha_np.dtype.type not in [np.float64, np.float32]:
        raise ValueError(
            "Invalid alpha. Allowed values are Iterable of floats."
        )
    if np.any((alpha_np <= 0) | (alpha_np >= 1)):
        raise ValueError(
            "Invalid alpha. Allowed values are between 0 and 1."
        )
    return alpha_np


def check_n_features_in(
    X: ArrayLike,
    cv: Optional[Union[float, str]] = None,
    estimator: Optional[Union[RegressorMixin, ClassifierMixin]] = None
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
                "Invalid mismatch between "
                "X.shape and estimator.n_features_in_."
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
        1/alpha (or 1/(1 - alpha)) must be lower than the number of samples.
    """
    for alpha in alphas:
        if n < 1/alpha or n < 1/(1 - alpha):
            raise ValueError(
                    "Number of samples of the score is too low,"
                    " 1/alpha (or 1/(1 - alpha)) must be lower "
                    "than the number of samples."
                )


def check_n_jobs(n_jobs: Optional[int] = None) -> None:
    """
    Check parameter ``n_jobs``.

    Raises
    ------
    ValueError
        If parameter is not valid.
    """
    if not isinstance(n_jobs, (int, type(None))):
        raise ValueError("Invalid n_jobs argument. Must be an integer.")

    if n_jobs == 0:
        raise ValueError("Invalid n_jobs argument. Must be different than 0.")


def check_verbose(verbose: int) -> None:
    """
    Check parameter ``verbose``.

    Raises
    ------
    ValueError
        If parameter is not valid.
    """
    if not isinstance(verbose, int):
        raise ValueError("Invalid verbose argument. Must be an integer.")

    if verbose < 0:
        raise ValueError("Invalid verbose argument. Must be non-negative.")
