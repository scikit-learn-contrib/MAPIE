import warnings
from inspect import signature
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import resample
from sklearn.utils.validation import _check_sample_weight, _num_samples

from ._typing import ArrayLike


def phi1D(
    x: ArrayLike,
    B: ArrayLike,
    fun: Callable[[ArrayLike], ArrayLike],
) -> ArrayLike:
    """
    The function phi1D is called by phi2D. It aims at multiplying the vector of
    predictions, made by refitted estimators, by a 1-nan matrix specifying, for
    each training sample, if it has to be taken into account by the aggregating
    function, before aggregation

    Parameters
    ----------
    x : ArrayLike
        1D vector
    B : ArrayLike
        2D vector whose number of columns is the length of x
    fun : function
        Vectorized function applying to Arraylike, and that should ignore nan

    Returns
    -------
    phi1D(x, B, fun): ArrayLike
        Each row of B is multiply by x and then the function fun is applied.
        Typically, ``fun`` is a numpy function, with argument ``axis`` set to 1

    """

    return fun(x * B)


def phi2D(
    A: ArrayLike,
    B: ArrayLike,
    fun: Callable[[ArrayLike], ArrayLike],
) -> ArrayLike:
    """
    The function phi2D is a loop along the testing set. For each sample of the
    testing set it applies phi1D to multiply the vector of predictions, made by
    the refitted estimators, by a 1-nan matrix, to compute the aggregated
    predictions ignoring the nans

    Parameters
    ----------
    A : ArrayLike
    B : ArrayLike
        A and B must have the same number of columns
    fun : function
        Vectorized function applying to Arraylike, and that should ignore nan

    Returns
    -------
    phi2D(A, B, fun): ArrayLike
        Apply phi1D(x, B, fun) to each row x of A

    """
    return np.apply_along_axis(
        phi1D,
        axis=1,
        arr=A,
        B=B,
        fun=fun,
    )


class JackknifeAfterBootstrap:
    """
    Generate a sampling method, that resamples the training set with
    possible bootstrap. It can replace KFold as cv argument in the MAPIE
    class

    Parameters
    ----------
    agg_function: str,
        Choose among:
        - "mean"
        - "median"

    n_resamplings : int
        Number of resamplings
    n_samples: int
        Number of samples in each resampling. By default None,
        the size of the training set
    replace: bool
        Wheter to replace samples in resamplings or not
    random_states: Optional
        List to fix random states

    Attributes
    ----------
    split: method
        Equivalent of KFold's split method
    aggregate_fit: method
        Aggregation function to determine aggregated predictions
        on the training set
    aggregation_predict: method
        Aggregation function to determine aggregated predictions
        on the testing set

    Examples
    --------
    >>> import pandas as pd
    >>> from mapie.utils import JackknifeAfterBootstrap
    >>> cv = JackknifeAfterBootstrap(n_resamplings=2,random_states=[0,1])
    >>> X = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9,10]))
    >>> for train_index, test_index in cv.split(X):
    ...    print(f"train index is {train_index}, test index is {test_index}")
    train index is [5 0 3 3 7 9 3 5 2 4], test index is [8 1 6]
    train index is [5 8 9 5 0 0 1 7 6 9], test index is [2 3 4]
    """

    valid_agg_functions_ = ["mean", "median"]

    def __init__(
        self,
        n_resamplings: int,
        agg_function: Optional[str] = "mean",
        n_samples: Optional[int] = None,
        replace: bool = True,
        random_states: Optional[List[int]] = None,
    ) -> None:

        self.check_parameters_JackknifeAfterBoostrap(
            agg_function=agg_function,
            valid_agg_functions=self.valid_agg_functions_,
            random_states=random_states,
            n_resamplings=n_resamplings,
        )
        self.agg_function = agg_function
        self.n_resamplings = n_resamplings
        self.n_samples = n_samples
        self.replace = replace
        self.random_states = random_states

    def check_parameters_JackknifeAfterBoostrap(
        self,
        agg_function: Optional[str],
        valid_agg_functions: List[str],
        random_states: Optional[List[int]],
        n_resamplings: int,
    ) -> None:
        if not (agg_function in valid_agg_functions):
            raise ValueError(
                "Invalid aggregation function. "
                "Allowed values are 'mean', 'median', and in the "
                "last case 'proportiontocut' has to be between 0 and 1"
            )
        if (random_states is not None) and (
            len(random_states) != n_resamplings
        ):
            raise ValueError("Incoherent number of random states")

    def split(
        self, X: ArrayLike
    ) -> Generator[Tuple[Any, ArrayLike], None, None]:
        """
        Generate indices to split data into training and test set.

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

    def aggregate_fit(self, x: ArrayLike) -> ArrayLike:
        """
        Take the array of predictions, made by the refitted estimators,
        on the training set, and aggregate to produce phi-{i}(x_i) for
        each training sample x_i

        Parameters:
        -----------
            x : ArrayLike
            Array of predictions, made by the refitted estimators

        Returns:
        --------
            ArrayLike:
            Array of phi-{i}(x_i) for each training sample x_i
        """
        if self.agg_function == "median":
            return np.nanmedian(x, axis=1)
        else:
            return np.nanmean(x, axis=1)

    def aggregate_predict(self, x: ArrayLike, k: ArrayLike) -> ArrayLike:
        """
        Take the array of predictions, made by the refitted estimators,
        on the testing set, and the 1-nan array indicating for each training
        sample which one to integrate, and aggregate to produce phi-{t}(x_t)
        for each training sample x_t


        Parameters:
        -----------
            x : ArrayLike
                Array of predictions, made by the refitted estimators,
                for each sample of the testing set
            k : ArrayLike
                1-nan array, that indicate whether to integrate the prediction
                of a given estimator into the aggregation, for each training
                sample

        Returns:
        --------
                ArrayLike
                Array of shape (testing set size,) of aggregated predictions
                for each testing  sample

        """
        if self.agg_function == "median":
            return phi2D(A=x, B=k, fun=lambda x: np.nanmedian(x, axis=1))
        else:
            K = np.where(np.isnan(k), 0.0, k)
            return np.matmul(x, (K / (K.sum(axis=1, keepdims=True))).T)


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
        raise ValueError("Invalid alpha. Allowed values are between 0 and 1.")
    return alpha_np


def check_n_features_in(
    X: ArrayLike,
    cv: Optional[Union[float, str, JackknifeAfterBootstrap]] = None,
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
        if n < 1 / alpha or n < 1 / (1 - alpha):
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


def check_nan_in_aposteriori_prediction(X: ArrayLike) -> None:
    """
    Parameters
    ----------
    X : Array of shape (size of training set, number of estimators) whose rows
    are the predictions by each estimator of each training sample.

    Raises
    ------
    Warning
        If the aggregated predictions of any training sample would be nan.
    """

    if np.any(np.all(np.isnan(X), axis=1), axis=0):
        warnings.warn(
            "WARNING: at least one point of training set "
            + "belongs to every resamplings. Increase the "
            + "number of resamplings"
        )
