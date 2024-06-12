from __future__ import annotations

import inspect
import numpy as np
from typing import (Iterable, Callable, Optional, Tuple, Union,
                    cast, Dict, List)
from mapie._typing import ArrayLike, NDArray
import warnings
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples


def format_functions(
    functions: Optional[Union[Callable, Iterable[Callable]]],
    bias: bool,
) -> List[Callable]:
    """
    Validate functions for required and optional arguments.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable[Callable]]]
        List of functions (or CCP objects) or single function.
        Each function can take a combinaison of the following arguments:
        - ``X``: Input dataset, of shape (n_samples, ``n_in``)
        - ``y_pred``: estimator prediction, of shape (n_samples,)
        - ``z``: exogenous variable, of shape (n_samples, n_features).
            It should be given in the ``fit`` and ``predict`` methods.
        The results of each functions will be concatenated to build the final
        result of the phi function, of shape (n_samples, ``n_out``).
        If ``None``, the resulting phi object will return a column of ones,
        when called. It will result, in the MapieCCPRegressor, in a basic
        split CP approach.

    bias: bool
        Add a column of ones to the features, for safety reason
        (to garanty the marginal coverage, no matter how the other features
        the ``CCP``object were built).
        If the ``CCP``object definition covers all the dataset
        (meaning, for all calibration and test samples, ``phi(X, y_pred, z)``
        is never all zeros), this column of ones is not necessary
        to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        Note: Even if it is not always necessary to guarantee the marginal
        coverage, it can't degrade the prediction intervals.

    Returns
    -------
    List[Callable]
        ``functions`` as a not empty list
    """
    if functions is None:
        functions = []
    elif isinstance(functions, Iterable):
        functions = list(functions)
    else:
        functions = [functions]

    if bias:
        functions.append(lambda X: np.ones((_num_samples(X), 1)))
    if (len(functions) == 0):
        raise ValueError("You need to define the `functions` argument "
                         "with a function or a list of functions, "
                         "or keep bias argument to True.")
    return functions


def compile_functions_warnings_errors(
    functions: List[Callable]
) -> None:
    """
    Raise warnings and errors if the elements in ``functions`` have
    unexpected arguments.

    Raises
    ------
    ValueError
        If no functions are provided and `bias` is False.
        If functions contain unknown required arguments.

    Warns
    -----
    UserWarning
        If functions contain unknown optional arguments.

    Notes
    -----
    This method ensures that the provided functions only use recognized
    arguments ('X', 'y_pred', 'z'). Unknown optional arguments are allowed,
    but will always use their default values.
    """

    warn_ind: Dict[str, List[int]] = {}
    error_ind: Dict[str, List[int]] = {}
    for i, funct in enumerate(functions):
        assert callable(funct)
        params = inspect.signature(funct).parameters

        for param, arg in params.items():
            if (
                param not in ["X", "y_pred", "z"]
                and param != "disable_marginal_guarantee"
            ):
                if arg.default is inspect.Parameter.empty:
                    if param in error_ind:
                        error_ind[param].append(i)
                    else:
                        error_ind[param] = [i]
                else:
                    if param in warn_ind:
                        warn_ind[param].append(i)
                    else:
                        warn_ind[param] = [i]

    if len(warn_ind) > 0:
        warn_msg = ""
        for param, inds in warn_ind.items():
            warn_msg += (
                f"The functions at index ({', '.join(map(str, inds))}) "
                + "of the 'functions' argument, has an unknown optional "
                + f"argument '{param}'.\n"
            )
        warnings.warn(
            "WARNING: Unknown optional arguments.\n"
            + warn_msg +
            "The only recognized arguments are : 'X', 'y_pred' and 'z'. "
            "The other optional arguments will act as parameters, "
            "as it is always their default value which will be used."
        )
    if len(error_ind) > 0:
        error_msg = ""
        for param, inds in error_ind.items():
            error_msg += (
                f"The functions at index ({', '.join(map(str, inds))}) "
                + "of the 'functions' argument, has an unknown required "
                + f"argument '{param}'.\n"
            )
        raise ValueError(
            "Forbidden required argument.\n"
            f"{error_msg}"
            "The only allowed required argument are : 'X', "
            "'y_pred' and 'z'.\n"
            "Note: You can use optional arguments if you want "
            "to. They will act as parameters, as it is always "
            "their default value which will be used."
        )


def sample_points(
    X: ArrayLike,
    points: Optional[Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]]
) -> NDArray:
    """
    Generate the ``points_`` attribute from the ``points`` and ``X`` arguments

    Parameters
    ----------
    X : ArrayLike
        Samples
    points : Optional[Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]]
        If Array: List of data points, used as centers to compute
        gaussian distances. Should be an array of shape (n_points, n_in).

        If integer, the points will be sampled randomly from the ``X``
        set, where ``X`` is the data give to the
        ``GaussianCCP.fit`` method, which usually correspond to
        the ``X`` argument of the ``MapieCCPRegressor.calibrate`` method
        (unless you call ``GaussianCCP.fit(X)`` yourself).

        You can pass a Tuple[ArrayLike, ArrayLike], to have a different
        ``sigma`` value for each point. The two elements of the
        tuple should be:
         - Data points: 2D array of shape (n_points, n_in)
         - Sigma values 2D array of shape (n_points, n_in) or (n_points, 1)
        In this case, the ``sigma``, ``random_sigma`` and ``X`` argument are
        ignored.

        If ``None``, default to ``20``.

    Returns
    -------
    points_
        2D NDArray of points

    Raises
    ------
    ValueError
        If ``points`` is an invalid argument.
    """
    if points is None:
        points = 20
    if isinstance(points, int):
        points_index = np.random.choice(
            _num_samples(X), size=points, replace=False
        )
        points_ = _safe_indexing(X, points_index)
    elif isinstance(points, tuple):
        points_ = np.array(points[0])
    elif len(np.array(points).shape) == 2:
        points_ = np.array(points)
    else:
        raise ValueError("Invalid `points` argument. The points argument"
                         "should be an integer, "
                         "a 2D array or a tuple of two 2D arrays.")
    return points_


def compute_sigma(
    X: ArrayLike,
    points: Optional[Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]],
    points_: NDArray,
    sigma: Optional[Union[float, ArrayLike]],
    random_sigma: bool,
) -> NDArray:
    """
    Generate the ``sigmas_`` attribute from the ``points``, ``sigma``, ``X``
    arguments and the fitted ``points_``.

    Parameters
    ----------
    X : ArrayLike
        Samples

    points : Optional[Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]]
        If Array: List of data points, used as centers to compute
        gaussian distances. Should be an array of shape (n_points, n_in).

        If integer, the points will be sampled randomly from the ``X``
        set, where ``X`` is the data give to the
        ``GaussianCCP.fit`` method, which usually correspond to
        the ``X`` argument of the ``MapieCCPRegressor.calibrate`` method
        (unless you call ``GaussianCCP.fit(X)`` yourself).

        You can pass a Tuple[ArrayLike, ArrayLike], to have a different
        ``sigma`` value for each point. The two elements of the
        tuple should be:
         - Data points: 2D array of shape (n_points, n_in)
         - Sigma values 2D array of shape (n_points, n_in) or (n_points, 1)
        In this case, the ``sigma``, ``random_sigma`` and ``X`` argument are
        ignored.

        If ``None``, default to ``20``.

    points_ : NDArray
        Fitted 2D arrray of points

    sigma : Optional[Union[float, ArrayLike]]
        Standard deviation value used to compute the guassian distances,
        with the formula:
        np.exp(-0.5 * ((X - point) / ``sigma``) ** 2)
         - It can be an integer
         - It can be a 1D array of float with as many
        values as dimensions in the dataset

        If you want different standard deviation values of each points,
        you can indicate the sigma value of each point in the ``points``
        argument.

        If ``None``, ``sigma`` will default to a float equal to
        ``np.std(X)/(n**0.5)``, where ``X`` is the data give to the
        ``GaussianCCP.fit`` method, which correspond to the ``X``
        argument of the ``MapieCCPRegressor.calibrate`` method
        (unless you call ``GaussianCCP.fit(X)`` yourself).

    random_sigma : bool
        Whether to apply to the standard deviation values, a random multiplier,
        different for each point, equal to:

        2**np.random.normal(0, 1*2**(-2+np.log10(len(``points``))))

        Exemple:
         - For 10 points, the sigma value will, in general,
        be multiplied by a value between 0.7 and 1.4
         - For 100 points, the sigma value will, in general,
        be multiplied by a value between 0.5 and 2


    Returns
    -------
    sigmas_
        2D NDArray of standard deviation values
    """
    # If each point has a corresponding sigma value
    if isinstance(points, tuple):
        sigmas_ = np.array(points[1], dtype=float)
        if len(sigmas_.shape) == 1:
            sigmas_ = sigmas_.reshape(-1, 1)
    # If sigma is not defined
    elif sigma is None:
        points_std = np.std(
            np.array(X), axis=0)/(_num_samples(points_)**0.5)
        sigmas_ = np.ones((_num_samples(points_), 1))*points_std
    # If sigma is defined
    elif isinstance(points, int):
        sigmas_ = _init_sigmas(sigma, points)
    else:
        sigmas_ = _init_sigmas(sigma, _num_samples(points))

    if random_sigma:
        n = _num_samples(points_)
        sigmas_ *= (
            2**np.random.normal(0, 1*2**(-2+np.log10(n)), n)
            .reshape(-1, 1)
        )
    return cast(NDArray, sigmas_)


def _init_sigmas(
    sigma: Union[float, ArrayLike],
    n_points: int,
) -> NDArray:
    """
    If ``sigma`` is not ``None``, take a sigma value, and set ``sigmas_``
    to a standard deviation 2D array of shape (n_points, n_sigma),
    n_sigma being 1 or the number of dimensions of X.

    Parameters
    ----------
    sigma : Union[float, ArrayLike]
        standard deviation, as float or 1D array of length n_in
        (number of dimensins of the dataset)

    n_points : int
        Number of points user for gaussian distances calculation

    Raises
    ------
    ValueError
        If ``sigma`` is not None, a float or a 1D array
    """
    if isinstance(sigma, (float, int)):
        return np.ones((n_points, 1))*sigma
    else:
        if len(np.array(sigma).shape) != 1:
            raise ValueError("sigma argument should be a float "
                             "or a 1D array of floats.")
        return np.ones((n_points, 1))*np.array(sigma)


def dynamic_arguments_call(f: Callable, params_mapping: Dict) -> NDArray:
    """
    Call the function ``f``, with the correct arguments

    Parameters
    ----------
    f : Callable
        function to call

    params_mapping : Dict
        Dictionnary of argument names / values

    Returns
    -------
    NDArray
        result as 2D array
    """

    params = inspect.signature(f).parameters
    used_params = {
        p: params_mapping[p] for p in params
        if p in params_mapping and params_mapping[p] is not None
    }
    res = np.array(f(**used_params), dtype=float)
    if len(res.shape) == 1:
        res = np.expand_dims(res, axis=1)

    return res


def concatenate_functions(
    functions: List[Callable], params_mapping: Dict,
    multipliers: Optional[List[Callable]]
) -> NDArray:
    """
    Call the function of ``functions``, with the
    correct arguments, and concatenate the results

    Parameters
    ----------
    functions : List[Callable]
        List of functions to call

    params_mapping : Dict
        Dictionnary of argument names / values

    Returns
    -------
    NDArray
        Concatenated result
    """
    # Compute phi(X, y_pred, z)
    result = np.hstack([
        dynamic_arguments_call(f, params_mapping) for f in functions
    ])
    # Multiply the result by each multiplier function
    if multipliers is not None:
        for f in multipliers:
            result *= dynamic_arguments_call(f, params_mapping)
    return result


def check_multiplier(
    multipliers: Optional[List[Callable]],
    X: Optional[ArrayLike] = None,
    y_pred: Optional[ArrayLike] = None,
    z: Optional[ArrayLike] = None,
) -> None:
    """
    Check is ``funct`` is a valid ``multiplier`` argument

    Parameters
    ----------
    multipliers : List[Callable]
        function which sould return an array of shape (n_samples, 1) or
        (n_samples, )

    X : ArrayLike
            Observed samples

    y_pred : ArrayLike
        Target prediction

    z : ArrayLike
        Exogenous variable
    """
    if multipliers is None:
        return
    params_mapping = {"X": X, "y_pred": y_pred, "z": z}
    for f in multipliers:
        res = dynamic_arguments_call(f, params_mapping)
        if res.shape != (_num_samples(X), 1):
            raise ValueError("The function used as multiplier should return an"
                             "array of shape n_samples, 1) or (n_samples, ).\n"
                             f"Got shape = {res.shape}.")


def fast_mean_pinball_loss(
    y_true, y_pred, *, sample_weight=None, alpha=0.5
) -> float:
    """
    Pinball loss for quantile regression.
    Copy of the sklearn.metric.mean_minball_loss, but without the checks on
    the ``y_true`` and ``y_pred`` arrays, for faster computation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    alpha : float, slope of the pinball loss, default=0.5,
        This loss is equivalent to :ref:`mean_absolute_error` when `alpha=0.5`,
        `alpha=0.95` is minimized by estimators of the 95th percentile.

    Returns
    -------
    loss : float
        Weighted average of all output errors.
        The pinball loss output is a non-negative floating point. The best
        value is 0.0.

    Examples
    --------
    >>> from sklearn.metrics import mean_pinball_loss
    >>> y_true = [1, 2, 3]
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.1)
    0.03...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.1)
    0.3...
    >>> mean_pinball_loss(y_true, [0, 2, 3], alpha=0.9)
    0.3...
    >>> mean_pinball_loss(y_true, [1, 2, 4], alpha=0.9)
    0.03...
    >>> mean_pinball_loss(y_true, y_true, alpha=0.1)
    0.0
    >>> mean_pinball_loss(y_true, y_true, alpha=0.9)
    0.0
    """
    diff = y_true - y_pred
    sign = (diff >= 0).astype(diff.dtype)
    loss = alpha * sign * diff - (1 - alpha) * (1 - sign) * diff
    output_errors = np.average(loss, weights=sample_weight, axis=0)

    return np.mean(output_errors)


def calibrator_optim_objective(
    beta: NDArray, phi_x: NDArray, conformity_scores: NDArray, q: float,
    sample_weight: NDArray,
) -> float:
    """
    Objective funtcion to minimize to get the estimation of
    the conformity scores ``q`` quantile, caracterized by
    the scalar parameters in the ``beta`` vector.

    Parameters
    ----------
    beta : NDArray
        Parameters to optimize to minimize the objective function

    phi_x : NDArray
        Transformation of the data X using the ``CCP``.

    conformity_scores : NDArray
        Conformity scores of X

    q : float
        Between ``0.0`` and ``1.0``, represents the quantile, being
        ``1-alpha`` if ``alpha`` is the risk level of the confidence interval.

    sample_weight: Optional[ArrayLike] of shape (n_samples,)
        Sample weights for fitting the out-of-fold models.
        If ``None``, then samples are equally weighted.
        If some weights are null,
        their corresponding observations are removed
        before the fitting process and hence have no residuals.
        If weights are non-uniform, residuals are still uniformly weighted.
        Note that the sample weight defined are only for the training, not
        for the calibration procedure.

        By default ``None``.

    Returns
    -------
    float
        Scalar value to minimize, being the sum of the pinball losses.
    """
    return fast_mean_pinball_loss(
        y_true=conformity_scores, y_pred=phi_x.dot(beta),
        alpha=q, sample_weight=sample_weight,
    )
