from __future__ import annotations

from abc import ABCMeta, abstractmethod
import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union, cast, Iterable
import numbers
import warnings

import numpy as np
from mapie._typing import ArrayLike, NDArray
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples, check_is_fitted
from sklearn.base import BaseEstimator


def _is_fitted(estimator, attributes=None, all_or_any=all):
    """Determine if an estimator is fitted

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    fitted : bool
        Whether the estimator is fitted.
    """
    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        return all_or_any([
            hasattr(estimator, attr) and getattr(estimator, attr) is not None
            for attr in attributes
        ])

    if hasattr(estimator, "__sklearn_is_fitted__"):
        return estimator.__sklearn_is_fitted__()

    fitted_attrs = [
        v for v in vars(estimator)
        if v.endswith("_") and not v.startswith("__")
        and getattr(estimator, v) is not None
    ]
    return len(fitted_attrs) > 0


class PhiFunction(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for the phi functions,
    used in the Gibbs et al. method to model the conformity scores.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable]]
        List of functions (or PhiFunction objects) or single function.
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

        By default ``None``.

    marginal_guarantee: bool
        Add a column of ones to the features, for safety reason
        (to garanty the marginal coverage, no matter how the other features
        the ``PhiFunction``object were built).
        If the ``PhiFunction``object definition covers all the dataset
        (meaning, for all calibration and test samples, ``phi(X, y_pred, z)``
        is never all zeros), this column of ones is not necessary
        to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        Note: Even if it is not always necessary to guarantee the marginal
        coverage, it can't degrade the prediction intervals.

        By default ``True``.

    normalized: bool
        Whether or not to normalized ``phi(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a interval of zero width for out-of-distribution or
        new samples. On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``False``

    Attributes
    ----------
    fit_attributes: List[str]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)
    """

    fit_attributes: List[str] = []
    output_attributes = ["n_in", "n_out", "init_value"]

    def __init__(
        self,
        functions: Optional[Union[Callable, Iterable]] = None,
        marginal_guarantee: bool = True,
        normalized: bool = False,
    ) -> None:
        self.functions = functions
        self.marginal_guarantee = marginal_guarantee
        self.normalized = normalized

    def _check_transform_parameters(self) -> None:
        """
        Check that ``functions_`` are functions that take as input
        allowed arguments
        """
        self.functions_ = self._check_functions()

    def _check_functions(
            self,
    ) -> NDArray:
        """
        Validate functions for required and optional arguments.

        Raises
        ------
        ValueError
            If no functions are provided and `marginal_guarantee` is False.
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
        if self.functions is None:
            self.functions = cast(NDArray, [])
        elif isinstance(self.functions, Iterable):
            self.functions = cast(NDArray, self.functions)
        else:
            self.functions = cast(NDArray, [self.functions])

        if (len(self.functions) == 0) and not self.marginal_guarantee:
            raise ValueError("You need to define the `functions` argument "
                             "with a function or a list of functions, "
                             "or keep marginal_guarantee argument to True.")

        warn_ind: Dict[str, List[int]] = {}
        error_ind: Dict[str, List[int]] = {}
        for i, funct in enumerate(self.functions):
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
                    elif not isinstance(self, (PolynomialPhiFunction,
                                               GaussianPhiFunction)):
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
        return cast(NDArray, self.functions)

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
    ) -> None:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected transformation.

        It should set all the attributes of ``fit_attributes``

        Parameters
        ----------
        X : Optional[ArrayLike]
            Samples
        """

    def transform(
            self,
            X: Optional[ArrayLike] = None,
            y_pred: Optional[ArrayLike] = None,
            z: Optional[ArrayLike] = None,
            disable_marginal_guarantee: bool = False,
    ) -> NDArray:
        check_is_fitted(self, self.fit_attributes)
        self._check_transform_parameters()

        params_mapping = {"X": X, "y_pred": y_pred, "z": z}
        res = []

        funct_list = list(self.functions_)
        if not disable_marginal_guarantee and self.marginal_guarantee:
            funct_list.append(lambda X: np.ones((len(X), 1)))

        for f in funct_list:
            params = inspect.signature(f).parameters

            used_params = {
                p: params_mapping[p] for p in params
                if p in params_mapping and params_mapping[p] is not None
            }

            res.append(np.array(f(**used_params), dtype=float))

            if len(res[-1].shape) == 1:
                res[-1] = np.expand_dims(res[-1], axis=1)

        result = np.hstack(res)
        if self.normalized:
            norm = np.linalg.norm(result, axis=1).reshape(-1, 1)
            result[(abs(norm) == 0)[:, 0], :] = np.ones(result.shape[1])

            norm[abs(norm) == 0] = 1
            result /= norm

        if not _is_fitted(self, self.output_attributes):
            self.n_in = len(_safe_indexing(X, 0))
            self.n_out = len(_safe_indexing(result, 0))
            self.init_value = np.random.normal(0, 1, self.n_out)

        if np.any(np.all(result == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation "
                          "phi(X, y_pred, z) is full of zeros. "
                          "It will result in a prediction interval of zero "
                          "width. Consider changing the PhiFunction "
                          "definintion.\nFix: Use `marginal_guarantee=True` "
                          "in the `PhiFunction` definition.")
        return result


class CustomPhiFunction(PhiFunction):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``PhiFunction`` object with custom features of
    X, y_pred or z, defined as a list of functions in ``functions`` argument.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable]]
        List of functions (or PhiFunction objects) or single function.
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

        By default ``None``.

    marginal_guarantee: bool
        Add a column of ones to the features, for safety reason
        (to garanty the marginal coverage, no matter how the other features
        the ``PhiFunction``object were built).
        If the ``PhiFunction``object definition covers all the dataset
        (meaning, for all calibration and test samples, ``phi(X, y_pred, z)``
        is never all zeros), this column of ones is not necessary
        to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        Note: Even if it is not always necessary to guarantee the marginal
        coverage, it can't degrade the prediction intervals.

        By default ``True``.

    normalized: bool
        Whether or not to normalized ``phi(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a interval of zero width for out-of-distribution or
        new samples. On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``False``

    Attributes
    ----------
    fit_attributes: List[str]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression.utils import CustomPhiFunction
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_pred = np.array([0, 0, 1])
    >>> phi = CustomPhiFunction(
    ...     functions=[
    ...         lambda X: X * (y_pred[:, np.newaxis] == 0), # X, if y_pred is 0
    ...         lambda y_pred: y_pred,                     # y_pred
    ...     ],
    ...     normalized=False,
    ... )
    >>> print(phi.transform(X, y_pred))
    [[1. 2. 0. 1.]
     [3. 4. 0. 1.]
     [0. 0. 1. 1.]]
    >>> print(phi.n_out)
    4
    """
    fit_attributes = []

    def __init__(
        self,
        functions: Optional[Union[Callable, Iterable]] = None,
        marginal_guarantee: bool = True,
        normalized: bool = False,
    ) -> None:
        super().__init__(functions, marginal_guarantee, normalized)

    def fit(
        self,
        X: ArrayLike,
    ) -> None:
        """
        ``PolynomialPhiFunction`` don't need to be fitted.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Samples
        """
        return


class PolynomialPhiFunction(PhiFunction):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``PhiFunction`` object with polynomial features of
    X, y_pred or z.

    Parameters
    ----------
    degree: Union[int, List[int]]
        If ``degree``is an integer, it correspond to the degree of the
        polynomial features transformer. It will create the features
        ``1``, ``variable``, ``variable``**2, ..., ``variable``**``degree``.

        If ``degree``is an iterable of integers, it will create the features
        ``variable``**d, for all integer d in ``degree``

        ``variable`` may be ``X``, ``y_pred`` or ``z``, depending on the
        ``variable``argument value.

        By default ``1``.

    variable: Literal["X", "y_pred", "z"]
        String, used to choose which argument between ``X``, ``y_pred`` and
        ``z`` is used to build the polynomial features.

        By default ``"X"``

    marginal_guarantee: bool
        Add a column of ones to the features, for safety reason
        (to garanty the marginal coverage, no matter how the other features
        the ``PhiFunction``object were built).
        If the ``PhiFunction``object definition covers all the dataset
        (meaning, for all calibration and test samples, ``phi(X, y_pred, z)``
        is never all zeros), this column of ones is not necessary
        to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        Note: Even if it is not always necessary to guarantee the marginal
        coverage, it can't degrade the prediction intervals.

        By default ``True``.

    normalized: bool
        Whether or not to normalized ``phi(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a interval of zero width for out-of-distribution or
        new samples. On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``False``

    Attributes
    ----------
    fit_attributes: List[str]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)

    degrees: List[int]
        List of degrees of the built polynomial features

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression.utils import PolynomialPhiFunction
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_pred = np.array([1, 2, 3])
    >>> phi = PolynomialPhiFunction(3)
    >>> print(phi.transform(X, y_pred))
    [[  1.   2.   1.   4.   1.   8.   1.]
     [  3.   4.   9.  16.  27.  64.   1.]
     [  5.   6.  25.  36. 125. 216.   1.]]
    >>> print(phi.degrees)
    [0, 1, 2, 3]
    >>> phi = PolynomialPhiFunction([1, 2, 5], "y_pred",
    ...                             marginal_guarantee=False)
    >>> print(phi.transform(X, y_pred))
    [[  1.   1.   1.]
     [  2.   4.  32.]
     [  3.   9. 243.]]
    >>> print(phi.degree)
    [1, 2, 5]
    """
    fit_attributes = []

    def __init__(
        self,
        degree: Union[int, List[int]] = 1,
        variable: str = "X",
        marginal_guarantee: bool = True,
        normalized: bool = False,
    ) -> None:
        self.degree = degree
        self.variable = variable

        if isinstance(degree, int):
            self.degrees = list(range(degree+1))
        else:
            self.degrees = degree

        functions: List[Callable] = []
        if 0 in self.degrees and not marginal_guarantee:
            functions.append(lambda X: np.ones((len(X), 1)))
        if variable == "X":
            functions += [lambda X, d=d: X**d for d in self.degrees if d != 0]
        elif variable == "y_pred":
            functions += [lambda y_pred, d=d: y_pred**d
                          for d in self.degrees if d != 0]
        elif variable == "z":
            functions += [lambda z, d=d: z**d for d in self.degrees if d != 0]
        else:
            raise ValueError("variable must be 'X', 'y_pred' or 'z'")

        super().__init__(functions, marginal_guarantee, normalized)

    def fit(
        self,
        X: ArrayLike,
    ) -> None:
        """
        ``PolynomialPhiFunction`` don't need to be fitted.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Samples
        """
        return


class GaussianPhiFunction(PhiFunction):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``PhiFunction`` object with features been the gaussian
    distances between X and some defined points.

    Parameters
    ----------
    points : Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]
        If Array: List of data points, used as centers to compute
        gaussian distances. Should be an array of shape (n_points, n_in).

        If integer, the points will be sampled randomly from the ``X``
        set, where ``X`` is the data give to the
        ``GaussianPhiFunction.fit`` method, which usually correspond to
        the ``X`` argument of the ``MapieCCPRegressor.calibrate`` method
        (unless you call ``GaussianPhiFunction.fit(X)`` yourself).

        You can pass a Tuple[ArrayLike, ArrayLike], to have a different
        ``sigma`` value for each point. The two elements of the
        tuple should be:
         - Data points: 2D array of shape (n_points, n_in)
         - Sigma values 2D array of shape (n_points, n_in) or (n_points, 1)
        In this case, the ``sigma``, ``random_sigma`` and ``X`` argument are
        ignored.

        By default, ``10``

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
        ``GaussianPhiFunction.fit`` method, which correspond to the ``X``
        argument of the ``MapieCCPRegressor.calibrate`` method
        (unless you call ``GaussianPhiFunction.fit(X)`` yourself).

        By default, ``None``

    random_sigma : bool
        Whether to apply to the standard deviation values, a random multiplier,
        different for each point, equal to:

        2**np.random.normal(0, 1*2**(-2+np.log10(len(``points``))))

        Exemple:
         - For 10 points, the sigma value will, in general,
        be multiplied by a value between 0.7 and 1.4
         - For 100 points, the sigma value will, in general,
        be multiplied by a value between 0.5 and 2

        Note: This is a default suggestion of randomization,
        which allow to have in the same time wide and narrow gaussians
        (with a gigger range of multipliers for huge amount of points).

        You can use fully custom sigma values, buy passing to the
        ``points`` argument, a different sigma value for each point.

        If ``None``, it is enabled if ``sigma`` is not defined (``None``, and
        ``points`` is not a Tuple of (points, sigmas)), disabled otherwise.

        By default, ``None``

    marginal_guarantee: bool
        Add a column of ones to the features, for safety reason
        (to garanty the marginal coverage, no matter how the other features
        the ``PhiFunction``object were built).
        If the ``PhiFunction``object definition covers all the dataset
        (meaning, for all calibration and test samples, ``phi(X, y_pred, z)``
        is never all zeros), this column of ones is not necessary
        to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        Note: In this case, with ``GaussianPhiFunction``, if ``normalized`` is
        ``True`` (it is, by default), the ``phi(X, y_pred, z)`` will never
        be all zeros, so this ``marginal_guarantee`` is not required
        sto have coverage guarantee.

        By default ``False``.

    normalized: bool
        Whether or not to normalized ``phi(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a interval of zero width for out-of-distribution or
        new samples. On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``True``

    Attributes
    ----------
    fit_attributes: List[str]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)

    points_: NDArray
        Array of shape (n_points, n_in), corresponding to the points used to
        compute the gaussian distanes.

    sigmas_: NDArray of shape (len(points), 1) or (len(points), n_in)
        Standard deviation values

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression.utils import GaussianPhiFunction
    >>> np.random.seed(1)
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> phi = GaussianPhiFunction(2, marginal_guarantee=False,
    ...                           normalized=False)
    >>> phi.fit(X)
    >>> print(np.round(phi.transform(X), 2))
    [[0.14 0.61]
     [0.61 1.  ]
     [1.   0.61]
     [0.61 0.14]
     [0.14 0.01]]
    >>> print(phi.points_)
    [[3]
     [2]]
    >>> print(phi.sigmas_)
    [[1.]
     [1.]]
    >>> phi = GaussianPhiFunction([[3],[4]], 0.5)
    >>> print(np.round(phi.transform(X), 2))
    [[1.   0.  ]
     [1.   0.  ]
     [0.99 0.13]
     [0.13 0.99]
     [0.   1.  ]]
    >>> print(phi.points_)
    [[3]
     [4]]
    >>> print(phi.sigmas_)
    [[0.5]
     [0.5]]
    """
    fit_attributes = ["points_", "sigmas_"]

    def __init__(
        self,
        points: Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]] = 20,
        sigma: Optional[Union[float, ArrayLike]] = None,
        random_sigma: Optional[bool] = None,
        marginal_guarantee: bool = False,
        normalized: bool = True,
    ) -> None:
        self.points = points
        self.sigma = sigma
        self.random_sigma = random_sigma

        self.points_: Optional[NDArray]
        self.sigmas_: Optional[NDArray]

        if isinstance(points, int):
            self._init_sigmas(sigma, points)

        elif isinstance(points, tuple):
            self.points_ = np.array(points[0])
            self.sigmas_ = np.array(points[1])
            if len(self.sigmas_.shape) == 1:
                self.sigmas_ = self.sigmas_.reshape(-1, 1)

        elif len(np.array(points).shape) == 2:
            self._init_sigmas(sigma, _num_samples(points))
            self.points_ = cast(NDArray, np.array(points))

        else:
            raise ValueError("Invalid `points` argument. The points argument"
                             "should be an integer, "
                             "a 2D array or a tuple of two 2D arrays.")

        if _is_fitted(self, self.fit_attributes):
            self.sigmas_ = cast(NDArray, self.sigmas_)
            self.points_ = cast(NDArray, self.points_)
            self._check_parameters(self.points_, self.sigmas_)
            if self.random_sigma:
                n = _num_samples(self.points_)
                self.sigmas_ = self.sigmas_ * (
                    2**np.random.normal(0, 1*2**(-2+np.log10(n)), n)
                    .reshape(-1, 1)
                )

            functions = [
                lambda X, mu=_safe_indexing(self.points_, i),
                sigma=_safe_indexing(self.sigmas_, i):
                np.exp(-0.5 * np.sum(((X - mu) / sigma) ** 2, axis=1))
                for i in range(_num_samples(self.points_))
            ]
        else:
            functions = []
        super().__init__(functions, marginal_guarantee, normalized)

    def _check_transform_parameters(self) -> None:
        """
        Check that ``functions_`` are functions that take as input
        allowed arguments
        """
        self.sigmas_ = cast(NDArray, self.sigmas_)
        self.points_ = cast(NDArray, self.points_)

        self._check_parameters(self.points_, self.sigmas_)
        self.functions_ = self._check_functions()

    def _check_parameters(self, points: ArrayLike, sigmas: ArrayLike) -> None:
        """
        Check that ``points`` and ``sigmas`` have compatible shapes

        Parameters
        ----------
        points : ArrayLike
            2D array of shape (n_points, n_in)
        sigmas : ArrayLike
            2D array of shape (n_points, 1) or (n_points, n_in)
        """
        self._check_points_sigma(points, sigmas)
        self.random_sigma = self._check_random_sigma()

    def _check_random_sigma(self) -> bool:
        if self.random_sigma is None and self.sigma is None:
            if isinstance(self.points, tuple):
                return False
            else:
                return True
        if self.random_sigma is None:
            return False
        else:
            return self.random_sigma

    def _init_sigmas(
        self,
        sigma: Optional[Union[float, ArrayLike]],
        n_points: int,
    ) -> None:
        """
        If ``sigma`` is not ``None``, take a sigma value, and set ``sigmas_``
        to a standard deviation 2D array of shape (n_points, n_sigma),
        n_sigma being 1 or the number of dimensions of X.

        Parameters
        ----------
        sigma : Optional[Union[float, ArrayLike]]
            standard deviation, as float or 1D array of length n_in
            (number of dimensins of the dataset)

        n_points : int
            Number of points user for gaussian distances calculation

        Raises
        ------
        ValueError
            If ``sigma`` is not None, a float or a 1D array
        """
        if isinstance(sigma, numbers.Number):
            self.sigmas_ = np.ones((n_points, 1))*sigma
        elif sigma is not None:
            if len(np.array(sigma).shape) != 1:
                raise ValueError("sigma argument should be a float "
                                 "or a 1D array of floats.")
            self.sigmas_ = np.ones((n_points, 1))*np.array(sigma)

    def _check_points_sigma(
        self, points: ArrayLike, sigmas: ArrayLike
    ) -> None:
        """
        Take 2D arrays of points and standard deviations and check
        compatibility

        Parameters
        ----------
        points : ArrayLike
            2D array of shape (n_points, n_in)
        sigmas : ArrayLike
            2D array of shape (n_points, 1) or (n_points, n_in)

        Raises
        ------
        ValueError
            If ``sigmas``is not of shape (n_points, 1) or (n_points, n_in)
        """
        if _num_samples(points) != _num_samples(sigmas):
            raise ValueError("There should have as many points as "
                             "standard deviation values")
        if len(_safe_indexing(sigmas, 0)) not in [
            1, len(_safe_indexing(points, 0))
        ]:
            raise ValueError("The standard deviation 2D array should be of "
                             "shape (n_points, 1) or (n_points, n_in).\n"
                             f"Got sigma of shape: ({_num_samples(sigmas)}, "
                             f"{len(_safe_indexing(points, 0))}).")

    def fit(
        self,
        X: ArrayLike,
    ) -> None:
        """
        ``GaussianPhiFunction`` fit method is used to sample points and compute
        the standard deviation values if needed.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Samples
        """
        if not _is_fitted(self, self.fit_attributes):
            if isinstance(self.points, int):
                points_index = np.random.choice(
                    _num_samples(X), size=self.points, replace=False
                )
                self.points_ = cast(NDArray, _safe_indexing(X, points_index))
            if self.sigma is None:
                self.sigmas_ = np.ones((_num_samples(self.points_), 1))*np.std(
                    np.array(X), axis=0)/(_num_samples(self.points_)**0.5)

            if self.random_sigma:
                n = _num_samples(self.points_)
                self.sigmas_ *= (
                    2**np.random.normal(0, 1*2**(-2+np.log10(n)), n)
                    .reshape(-1, 1)
                )

            self.functions = [
                    lambda X, mu=_safe_indexing(self.points_, i),
                    sigma=_safe_indexing(self.sigmas_, i):
                    np.exp(-0.5 * np.sum(((X - mu) / sigma) ** 2, axis=1))
                    for i in range(_num_samples(self.points_))
            ]
