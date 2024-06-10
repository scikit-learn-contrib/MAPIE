from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Iterable, Callable, Optional, Tuple, Union, List
import warnings

import numpy as np
from mapie._typing import ArrayLike, NDArray
from .utils import (compile_functions_warnings_errors, format_functions,
                    compute_sigma, sample_points, concatenate_functions,
                    check_multiplier)
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples, check_is_fitted
from sklearn.base import BaseEstimator, clone


class PhiFunction(BaseEstimator, metaclass=ABCMeta):
    """
    Base abstract class for the phi functions,
    used in the Gibbs et al. method to model the conformity scores.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable[Callable]]]
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

    bias: bool
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

        By default ``False``.

    normalized: bool
        Whether or not to normalized ``phi(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a interval of zero width for out-of-distribution or
        new samples. On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``False``

    init_value: Optional[ArrayLike]
        Optimization initialisation value.
        If ``None``, is sampled from a normal distribution.

        By default ``None``.

    Attributes
    ----------
    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)
    """

    fit_attributes: List[str] = ["functions_"]

    def __init__(
        self,
        functions: Optional[Union[Callable, Iterable[Callable]]] = None,
        bias: bool = False,
        normalized: bool = False,
        init_value: Optional[ArrayLike] = None,
        multipliers: Optional[List[Callable]] = None,
    ) -> None:
        self.functions = functions
        self.bias = bias
        self.normalized = normalized
        self.init_value = init_value
        self.multipliers = multipliers

    @abstractmethod
    def _check_fit_parameters(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> None:
        """
        Check fit parameters

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """

    def _check_init_value(
        self, init_value: Optional[ArrayLike], n_out: int
    ) -> ArrayLike:
        """
        Set the ``init_value_`` attribute depending on ``init_value`` argument.

        Parameters
        ----------
        init_value : Optional[ArrayLike]
            Optimization initialisation value, set at ``PhiFunction``
            initialisation.
        n_out : int
            Number of dimensions of the ``PhiFunction`` transformation.

        Returns
        -------
        ArrayLike
            Optimization initialisation value
        """
        if init_value is None:
            return np.random.normal(0, 1, n_out)
        else:
            return init_value

    def fit(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> PhiFunction:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected transformation.

        It should set all the attributes of ``fit_attributes``.
        It should also set, once fitted, ``n_in``, ``n_out`` and
        ``init_value``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        self._check_fit_parameters(X, y_pred, z)
        result = self.transform(X, y_pred, z)
        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = len(_safe_indexing(result, 0))
        self.init_value_ = self._check_init_value(self.init_value, self.n_out)
        check_multiplier(self.multipliers, X, y_pred, z)
        return self

    def transform(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Transform ``(X, y_pred, z)`` into an array of shape
        ``(n_samples, n_out)``

        Parameters
        ----------
        X : ArrayLike
            Observed samples

        y_pred : ArrayLike
            Target prediction

        z : ArrayLike
            Exogenous variable

        Returns
        -------
        NDArray
            Transformation
        """
        check_is_fitted(self, self.fit_attributes)

        params_mapping = {"X": X, "y_pred": y_pred, "z": z}
        result = concatenate_functions(self.functions_, params_mapping,
                                       self.multipliers)
        if self.normalized:
            norm = np.linalg.norm(result, axis=1).reshape(-1, 1)
            result[(abs(norm) == 0)[:, 0], :] = np.ones(result.shape[1])

            norm[abs(norm) == 0] = 1
            result /= norm

        if np.any(np.all(result == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation "
                          "phi(X, y_pred, z) is full of zeros. "
                          "It will result in a prediction interval of zero "
                          "width. Consider changing the PhiFunction "
                          "definintion.\nFix: Use `bias=True` "
                          "in the `PhiFunction` definition.")
        return result

    def __call__(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        return self.transform(X, y_pred, z)

    def __mul__(self, funct: Optional[Callable]):
        """
        Multiply a ``PhiFunction`` with another function.
        This other function should return an array of shape (n_samples, 1)
        or (n_samples, )

        Parameters
        ----------
        funct : Optional[Callable]
            function which should return an array of shape (n_samples, 1)
        or (n_samples, )

        Returns
        -------
        PhiFunction
            self, with ``funct`` as a multiplier
        """
        if funct is None:
            return funct
        else:
            compile_functions_warnings_errors([funct])
            new_phi = clone(self)
            if new_phi.multipliers is None:
                new_phi.multipliers = [funct]
            else:
                new_phi.multipliers.append(funct)
            return new_phi

    def __rmul__(self, other):
        return self.__mul__(other)


class CustomPhiFunction(PhiFunction):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``PhiFunction`` object with custom features of
    X, y_pred or z, defined as a list of functions in ``functions`` argument.

    This class can be used to concatenate ``PhiFunction`` instances.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable[Callable]]]
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

    bias: bool
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

        By default ``False``.

    normalized: bool
        Whether or not to normalized the output result. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a interval of zero width for out-of-distribution or
        new samples. On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``False``

    init_value: Optional[ArrayLike]
        Optimization initialisation value.
        If ``None``, is sampled from a normal distribution.

        By default ``None``.

    Attributes
    ----------
    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``).

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.phi_function import CustomPhiFunction
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_pred = np.array([0, 0, 1])
    >>> phi = CustomPhiFunction(
    ...     functions=[
    ...         lambda X: X * (y_pred[:, np.newaxis] == 0), # X, if y_pred is 0
    ...         lambda y_pred: y_pred,                     # y_pred
    ...     ],
    ...     normalized=False,
    ... ).fit(X, y_pred)
    >>> print(phi.transform(X, y_pred))
    [[1. 2. 0.]
     [3. 4. 0.]
     [0. 0. 1.]]
    >>> print(phi.n_out)
    3
    """
    fit_attributes: List[str] = ["is_fitted_"]

    def __init__(
        self,
        functions: Optional[Union[Callable, Iterable[Callable]]] = None,
        bias: bool = False,
        normalized: bool = False,
        init_value: Optional[ArrayLike] = None,
        multipliers: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__(functions, bias, normalized, init_value, multipliers)

    def _check_fit_parameters(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> None:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected transformation.

        It should set all the attributes of ``fit_attributes``.
        It should also set, once fitted, ``n_in``, ``n_out`` and
        ``init_value``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        self.functions_ = format_functions(self.functions, self.bias)
        compile_functions_warnings_errors(self.functions_)

    def fit(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> CustomPhiFunction:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected transformation.

        It should set all the attributes of ``fit_attributes``.
        It should also set, once fitted, ``n_in``, ``n_out`` and
        ``init_value``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        self._check_fit_parameters(X, y_pred, z)

        for phi in self.functions_:
            if isinstance(phi, PhiFunction):
                phi.fit(X)
        self.is_fitted_ = True

        result = self.transform(X, y_pred, z)
        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = len(_safe_indexing(result, 0))
        self.init_value_ = self._check_init_value(self.init_value, self.n_out)
        return self


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

        If ``None``, it will default to ``degree=1``.

        By default ``None``.

    variable: Literal["X", "y_pred", "z"]
        String, used to choose which argument between ``X``, ``y_pred`` and
        ``z`` is used to build the polynomial features.

        By default ``"X"``

    bias: bool
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

        By default ``False``.

    normalized: bool
        Whether or not to normalized ``phi(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a interval of zero width for out-of-distribution or
        new samples. On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``False``

    init_value: Optional[ArrayLike]
        Optimization initialisation value.
        If ``None``, is sampled from a normal distribution.

        By default ``None``.

    Attributes
    ----------
    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)

    exponents: List[int]
        List of exponents of the built polynomial features

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.phi_function import PolynomialPhiFunction
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_pred = np.array([1, 2, 3])
    >>> phi = PolynomialPhiFunction(3).fit(X, y_pred)
    >>> print(phi.transform(X, y_pred))
    [[  1.   2.   1.   4.   1.   8.   1.]
     [  3.   4.   9.  16.  27.  64.   1.]
     [  5.   6.  25.  36. 125. 216.   1.]]
    >>> print(phi.exponents)
    [0, 1, 2, 3]
    >>> phi = PolynomialPhiFunction([1, 2, 5], "y_pred",
    ...                             bias=False).fit(X, y_pred)
    >>> print(phi.transform(X, y_pred))
    [[  1.   1.   1.]
     [  2.   4.  32.]
     [  3.   9. 243.]]
    >>> print(phi.degree)
    [1, 2, 5]
    """
    fit_attributes: List[str] = []

    def __init__(
        self,
        degree: Optional[Union[int, List[int]]] = None,
        variable: str = "X",
        bias: bool = False,
        normalized: bool = False,
        init_value: Optional[ArrayLike] = None,
        multipliers: Optional[List[Callable]] = None,
    ) -> None:
        self.degree = degree
        self.variable = variable
        self.bias = bias
        self.normalized = normalized
        self.init_value = init_value
        self.multipliers = multipliers

    def _convert_degree(
        self, degree: Optional[Union[int, List[int]]], bias: bool
    ) -> Tuple[List[int], bool]:
        """
        Convert ``degree`` argument into a list of exponents

        Parameters
        ----------
        degree: Union[int, List[int]]
            If ``degree``is an integer, it correspond to the degree of the
            polynomial features transformer. It will create the features
            ``1``, ``variable``, ``variable``**2, ...,
            ``variable``**``degree``.

            If ``degree``is an iterable of integers, it will create the
            features ``variable``**d, for all integer d in ``degree``

            ``variable`` may be ``X``, ``y_pred`` or ``z``, depending on the
            ``variable``argument value.

            If ``None``, it will default to ``degree=1``.

            By default ``None``.

        bias: bool
            Add a column of ones to the features, for safety reason
            (to garanty the marginal coverage, no matter how the other features
            the ``PhiFunction``object were built).
            If the ``PhiFunction``object definition covers all the dataset
            (meaning, for all calibration and test samples,
            ``phi(X, y_pred, z)`` is never all zeros), this column of ones
            is not necessary to obtain marginal coverage.
            In this case, you can set this argument to ``False``.

            Note: Even if it is not always necessary to guarantee the marginal
            coverage, it can't degrade the prediction intervals.

        Returns
        -------
        Tuple[List[int], bool]
            - List of exponents (the exponent ``0`` will be replaced by
            ``bias=True``, which is equivalent. It is useless to add as many
            columns of ones as dimensions of ``X``. Only one is enough.)
            - new ``bias`` value.
        """
        if degree is None:
            exponents = [0, 1]
        elif isinstance(degree, int):
            exponents = list(range(degree+1))
        else:
            exponents = degree

        return exponents, (0 in exponents) or bias

    def _create_functions(
        self, exponents: List[int], variable: str
    ) -> List[Callable]:
        """
        Create the list of lambda functions, based on the list ``exponents``
        and the ``variable`` value.

        Parameters
        ----------
        exponents: List[int]
            List of exponents to apply on the ``variable```

        variable: Literal["X", "y_pred", "z"]
            Variable on which to apply the exponents.
        """
        if variable == "X":
            return [lambda X, d=d: X**d for d in exponents if d != 0]
        elif variable == "y_pred":
            return [lambda y_pred, d=d: y_pred**d for d in exponents if d != 0]
        elif variable == "z":
            return [lambda z, d=d: z**d for d in exponents if d != 0]
        else:
            raise ValueError("variable must be 'X', 'y_pred' or 'z'")

    def _check_fit_parameters(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> None:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected transformation.

        It should set all the attributes of ``fit_attributes``.
        It should also set, once fitted, ``n_in``, ``n_out`` and
        ``init_value``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        self.exponents, self.bias = self._convert_degree(
            self.degree, self.bias)
        functions = self._create_functions(self.exponents, self.variable)
        self.functions_ = format_functions(functions, self.bias)


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

        If ``None``, default to ``20``.

        By default, ``None``

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

        If ``None``, default to ``False``.

        By default, ``None``

    bias: bool
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
        be all zeros, so this ``bias`` is not required
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

    init_value: Optional[ArrayLike]
        Optimization initialisation value.
        If ``None``, is sampled from a normal distribution.

        By default ``None``.

    Attributes
    ----------
    fit_attributes: Optional[List[str]]
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
    >>> from mapie.phi_function import GaussianPhiFunction
    >>> np.random.seed(1)
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> phi = GaussianPhiFunction(2, bias=False,
    ...                           normalized=False).fit(X)
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
    >>> phi = GaussianPhiFunction([[3],[4]], 0.5).fit(X)
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
    fit_attributes: List[str] = ["points_", "sigmas_", "functions_"]

    def __init__(
        self,
        points: Optional[Union[int, ArrayLike,
                               Tuple[ArrayLike, ArrayLike]]] = None,
        sigma: Optional[Union[float, ArrayLike]] = None,
        random_sigma: Optional[bool] = None,
        bias: bool = False,
        normalized: bool = True,
        init_value: Optional[ArrayLike] = None,
        multipliers: Optional[List[Callable]] = None,
    ) -> None:
        self.points = points
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.bias = bias
        self.normalized = normalized
        self.init_value = init_value
        self.multipliers = multipliers

    def _check_random_sigma(self) -> bool:
        """
        Check ``random_sigma``

        Returns
        -------
        bool
            checked ``random_sigma``
        """
        if self.random_sigma is None:
            return False
        else:
            return self.random_sigma

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

    def _check_fit_parameters(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> None:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected transformation.

        It should set all the attributes of ``fit_attributes``.
        It should also set, once fitted, ``n_in``, ``n_out`` and
        ``init_value``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        self.random_sigma = self._check_random_sigma()
        self.points_ = sample_points(X, self.points)
        self.sigmas_ = compute_sigma(X, self.points, self.points_,
                                     self.sigma, self.random_sigma)
        self._check_points_sigma(self.points_, self.sigmas_)

        functions = [
            lambda X, mu=_safe_indexing(self.points_, i),
            sigma=_safe_indexing(self.sigmas_, i):
            np.exp(-0.5 * np.sum(((X - mu) / sigma) ** 2, axis=1))
            for i in range(_num_samples(self.points_))
        ]
        self.functions_ = format_functions(functions, self.bias)
