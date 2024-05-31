from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union, cast
import numbers
import warnings

import numpy as np
from mapie._typing import NDArray
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples


class PhiFunction():
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    Phi takes as input X (and can take y_pred and any exogenous variables z)
    and return an array of shape (n_samples, d), for any integer d.

    Parameters
    ----------
    functions: Optional[Union[
                Union[Callable, "PhiFunction"],
                List[Union[Callable, "PhiFunction"]]
            ]]
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

        By default ``True``

    Attributes
    ----------

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import PhiFunction
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_pred = np.array([0, 0, 1])
    >>> z = np.array([[10], [20], [30]])
    >>> def not_lambda_function(y_pred, z):
    ...     result = np.zeros((y_pred.shape[0], z.shape[1]))
    ...     cnd = (y_pred == 1)
    ...     result[cnd] = z[cnd]
    ...     return result
    >>> phi = PhiFunction(
    ...     functions=[
    ...         lambda X: X * (y_pred[:, np.newaxis] == 0), # X, if y_pred is 0
    ...         lambda y_pred: y_pred,                     # y_pred
    ...         not_lambda_function,                       # z, if y_pred is 1
    ...     ],
    ...     normalized=False,
    ... )
    >>> print(phi(X, y_pred, z))
    [[ 1.  2.  0.  0.  1.]
     [ 3.  4.  0.  0.  1.]
     [ 0.  0.  1. 30.  1.]]
    >>> print(phi.n_out)
    5
    >>> # We can also combine PhiFunction objects with other functions
    >>> compound_phi = PhiFunction(
    ...     functions=[
    ...         phi,
    ...         lambda X: 4 * np.ones((X.shape[0], 1)),
    ...     ],
    ...     normalized=False,
    ... )
    >>> print(compound_phi(X, y_pred, z))
    [[ 1.  2.  0.  0.  4.  1.]
     [ 3.  4.  0.  0.  4.  1.]
     [ 0.  0.  1. 30.  4.  1.]]
    """

    _need_x_calib = False

    def __init__(
            self,
            functions: Optional[Union[
                Union[Callable, "PhiFunction"],
                List[Union[Callable, "PhiFunction"]]
            ]] = None,
            marginal_guarantee: bool = True,
            normalized: bool = True,
    ) -> None:
        if isinstance(functions, list):
            self.functions = list(functions)
        elif functions is not None:
            self.functions = [functions]
        else:
            self.functions = []

        self.marginal_guarantee = marginal_guarantee
        self.normalized = normalized

        self.marginal_guarantee = self.marginal_guarantee or any(
            phi.marginal_guarantee for phi in self.functions
            if isinstance(phi, PhiFunction)
            )

        if not self._need_x_calib:
            self._check_functions(self.functions, self.marginal_guarantee)

        self.n_in: Optional[int] = None
        self.n_out: Optional[int] = None

    def _check_functions(
            self,
            functions: List[Union[Callable, "PhiFunction"]],
            marginal_guarantee: bool,
    ) -> None:
        """
        Validate functions for required and optional arguments.

        Parameters
        ----------
        functions : List[Union[Callable, "PhiFunction"]]
            List of functions or PhiFunction instances to be checked.

        marginal_guarantee : bool
            Flag indicating whether marginal guarantee is enabled.

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
        if len(functions) == 0 and not marginal_guarantee:
            raise ValueError("You need to define the `functions` argument "
                             "with a function or a list of functions, "
                             "or keep marginal_guarantee argument to True.")

        warn_ind: Dict[str, List[int]] = {}
        error_ind: Dict[str, List[int]] = {}
        for i, funct in enumerate(functions):
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

    def __call__(
            self,
            X: Optional[NDArray] = None,
            y_pred: Optional[NDArray] = None,
            z: Optional[NDArray] = None,
            disable_marginal_guarantee: bool = False,
    ) -> NDArray:
        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = 0

        params_mapping = {"X": X, "y_pred": y_pred, "z": z}
        res = []

        funct_list = list(self.functions)
        if not disable_marginal_guarantee and self.marginal_guarantee:
            funct_list.append(lambda X: np.ones((len(X), 1)))

        for f in funct_list:
            params = inspect.signature(f).parameters

            used_params = {
                p: params_mapping[p] for p in params
                if p in params_mapping and params_mapping[p] is not None
            }
            if isinstance(f, PhiFunction):
                # We only consider marginal_guaranty with the main PhiFunction
                res.append(np.array(
                    f(disable_marginal_guarantee=True, **used_params),
                    dtype=float))
            else:
                res.append(np.array(f(**used_params), dtype=float))

            if len(res[-1].shape) == 1:
                res[-1] = np.expand_dims(res[-1], axis=1)

            self.n_out += res[-1].shape[1]

        result = np.hstack(res)
        if self.normalized:
            norm = np.linalg.norm(result, axis=1).reshape(-1, 1)
            norm[abs(norm)<1e-8] = 1
            result /= norm
        return result

    def _check_need_calib(self, X: NDArray) -> None:
        for f in self.functions:
            if isinstance(f, PhiFunction):
                if f._need_x_calib:
                    f._check_need_calib(X)


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
    degree: List[int]
        List of degrees of the built polynomial features

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import PolynomialPhiFunction
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_pred = np.array([1, 2, 3])
    >>> phi = PolynomialPhiFunction(3)
    >>> print(phi(X, y_pred))
    [[  1.   2.   1.   4.   1.   8.   1.]
     [  3.   4.   9.  16.  27.  64.   1.]
     [  5.   6.  25.  36. 125. 216.   1.]]
    >>> print(phi.degree)
    [0, 1, 2, 3]
    >>> phi = PolynomialPhiFunction([1, 2, 5], "y_pred",
    ...                             marginal_guarantee=False)
    >>> print(phi(X, y_pred))
    [[  1.   1.   1.]
     [  2.   4.  32.]
     [  3.   9. 243.]]
    >>> print(phi.degree)
    [1, 2, 5]
    """
    def __init__(
            self,
            degree: Union[int, List[int]] = 1,
            variable: Literal["X", "y_pred", "z"] = "X",
            marginal_guarantee: bool = True,
            normalized: bool = False,
    ) -> None:
        if isinstance(degree, int):
            degree = list(range(degree+1))

        if variable not in ["X", "y_pred", "z"]:
            raise ValueError("variable must be 'X', 'y_pred' or 'z'")

        self.degree = degree

        functions: List[Callable] = []
        if 0 in degree and not marginal_guarantee:
            functions.append(lambda X: np.ones(len(X)))
        if variable == "X":
            functions += [lambda X, d=d: X**d for d in degree if d != 0]
        if variable == "y_pred":
            functions += [lambda y_pred, d=d: y_pred**d
                          for d in degree if d != 0]
        if variable == "z":
            functions += [lambda z, d=d: z**d for d in degree if d != 0]

        super().__init__(functions, marginal_guarantee, normalized)


class GaussianPhiFunction(PhiFunction):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``PhiFunction`` object with polynomial features of
    X, y_pred or z.

    Parameters
    ----------
    points : Union[int, NDArray, Tuple[NDArray, NDArray]]
        If Array: List of data points, used as centers to compute
        gaussian distances.

        If integer, the points will be sampled randomly from the training
        set. The points will be sampled in the ``X`` argument if it
        is not ``None``. If ``X`` is ``None``, it will use the
        training or calibration sets used in the ``fit`` or ``calibrate``
        methods of the ``MapieCCPRegressor`` object.

        You can pass a Tuple[NDArray, NDArray], to have a different
        ``sigma`` value for each point. The two elements of the
        tuple should be:
         - Data points: 2D array of shape (n_points, n_in)
         - Sigma values 2D array of shape (n_points, n_in) or (n_points, 1)
        In this case, the ``sigma``, ``random_sigma`` and ``X`` argument are
        ignored.

        By default, ``10``

    sigma : Optional[Union[float, NDArray]]
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
        np.std(X)/(n**0.5).
        If ``X`` is ``None``, we will wait for the ``calibrate`` method of the
        ``MapieCCPRegressor`` object to be called, and sample points from
        the calibration data.

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

    X : Optional[NDArray]
        Dataset, used to sample points, if ``points`` is an
        integer, and compute the default standard deviation, if
        ``sigma``=``None``. It should not overlap with the
        calibration or testing datasets.

        If ``X`` is ``None``, it will use the
        training or calibration sets used in the ``fit`` or ``calibrate``
        methods of the ``MapieCCPRegressor`` object.

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

        By default ``True``

    Attributes
    ----------
    points: NDArray
        Array of shape (n_points, n_in), corresponding to the points used to
        compute the gaussian distanes.

    sigmas: NDArray of shape (len(points), 1) or (len(points), n_in)
        Standard deviation values

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of phi(``X``, ``y_pred``, ``z``)

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import PolynomialPhiFunction
    >>> np.random.seed(1)
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> phi = GaussianPhiFunction(2, X=X, marginal_guarantee=False,
    ...                           normalized=False)
    >>> print(np.round(phi(X), 2))
    [[0.08 0.4 ]
     [0.53 1.  ]
     [1.   0.4 ]
     [0.53 0.03]
     [0.08 0.  ]]
    >>> print(phi.points)
    [[3]
     [2]]
    >>> print(phi.sigmas)
    [[0.8892586 ]
     [0.74118567]]
    >>> phi = GaussianPhiFunction(points=([[3],[3]], [1,2]))
    >>> print(np.round(phi(X), 2))
    [[0.11 0.52 0.85]
     [0.41 0.6  0.68]
     [0.58 0.58 0.58]
     [0.41 0.6  0.68]
     [0.11 0.52 0.85]]
    >>> print(phi.points)
    [[3]
     [3]]
    >>> print(phi.sigmas)
    [[1]
     [2]]
    """
    def __init__(
            self,
            points: Union[int, NDArray, Tuple[NDArray, NDArray]] = 10,
            sigma: Optional[Union[float, NDArray]] = None,
            random_sigma: Optional[bool] = None,
            X: Optional[NDArray] = None,
            marginal_guarantee: bool = True,
            normalized: bool = True,
    ) -> None:
        self.points = points
        self.sigmas: Optional[NDArray] = None
        self.random_sigma = random_sigma
        if random_sigma is None and sigma is None:
            self.random_sigma = True

        if isinstance(points, int):
            self.sigmas = self._init_sigma(sigma, points)
            if X is None:
                self._need_x_calib = True
            else:
                points_index = np.random.choice(_num_samples(X), size=points,
                                                replace=False)
                self.points = cast(NDArray, _safe_indexing(X, points_index))

                if self.sigmas is None:
                    self.sigmas = np.ones((len(self.points), 1))*np.std(
                        X, axis=0)/(len(self.points)**0.5)

        elif isinstance(points, tuple):
            self.points = np.array(points[0])
            self.sigmas = np.array(points[1])
            if len(self.sigmas.shape) == 1:
                self.sigmas = self.sigmas.reshape(-1, 1)

            self._check_points_sigma(self.points, self.sigmas)
            if random_sigma is None and sigma is None:
                self.random_sigma = False

        elif len(np.array(points).shape) == 2:
            self.sigmas = self._init_sigma(sigma, len(points))
            self.points = np.array(points)

            if self.sigmas is None:
                if X is None:
                    self._need_x_calib = True
                else:
                    self.sigmas = np.ones((len(self.points), 1))*np.std(
                        X, axis=0)/(len(self.points)**0.5)

        else:
            raise ValueError("The points argument should be an integer, "
                             "a 2D array or a tuple of two 2D arrays.")

        if self._need_x_calib:
            functions = []
        else:
            self.points = cast(NDArray, self.points)
            self.sigmas = cast(NDArray, np.array(self.sigmas))
            self._check_points_sigma(self.points, self.sigmas)

            if self.random_sigma:
                n = len(self.points)
                self.sigmas = self.sigmas * (
                    2**np.random.normal(0, 1*2**(-2+np.log10(n)), n)
                    .reshape(-1, 1)
                )

            functions = [
                lambda X, mu=_safe_indexing(self.points, i),
                sigma=_safe_indexing(self.sigmas, i):
                np.exp(-0.5 * ((X - mu) / sigma) ** 2)
                for i in range(len(self.points))
            ]
        super().__init__(functions, marginal_guarantee, normalized)

    def _init_sigma(
            self,
            sigma: Optional[Union[float, NDArray]],
            n: int,
    ) -> Optional[NDArray]:
        """
        Return a standard deviation 2D array

        Parameters
        ----------
        sigma : Optional[Union[float, NDArray]]
            standard deviation, as float or 1D array of length n_in
            (number of dimensins of the dataset)

        n : int
            Number of points user for gaussian distances calculation

        Returns
        -------
        Optional[NDArray]
            2D array Standard deviation

        Raises
        ------
        ValueError
            If ``sigma`` is not None, a float or a 1D array
        """
        if isinstance(sigma, numbers.Number):
            sigmas = np.ones((n, 1))*sigma
        elif sigma is not None:
            if len(np.array(sigma).shape) != 1:
                raise ValueError("sigma argument should be a float "
                                 "or a 1D array of floats.")
            sigmas = np.ones((n, 1))*np.array(sigma)
        else:
            sigmas = None
        return sigmas

    def _check_points_sigma(self, points: NDArray, sigmas: NDArray) -> None:
        """
        Take 2D arrays of points and standard deviations and check
        compatibility

        Parameters
        ----------
        points : NDArray
            2D array of shape (n_points, n_in)
        sigmas : NDArray
            2D array of shape (n_points, 1) or (n_points, n_in)

        Raises
        ------
        ValueError
            If ``sigmas``is not of shape (n_points, 1) or (n_points, n_in)
        """
        if points.shape[0] != sigmas.shape[0]:
            raise ValueError("There should have as many points as "
                             "standard deviation values")
        if sigmas.shape[1] not in [1, points.shape[1]]:
            raise ValueError("The standard deviation 2D array should be of "
                             "shape (n_points, 1) or (n_points, n_in).\n"
                             f"Got sigma of shape: {sigmas.shape}")

    def _check_need_calib(self, X: NDArray) -> None:
        """
        Complete the definition of the phi function using the X training or
        calibration data, if the ``X`` argument was ``None`` during the
        ``GaussianPhiFunction``` initialisation.

        Parameters
        ----------
        X : NDArray
            Some samples (training or calibration data)
        """
        if isinstance(self.points, int):
            points_index = np.random.choice(_num_samples(X),
                                            size=self.points, replace=False)
            self.points = cast(NDArray, _safe_indexing(X, points_index))
        if self.sigmas is None:
            self.sigmas = np.ones((len(self.points), 1))*np.std(
                X, axis=0)/(len(self.points)**0.5)

        if self.random_sigma:
            n = len(self.points)
            self.sigmas = self.sigmas * (
                2**np.random.normal(0, 1*2**(-2+np.log10(n)), n)
                .reshape(-1, 1)
            )

        self._need_x_calib = False

        self.functions = [
                lambda X, mu=_safe_indexing(self.points, i),
                sigma=_safe_indexing(self.sigmas, i):
                np.exp(-0.5 * ((X - mu) / sigma) ** 2)
                for i in range(len(self.points))
            ]
