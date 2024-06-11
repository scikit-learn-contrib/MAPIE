from __future__ import annotations

from typing import Callable, Optional, Tuple, Union, List

import numpy as np
from mapie._typing import ArrayLike
from .base import CCP
from .utils import format_functions, compute_sigma, sample_points
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples


class GaussianCCP(CCP):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``CCP`` object with features been the gaussian
    distances between X and some defined points.

    Parameters
    ----------
    points : Union[int, ArrayLike, Tuple[ArrayLike, ArrayLike]]
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
        ``GaussianCCP.fit`` method, which correspond to the ``X``
        argument of the ``MapieCCPRegressor.calibrate`` method
        (unless you call ``GaussianCCP.fit(X)`` yourself).

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
        the ``CCP``object were built).
        If the ``CCP``object definition covers all the dataset
        (meaning, for all calibration and test samples, ``phi(X, y_pred, z)``
        is never all zeros), this column of ones is not necessary
        to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        Note: In this case, with ``GaussianCCP``, if ``normalized`` is
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
    >>> from mapie.phi_function import GaussianCCP
    >>> np.random.seed(1)
    >>> X = np.array([[1], [2], [3], [4], [5]])
    >>> phi = GaussianCCP(2, bias=False,
    ...                           normalized=False).fit(X)
    >>> print(np.round(phi.predict(X), 2))
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
    >>> phi = GaussianCCP([[3],[4]], 0.5).fit(X)
    >>> print(np.round(phi.predict(X), 2))
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


def check_phi(
    phi: Optional[CCP],
) -> CCP:
    """
    Check if ``phi`` is a ``CCP`` instance.

    Parameters
    ----------
    phi: Optional[CCP]
        A ``CCP`` instance used to estimate the conformity scores.

        If ``None``, use as default a ``GaussianCCP`` instance.
        See the examples and the documentation to build a ``CCP``
        adaptated to your dataset and constraints.

    Returns
    -------
    CCP
        ``phi`` if defined, a ``GaussianCCP`` instance otherwise.

    Raises
    ------
    ValueError
        If ``phi`` is not ``None`` nor a ``CCP`` instance.
    """
    if phi is None:
        return GaussianCCP()
    elif isinstance(phi, CCP):
        return phi
    else:
        raise ValueError("Invalid `phi` argument. It must be `None` or a "
                         "`CCP` instance.")
