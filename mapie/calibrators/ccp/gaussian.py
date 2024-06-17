from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from mapie._typing import ArrayLike
from mapie.calibrators import BaseCalibrator
from mapie.calibrators.ccp import CCPCalibrator
from .utils import compute_sigma, format_functions, sample_points
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples


class GaussianCCP(CCPCalibrator):
    """
    Calibrator used for the ``SplitCP`` method to estimate the
    conformity scores. It corresponds to the adaptative conformal
    prediction method proposed by Gibbs et al. (2023)
    in "Conformal Prediction With Conditional Guarantees".

    The goal of to learn the quantile of the conformity scores distribution,
    to built the prediction interval, not with a constant ``q`` (as it is the
    case in the standard CP), but with a function ``q(X)`` which is adaptative
    as it depends on ``X``.

    This class builds a ``CCPCalibrator`` object with gaussian kernel features,
    by sampling some points (or set by the user), and computing the gaussian
    distance between ``X`` and the point.

    See the examples and the documentation to build a ``CCPCalibrator``
    adaptated to your dataset and constraints.

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
        the ``CCPCalibrator``object were built).
        If the ``CCPCalibrator``object definition covers all the dataset
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
        Whether or not to normalized the resulting
        ``calibrator.predict(X, y_pred, z)``. Normalization
        will result in a bounded interval prediction width, avoiding the width
        to explode to +inf or crash to zero. It is particularly intersting when
        you know that the conformity scores are bounded. It also prevent the
        interval to have a width of zero for out-of-distribution samples.
        On the opposite, it is not recommended if the conformity
        scores can vary a lot.

        By default ``False``

    init_value: Optional[ArrayLike]
        Optimization initialisation value.
        If ``None``, is sampled from a normal distribution.

        By default ``None``.

    multipliers: Optional[List[Callable]]
        List of function which take any arguments of ``X, y_pred, z``
        and return an array of shape ``(n_samples, 1)``.
        The result of ``calibrator.transform(X, y_pred, z)`` will be multiply
        by the result of each function of ``multipliers``.

        Note: When you multiply a ``CCPCalibrator`` with a function, it create
        a new instance of ``CCPCalibrator`` (with the same arguments), but
        add the function to the ``multipliers`` list.

    reg_param: Optional[float]
        Constant that multiplies the L2 term, controlling regularization
        strength. ``alpha`` must be a non-negative
        float i.e. in ``[0, inf)``.

        Note: A too strong regularization may compromise the guaranteed
        marginal coverage. If ``calibrator.normalize=True``, it is usually
        recommanded to use ``reg_param < 0.01``.

        If ``None``, no regularization is used.

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

    beta_up_: Tuple[NDArray, bool]
        Calibration fitting results, used to build the upper bound of the
        prediction intervals.
        beta_up_[0]: Array of shape (calibrator.n_out, )
        beta_up_[1]: Whether the optimization process converged or not
                    (the coverage is not garantied if the optimization fail)

    beta_low_: Tuple[NDArray, bool]
        Same as beta_up, but for the lower bound

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.calibrators import GaussianCCP
    >>> from mapie.regression import SplitCPRegressor
    >>> np.random.seed(1)
    >>> X_train = np.arange(0,400, 2).reshape(-1, 1)
    >>> y_train = 1 + 2*X_train[:,0] + np.random.rand(len(X_train))
    >>> mapie = SplitCPRegressor(
    ...     calibrator=GaussianCCP(2), alpha=0.1, random_state=1,
    ... ).fit(X_train, y_train)
    >>> y_pred, y_pi = mapie.predict(X_train)
    >>> print(np.round(y_pred[:5], 2))
    [ 1.46  5.46  9.46 13.46 17.46]
    >>> print(np.round(y_pi[:5, :, 0], 2))
    [[ 0.95  1.96]
     [ 4.95  5.96]
     [ 8.95  9.97]
     [12.95 13.97]
     [16.95 17.97]]
    >>> print(mapie.calibrator_.points_)
    [[204]
     [318]]
    >>> print(mapie.calibrator_.sigmas_)
    [[86.34106786]
     [86.34106786]]
    >>> print(mapie.calibrator_.n_out)
    2
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
        reg_param: Optional[float] = None,
    ) -> None:
        self.points = points
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.bias = bias
        self.normalized = normalized
        self.init_value = init_value
        self.multipliers = multipliers
        self.reg_param = reg_param

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
        Check fit parameters. In particular, check that the ``functions``
        attribute is valid and set the ``functions_``.

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


def check_calibrator(
    calibrator: Optional[BaseCalibrator],
) -> BaseCalibrator:
    """
    Check if ``calibrator`` is a ``BaseCalibrator`` instance.

    Parameters
    ----------
    calibrator: Optional[BaseCalibrator]
        A ``BaseCalibrator`` instance used to estimate the conformity scores
        quantiles.

        If ``None``, use as default a ``GaussianCCP`` instance.

    Returns
    -------
    BaseCalibrator
        ``calibrator`` if defined, a ``GaussianCCP`` instance otherwise.

    Raises
    ------
    ValueError
        If ``calibrator`` is not ``None`` nor a ``BaseCalibrator`` instance.
    """
    if calibrator is None:
        return GaussianCCP()
    elif isinstance(calibrator, BaseCalibrator):
        return calibrator
    else:
        raise ValueError("Invalid `calibrator` argument. It must be `None` "
                         "or a `BaseCalibrator` instance.")
