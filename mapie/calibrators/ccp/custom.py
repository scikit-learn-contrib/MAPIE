from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Union

from mapie._typing import ArrayLike
from .base import CCPCalibrator
from .utils import (check_multiplier, compile_functions_warnings_errors,
                    format_functions)
from sklearn.utils import _safe_indexing


class CustomCCP(CCPCalibrator):
    """
    Calibrator used for the ``SplitCP`` method to estimate the conformity scores.
    It corresponds to the adaptative conformal prediction method proposed by
    Gibbs et al. (2023) in "Conformal Prediction With Conditional Guarantees".

    The goal of to learn the quantile of the conformity scores distribution,
    to built the prediction interval, not with a constant ``q`` (as it is the
    case in the standard CP), but with a function ``q(X)`` which is adaptative
    as it depends on ``X``.

    This class builds a ``CCPCalibrator`` object with custom features,
    function of ``X``, ``y_pred`` or ``z``,
    defined as a list of functions in ``functions`` argument.

    This class can be used to concatenate ``CCPCalibrator`` instances.

    See the examples and the documentation to build a ``CCPCalibrator``
    adaptated to your dataset and constraints.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable[Callable]]]
        List of functions (or CCPCalibrator objects) or single function.

        Each function can take a combinaison of the following arguments:
        - ``X``: Input dataset, of shape (n_samples, ``n_in``)
        - ``y_pred``: estimator prediction, of shape (n_samples,)
        - ``z``: exogenous variable, of shape (n_samples, n_features).
            It should be given in the ``fit`` and ``predict`` methods.
        The results of each functions will be concatenated to build the final
        result of the transformation, of shape ``(n_samples, n_out)``, which
        will be used to estimate the conformity scores quantiles.

        By default ``None``.

    bias: bool
        Add a column of ones to the features, for safety reason
        (to garanty the marginal coverage, no matter how the other features
        the ``CCPCalibrator``object were built).
        If the ``CCPCalibrator``object definition covers all the dataset
        (meaning, for all calibration and test samples, the resulting
        ``calibrator.predict(X, y_pred, z)`` is never all zeros),
        this column of ones is not necessary to obtain marginal coverage.
        In this case, you can set this argument to ``False``.

        If you are not sur, use ``bias=True`` to garantee the marginal
        coverage.

        By default ``False``.

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

    Attributes
    ----------
    fit_attributes: Optional[List[str]]
        Name of attributes set during the ``fit`` method, and required to call
        ``transform``.

    n_in: int
        Number of features of ``X``

    n_out: int
        Number of features of ``calibrator.predict(X, y_pred, z)``

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
    >>> from mapie.conformity_scores import AbsoluteConformityScore
    >>> np.random.seed(1)
    >>> X_train = np.linspace(0, 3.14, 1001).reshape(-1, 1)
    >>> y_train = np.random.rand(len(X_train))*np.sin(X_train[:,0])
    >>> calibrator = CustomCCP(
    ...     functions=[
    ...         lambda X: np.sin(X[:,0]),
    ...     ],
    ...     bias=True,
    ... )
    >>> mapie = SplitMapieRegressor(
    ...     calibrator=calibrator, alpha=0.1, random_state=1,
    ...     conformity_score=AbsoluteConformityScore(sym=False)
    ... ).fit(X_train, y_train)
    >>> y_pred, y_pi = mapie.predict(X_train)
    >>> print(np.round(y_train[50::100], 2))
    [0.   0.03 0.   0.69 0.19 0.33 0.32 0.34 0.39 0.06]
    >>> print(np.round(y_pi[50::100, :, 0], 2))
    [[0.02 0.14]
     [0.02 0.42]
     [0.02 0.66]
     [0.03 0.84]
     [0.03 0.93]
     [0.02 0.93]
     [0.02 0.83]
     [0.02 0.66]
     [0.01 0.41]
     [0.01 0.12]]
    >>> print(mapie.calibrator_.n_out)
    2
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
        self.functions_ = format_functions(self.functions, self.bias)
        compile_functions_warnings_errors(self.functions_)

    def fit_params(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> CustomCCP:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected array of features.

        It should set all the attributes of ``fit_attributes``
        (i.e. ``functions_``). It should also set, once fitted, ``n_in``,
        ``n_out`` and ``init_value_``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y_pred: ArrayLike of shape (n_samples,)
            Training labels.

            By default ``None``

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``
        """
        check_multiplier(self.multipliers, X, y_pred, z)
        self._check_fit_parameters(X, y_pred, z)

        for phi in self.functions_:
            if isinstance(phi, CCPCalibrator):
                phi.fit_params(X, y_pred, z)
                check_multiplier(phi.multipliers, X, y_pred, z)
        self.is_fitted_ = True

        result = self.transform(X, y_pred, z)
        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = len(_safe_indexing(result, 0))
        self.init_value_ = self._check_init_value(self.init_value, self.n_out)
        return self
