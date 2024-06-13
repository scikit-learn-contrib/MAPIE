from __future__ import annotations

from typing import Iterable, Callable, Optional, Union, List

from mapie._typing import ArrayLike
from .base import CCPCalibrator
from .utils import (compile_functions_warnings_errors, format_functions,
                    check_multiplier)
from sklearn.utils import _safe_indexing


class CustomCCP(CCPCalibrator):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``CCP`` object with custom features of
    X, y_pred or z, defined as a list of functions in ``functions`` argument.

    This class can be used to concatenate ``CCP`` instances.

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

        By default ``None``.

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
    # >>> import numpy as np
    # >>> from mapie.calibrators import CustomCCP
    # >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    # >>> y_pred = np.array([0, 0, 1])
    # >>> phi = CustomCCP(
    # ...     functions=[
    # ...         lambda X: X, # X, if y_pred is 0
    # ...         lambda y_pred: y_pred,                     # y_pred
    # ...     ],
    # ...     normalized=False,
    # ... ).fit(X, y_pred)
    # >>> print(phi.predict(X, y_pred))
    # [[1. 2. 0.]
    #  [3. 4. 0.]
    #  [0. 0. 1.]]
    # >>> print(phi.n_out)
    # 3

    >>> import numpy as np
    >>> from mapie.calibrators import GaussianCCP
    >>> from mapie.regression import SplitMapieRegressor
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

    def fit_params(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> CustomCCP:
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
            if isinstance(phi, CCPCalibrator):
                phi.fit_params(X, y_pred, z)
                check_multiplier(phi.multipliers, X, y_pred, z)
        self.is_fitted_ = True

        result = self.transform(X, y_pred, z)
        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = len(_safe_indexing(result, 0))
        self.init_value_ = self._check_init_value(self.init_value, self.n_out)
        check_multiplier(self.multipliers, X, y_pred, z)
        return self
