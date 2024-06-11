from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Iterable, Callable, Optional, Union, List
import warnings

import numpy as np
from mapie._typing import ArrayLike, NDArray
from .utils import (compile_functions_warnings_errors, concatenate_functions,
                    check_multiplier)
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, clone


class CCP(BaseEstimator, metaclass=ABCMeta):
    """
    Base abstract class for the phi functions,
    used in the Gibbs et al. method to model the conformity scores.

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
            Optimization initialisation value, set at ``CCP``
            initialisation.
        n_out : int
            Number of dimensions of the ``CCP`` transformation.

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
    ) -> CCP:
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
                          "width. Consider changing the CCP "
                          "definintion.\nFix: Use `bias=True` "
                          "in the `CCP` definition.")
        return result

    def __call__(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        return self.transform(X, y_pred, z)

    def __mul__(self, funct: Optional[Callable]) -> CCP:
        """
        Multiply a ``CCP`` with another function.
        This other function should return an array of shape (n_samples, 1)
        or (n_samples, )

        Parameters
        ----------
        funct : Optional[Callable]
            function which should return an array of shape (n_samples, 1)
        or (n_samples, )

        Returns
        -------
        CCP
            self, with ``funct`` as a multiplier
        """
        if funct is None:
            return self
        else:
            compile_functions_warnings_errors([funct])
            new_phi = clone(self)
            if new_phi.multipliers is None:
                new_phi.multipliers = [funct]
            else:
                new_phi.multipliers.append(funct)
            return new_phi

    def __rmul__(self, other) -> CCP:
        return self.__mul__(other)
