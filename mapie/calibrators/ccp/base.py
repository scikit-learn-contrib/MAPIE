from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Iterable, Callable, Optional, Union, List, Tuple, cast
import warnings

import numpy as np
from scipy.optimize import minimize
from mapie._typing import ArrayLike, NDArray
from .utils import (compile_functions_warnings_errors, concatenate_functions,
                    check_multiplier)
from mapie.calibrators import Calibrator
from mapie.calibrators.ccp.utils import calibrator_optim_objective
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone


class CCP(Calibrator, metaclass=ABCMeta):
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

    beta_up_: Tuple[NDArray, bool]
        Calibration fitting results, used to build the upper bound of the
        prediction intervals.
        beta_up[0]: Array of shape (calibrator.n_out, )
        beta_up[1]: Whether the optimization process converged or not
                    (the coverage is not garantied if the optimization fail)

    beta_low_: Tuple[NDArray, bool]
        Same as beta_up, but for the lower bound

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

    def fit_params(
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

        y_pred: ArrayLike of shape (n_samples,)
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

    def fit(
        self,
        X_calib: ArrayLike,
        y_pred_calib: Optional[ArrayLike],
        z_calib: Optional[ArrayLike],
        calib_conformity_scores: NDArray,
        alpha: float,
        sym: bool,
        sample_weight_calib: Optional[ArrayLike] = None,
        random_state: Optional[int] = None,
        **optim_kwargs,
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
            Calibration data.

        y_pred: ArrayLike of shape (n_samples,)
            Calibration target.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

        conformity_scores: ArrayLike of shape (n_samples,)
            Calibration conformity scores

        alpha: float
            Between ``0.0`` and ``1.0``, represents the risk level of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

        sym: bool
            Weather or not, the prediction interval should be symetrical
            or not.

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

        random_state: Optional[int]
            Integer used to set the numpy seed, to get reproducible calibration
            results.
            If ``None``, the prediction intervals will be stochastics, and will
            change if you refit the calibration
            (even if no arguments have change).

            WARNING: If ``random_state``is not ``None``, ``np.random.seed``
            will be changed, which will reset the seed for all the other random
            number generators. It may have an impact on the rest of your code.

            By default ``None``.

        optim_kwargs: Dict
            Other argument, used in sklear.optimize.minimize
        """
        if sym:
            q_low = 1 - alpha
            q_up = 1 - alpha
        else:
            q_low = alpha / 2
            q_up = 1 - alpha / 2

        if random_state is None:
            warnings.warn("WARNING: The method implemented in "
                          "SplitMapie has a stochastic behavior. "
                          "To have reproductible results, use a integer "
                          "`random_state` value in the `SplitMapie` "
                          "initialisation.")
        else:
            np.random.seed(random_state)

        self.fit_params(X_calib, y_pred_calib, z_calib)

        phi_x = self.transform(X_calib, y_pred_calib, z_calib)

        not_nan_index = np.where(~np.isnan(calib_conformity_scores))[0]
        # Some conf. score values may be nan (ex: with ResidualNormalisedScore)

        optimal_beta_up = minimize(
            calibrator_optim_objective, self.init_value_,
            args=(
                phi_x[not_nan_index, :],
                calib_conformity_scores[not_nan_index],
                q_up,
                sample_weight_calib,
                ),
            **optim_kwargs,
            )

        if not sym:
            optimal_beta_low = minimize(
                calibrator_optim_objective, self.init_value_,
                args=(
                    phi_x[not_nan_index, :],
                    calib_conformity_scores[not_nan_index],
                    q_low,
                    sample_weight_calib,
                ),
                **optim_kwargs,
            )
        else:
            optimal_beta_low = optimal_beta_up

        if not optimal_beta_up.success:
            warnings.warn(
                "WARNING: The optimization process for the upper bound "
                f"failed with the following error: \n"
                f"{optimal_beta_low.message}\n"
                "The returned prediction interval may be inaccurate."
            )
        if (not sym
           and not optimal_beta_low.success):
            warnings.warn(
                "WARNING: The optimization process for the lower bound "
                f"failed with the following error: \n"
                f"{optimal_beta_low.message}\n"
                "The returned prediction interval may be inaccurate."
            )

        signed = -1 if sym else 1

        self.beta_up_ = cast(Tuple[NDArray, bool],
                             (optimal_beta_up.x, optimal_beta_up.success))
        self.beta_low_ = cast(Tuple[NDArray, bool],
                              (signed * optimal_beta_low.x,
                               optimal_beta_low.success))

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
        phi_x = concatenate_functions(self.functions_, params_mapping,
                                      self.multipliers)
        if self.normalized:
            norm = np.linalg.norm(phi_x, axis=1).reshape(-1, 1)
            phi_x[(abs(norm) == 0)[:, 0], :] = np.ones(phi_x.shape[1])

            norm[abs(norm) == 0] = 1
            phi_x /= norm

        if np.any(np.all(phi_x == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation "
                          "phi(X, y_pred, z) is full of zeros. "
                          "It will result in a prediction interval of zero "
                          "width. Consider changing the CCP "
                          "definintion.\nFix: Use `bias=True` "
                          "in the `CCP` definition.")

        return phi_x

    def predict(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Transform ``(X, y_pred, z)`` into an array of shape
        ``(n_samples, n_out)`` and compute the dot product with the
        optimized beta values, to get the conformity scores estimations.

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
        phi_x = self.transform(X, y_pred, z)

        y_pred_low = phi_x.dot(self.beta_low_[0][:, np.newaxis])
        y_pred_up = phi_x.dot(self.beta_up_[0][:, np.newaxis])

        return np.hstack([y_pred_low, y_pred_up])

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
