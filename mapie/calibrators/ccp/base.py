from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast
import warnings

from mapie._typing import ArrayLike, NDArray
from mapie.calibrators import BaseCalibrator
from mapie.calibrators.ccp.utils import (calibrator_optim_objective,
                                         check_multiplier,
                                         compile_functions_warnings_errors,
                                         concatenate_functions)
import numpy as np
from sklearn.base import clone
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _num_samples, check_is_fitted
from scipy.optimize import minimize


class CCPCalibrator(BaseCalibrator, metaclass=ABCMeta):
    """
    Base abstract class for the calibrators used for the ``SplitCP`` method
    to estimate the conformity scores.
    It corresponds to the adaptative conformal prediction method proposed by
    Gibbs et al. (2023) in "Conformal Prediction With Conditional Guarantees".

    The goal of to learn the quantile of the conformity scores distribution,
    to built the prediction interval, not with a constant ``q`` (as it is the
    case in the standard CP), but with a function ``q(X)`` which is adaptative
    as it depends on ``X``.

    See the examples and the documentation to build a ``CCPCalibrator``
    adaptated to your dataset and constraints.

    Parameters
    ----------
    functions: Optional[Union[Callable, Iterable[Callable]]]
        List of functions (or ``CCPCalibrator`` objects) or single function.

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

    References
    ----------
    Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
    "Conformal Prediction With Conditional Guarantees", 2023
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

    def _check_init_value(
        self, init_value: Optional[ArrayLike], n_out: int
    ) -> ArrayLike:
        """
        Set the ``init_value_`` attribute depending on ``init_value`` argument.
        If ``init_value=None``, ``init_value_`` is set to
        ``np.random.normal(0, 1, n_out)``.

        Parameters
        ----------
        init_value : Optional[ArrayLike]
            Optimization initialisation value, set at ``CCPCalibrator``
            initialisation.
        n_out : int
            Number of dimensions of the ``CCPCalibrator`` transformation.

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
    ) -> CCPCalibrator:
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
        result = self.transform(X, y_pred, z)
        self.n_in = len(_safe_indexing(X, 0))
        self.n_out = len(_safe_indexing(result, 0))
        self.init_value_ = self._check_init_value(self.init_value, self.n_out)
        return self

    def fit(
        self,
        X_calib: ArrayLike,
        conformity_scores_calib: NDArray,
        y_pred_calib: Optional[ArrayLike] = None,
        z_calib: Optional[ArrayLike] = None,
        sample_weight_calib: Optional[NDArray] = None,
        reg_param: Optional[float] = None,
        **optim_kwargs,
    ) -> CCPCalibrator:
        """
        Fit function : Set all the necessary attributes to be able to transform
        ``(X, y_pred, z)`` into the expected transformation.

        It should set all the attributes of ``fit_attributes``.
        It should also set, once fitted, ``n_in``, ``n_out`` and
        ``init_value``.

        Parameters
        ----------
        X_calib: ArrayLike of shape (n_samples, n_features)
            Calibration data with not-null weights.

        conformity_scores_calib: ArrayLike of shape (n_samples,)
            Calibration conformity scores with not-null weights.

        y_pred_calib: ArrayLike of shape (n_samples,)
            Calibration target with not-null weights.

        z_calib: Optional[ArrayLike] of shape
        (n_calib_samples, n_exog_features)
            Exogenous variables with not-null weights.

            By default ``None``.

        sample_weight_calib: Optional[ArrayLike] of shape (n_samples,)
            Sample weights of the calibration data, used as weights in the
            objective function of the optimization process.
            If ``None``, then samples are equally weighted.

            By default ``None``.

        reg_param: Optional[float]
            Constant that multiplies the L2 term, controlling regularization
            strength. ``alpha`` must be a non-negative
            float i.e. in ``[0, inf)``

            Note: A too strong regularization may compromise the guaranteed
            marginal coverage. If ``calibrator.normalize=True``, it is usually
            recommanded to use ``reg_param < 0.01``.

            By default ``None``.

        optim_kwargs: Dict
            Other argument, used in sklear.optimize.minimize.
            Can be any of : ``method, jac, hess, hessp, bounds, constraints,
            tol, callback, options``
        """
        assert self.alpha is not None

        n_calib = _num_samples(X_calib)
        if self.sym:
            q_cor = np.ceil((1 - self.alpha)*(n_calib+1))/n_calib
        else:
            q_cor = np.ceil((1 - self.alpha / 2)*(n_calib+1))/n_calib
        q_cor = np.clip(q_cor, a_min=0, a_max=1)

        if self.random_state is None:
            warnings.warn("WARNING: The method implemented in "
                          "SplitCP has a stochastic behavior. "
                          "To have reproductible results, use a integer "
                          "`random_state` value in the `SplitCP` "
                          "initialisation.")
        else:
            np.random.seed(self.random_state)

        self.fit_params(X_calib, y_pred_calib, z_calib)

        cs_features = self.transform(X_calib, y_pred_calib, z_calib)

        not_nan_index = np.where(~np.isnan(conformity_scores_calib))[0]
        # Some conf. score values may be nan (ex: with ResidualNormalisedScore)

        optimal_beta_up = minimize(
            calibrator_optim_objective, self.init_value_,
            args=(
                cs_features[not_nan_index, :],
                conformity_scores_calib[not_nan_index],
                q_cor,
                sample_weight_calib,
                reg_param,
                ),
            **optim_kwargs,
            )

        if not self.sym:
            optimal_beta_low = minimize(
                calibrator_optim_objective, self.init_value_,
                args=(
                    cs_features[not_nan_index, :],
                    -conformity_scores_calib[not_nan_index],
                    q_cor,
                    sample_weight_calib,
                    reg_param,
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
        if (not self.sym
           and not optimal_beta_low.success):
            warnings.warn(
                "WARNING: The optimization process for the lower bound "
                f"failed with the following error: \n"
                f"{optimal_beta_low.message}\n"
                "The returned prediction interval may be inaccurate."
            )

        self.beta_up_ = cast(Tuple[NDArray, bool],
                             (optimal_beta_up.x, optimal_beta_up.success))
        self.beta_low_ = cast(Tuple[NDArray, bool],
                              (optimal_beta_low.x,
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
        ``(n_samples, n_out)`` which represent features to estimate the
        conformity scores.

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
            features
        """
        check_is_fitted(self, self.fit_attributes)

        params_mapping = {"X": X, "y_pred": y_pred, "z": z}
        cs_features = concatenate_functions(self.functions_, params_mapping,
                                            self.multipliers)
        if self.normalized:
            norm = np.linalg.norm(cs_features, axis=1).reshape(-1, 1)
            cs_features[(abs(norm) == 0)[:, 0], :] = np.ones(
                cs_features.shape[1])

            norm[abs(norm) == 0] = 1
            cs_features /= norm

        if np.any(np.all(cs_features == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation "
                          "calibrator.transform(X, y_pred, z) is full of "
                          "zeros. It will result in a prediction interval of "
                          "zero width. Consider changing the `CCPCalibrator` "
                          "definintion.\nFix: Use `bias=True` "
                          "in the `CCPCalibrator` definition.")
        return cs_features

    def predict(
        self,
        X: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        **kwargs,
    ) -> NDArray:
        """
        Transform ``(X, y_pred, z)`` into an array of features of shape
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
        assert y_pred is not None

        cs_features = self.transform(X, y_pred, z)

        y_pred_low = -cs_features.dot(self.beta_low_[0][:, np.newaxis])
        y_pred_up = cs_features.dot(self.beta_up_[0][:, np.newaxis])

        return np.hstack([y_pred_low, y_pred_up])

    def __call__(
        self,
        X: Optional[ArrayLike] = None,
        y_pred: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        return self.transform(X, y_pred, z)

    def __mul__(self, funct: Optional[Callable]) -> CCPCalibrator:
        """
        Multiply a ``CCPCalibrator`` with another function.
        This other function should return an array of shape (n_samples, 1)
        or (n_samples, )

        Parameters
        ----------
        funct : Optional[Callable]
            function which should return an array of shape (n_samples, 1)
        or (n_samples, )

        Returns
        -------
        CCPCalibrator
            self, with ``funct`` append in the ``multipliers`` argument list.
        """
        if funct is None:
            return self
        else:
            compile_functions_warnings_errors([funct])
            new_phi = cast(CCPCalibrator, clone(self))
            if new_phi.multipliers is None:
                new_phi.multipliers = [funct]
            else:
                new_phi.multipliers.append(funct)
            return new_phi

    def __rmul__(self, other) -> CCPCalibrator:
        """
        Do the same as ``__mul__``
        """
        return self.__mul__(other)
