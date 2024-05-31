from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional, Union
import warnings

import numpy as np
from mapie._typing import NDArray
from sklearn.utils import _safe_indexing


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
