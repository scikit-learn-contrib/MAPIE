from __future__ import annotations

from typing import Callable, Optional, Tuple, Union, List

from mapie._typing import ArrayLike
from .base import CCP
from .utils import format_functions


class PolynomialCCP(CCP):
    """
    This class is used to define the transformation phi,
    used in the Gibbs et al. method to model the conformity scores.
    This class build a ``CCP`` object with polynomial features of
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

    exponents: List[int]
        List of exponents of the built polynomial features

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.phi_function import PolynomialCCP
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y_pred = np.array([1, 2, 3])
    >>> phi = PolynomialCCP(3).fit(X, y_pred)
    >>> print(phi.transform(X, y_pred))
    [[  1.   2.   1.   4.   1.   8.   1.]
     [  3.   4.   9.  16.  27.  64.   1.]
     [  5.   6.  25.  36. 125. 216.   1.]]
    >>> print(phi.exponents)
    [0, 1, 2, 3]
    >>> phi = PolynomialCCP([1, 2, 5], "y_pred",
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
            the ``CCP``object were built).
            If the ``CCP``object definition covers all the dataset
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
