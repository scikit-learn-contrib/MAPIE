import numpy as np
import warnings
from numpy.typing import NDArray
from typing import Tuple, List, Union, Optional, Any
from .p_values import compute_hoefdding_bentkus_p_value


def _ltt_procedure(
    r_hat: NDArray,
    alpha_np: NDArray,
    delta: Optional[float],
    n_obs: int
) -> Tuple[List[List[Any]], NDArray]:
    """
    Apply the learn then test procedure for risk control
    should be precision for multi-label-classification.
    Note that we will do a multipletest for r_hat that are
    less than level alpha.

    Procedure:
        - compute p_values for each lambdas descretize
        - Apply a fwer algorithm, here Bonferonni correction
        - Return the index lambdas that give you the control
        at alpha level

    Parameters
    ----------
    r_hat: NDArray of shape (n_samples, )
        Empirical risk of metric_control with respect
        to the lambdas.

    alpha: NDArray of control level. The empirical risk should
        be less than alpha with probability 1-delta.

    delta: Float value.
        Correspond to proportion of failure we don't
        want to exceed.

    Returns
    -------
    valid_index: NDArray of shape (n_alpha, ).
        Contain the valid index that satisfy fwer control
        for each alpha (shape aren't the same for each alpha)

    p_values: NDArray of shape (n_lambda, n_alpha)
        Contains the values of p_value for different alpha

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".
    """
    if delta is None:
        raise ValueError(
            "Invalid delta: delta cannot be None while"
            + " using LTT for precision control. "
        )
    p_values = compute_hoefdding_bentkus_p_value(r_hat, n_obs, alpha_np)
    valid_index = []
    for i in range(len(alpha_np)):
        l_index = np.where(p_values[:, i] <= delta/n_obs)[0].tolist()
        valid_index.append(l_index)
    return valid_index, p_values


def _find_lambda_control_star(
    r_hat: NDArray,
    valid_index: List[List[Any]],
    lambdas: NDArray
) -> Tuple[Union[NDArray, List], Union[NDArray, List]]:
    """
    Return the lambda that give the maximum precision with a control
    guarantee of level delta.

    Parameters
    ----------
    r_hat : NDArray of shape (n_samples, )
        Empirical risk of metric_control with respect
        to the lambdas.

    valid_index: NDArray of shape (n_alpha, ).
        Contain the valid index that satisfy fwer control
        for each alpha (shape aren't the same for each alpha)

    lambdas: Discretize parameters use for ltt procedure.

    Returns
    -------
    l_lambda_star: NDArray of shape (n_alpha, )
        the lambda that give the highest precision

    r_star: NDArray of shape (n_alpha, )
        the value of lowest risk.
    """
    if [] in valid_index:
        warnings.warn(
            """
            Warning: At least one sequence is empty!
            """
        )
    l_lambda_star = []  # type: List[Any]
    l_r_star = []  # type: List[Any]
    for i in range(len(valid_index)):
        if len(valid_index[i]) == 0:
            l_lambda_star.append([])
            l_r_star.append([])
        else:
            idx = np.argmin(r_hat[valid_index[i]])
            l_lambda_star.append(lambdas[valid_index[i][idx]])
            l_r_star.append(
                r_hat[valid_index[i][idx]])

    return l_lambda_star, l_r_star
