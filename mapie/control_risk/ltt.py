import warnings
from typing import Any, List, Tuple, Union, cast

import numpy as np

from numpy.typing import ArrayLike, NDArray

from mapie.control_risk.p_values import compute_hoeffding_bentkus_p_value


def ltt_procedure(
    r_hat: NDArray,
    alpha_np: NDArray,
    delta: float,
    n_obs: Union[int, NDArray],
    binary: bool = False,
) -> List[List[Any]]:
    """
    Apply the Learn-Then-Test procedure for risk control.
    Note that we will do a multiple test for ``r_hat`` that are
    less than level ``alpha_np``.
    The procedure follows the instructions in [1]:
        - Calculate p-values for each lambdas discretized
        - Apply a family wise error rate algorithm, here Bonferonni correction
        - Return the index lambdas that give you the control at alpha level

    Parameters
    ----------
    r_hat: NDArray of shape (n_lambdas, ) or (n_risks, n_lambdas) for multi risk.
        Empirical risk with respect to the lambdas.
        Here lambdas are thresholds that impact decision-making,
        therefore empirical risk.

    alpha_np: NDArray of shape (n_alpha, ).
        Contains the different alphas control level.
        The empirical risk should be less than alpha with
        probability 1-delta.

    delta: float.
        Probability of not controlling empirical risk.
        Correspond to proportion of failure we don't
        want to exceed.

    n_obs: Union[int, NDArray]
        Correspond to the number of observations used to compute the risk.
        In the case of a conditional loss, n_obs must be the
        number of effective observations used to compute the empirical risk
        for each lambda, hence of shape (n_lambdas, ).

    binary: bool, default=False
        Must be True if the loss associated to the risk is binary.

    Returns
    -------
    valid_index: List[List[Any]].
        Contain the valid index that satisfy FWER control
        for each alpha (length aren't the same for each alpha).

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".
    """
    if binary:
        n_obs = cast(NDArray, n_obs)
        p_values = np.array([
            compute_hoeffding_bentkus_p_value(r_hat_i, n_obs_i, alpha_np_i, binary)
            for r_hat_i, n_obs_i, alpha_np_i in zip(r_hat, n_obs, alpha_np)
        ])
        p_values = p_values.max(axis=0)  # take max over risks (no effect if mono risk)
        N = len(p_values)
        valid_index = []
        l_index = np.where(p_values <= delta/N)[0].tolist()
        valid_index.append(l_index)
    else:  # previous implementation (to correctly handle PrecisionRecallController)
        p_values = compute_hoeffding_bentkus_p_value(r_hat, n_obs, alpha_np, binary)
        N = len(p_values)
        valid_index = []
        for i in range(len(alpha_np)):
            l_index = np.where(p_values[:, i] <= delta/N)[0].tolist()
            valid_index.append(l_index)
    return valid_index


def find_lambda_control_star(
    r_hat: NDArray,
    valid_index: List[List[Any]],
    lambdas: NDArray
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Return the lambda that give the minimum precision along
    the lambdas that satisfy FWER control.
    Note: When a list in valid_index is empty, we can assume that
    lambda is equal to 1, as no lambda can verify the FWER control.
    And so to make our statement clear we also assume that the risk is
    high.

    Parameters
    ----------
    r_hat: NDArray of shape (n_lambdas, n_alpha)
        Empirical risk with respect
        to the lambdas and to the alphas.
        Here, lambdas are thresholds that impact decision-making
        and therefore the empirical risk.
        Alphas are levels of empirical risk control such that
        the empirical risk is less than alpha.

    valid_index: List[List[Any]].
        Contain the valid index that satisfy fwer control
        for each alpha (length aren't the same for each alpha).

    lambdas: NDArray of shape (n_lambda, )
        Discretize parameters use for ltt procedure.

    Returns
    -------
    l_lambda_star: ArrayLike of shape (n_alpha, ).
        The lambda that give the highest precision
        for a given alpha.

    r_star: ArrayLike of shape (n_alpha, ).
        The value of lowest risk for a given alpha.
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
            l_lambda_star.append(1)
            l_r_star.append(1)
        else:
            idx = np.argmin(valid_index[i])
            l_lambda_star.append(lambdas[valid_index[i][idx]])
            l_r_star.append(r_hat[valid_index[i][idx]])

    return l_lambda_star, l_r_star
