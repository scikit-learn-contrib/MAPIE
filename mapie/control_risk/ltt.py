import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Optional, Tuple
from .p_values import hoefdding_bentkus_p_value


def _ltt_procedure(
        r_hat: NDArray,
        alpha_np: NDArray,
        delta: Optional[float],
        n_obs: int
) -> Tuple[NDArray, NDArray]:
    """
    Apply the ltt procedure for risk control
    should be precision for multilabelclassif
    Note that we will do a multipletest for r_hat that are
    less than level alpha.
    Procedure: 
        - compute p_values for each lambdas descretize
        - Apply a fwer algorithm, here Bonferonni correction
        - Return the index lambdas that give you the control 
        at alpha level
        - n_obs is the length of data cal
   Parameters
    ----------
    r_hat : NDArray of shape (n_samples, )
        Empirical risk of metric_control with respect
        to the lambdas.
    alpha : NDArray of alphas level.
    delta: Float value
        Correspond to proportion of failure we don't 
        want to exceed.
    Returns
    ----------
    valid_index: NDArray of shape (n_alpha, ).
        Contain the valid index that satisfy fwer control
        for each alpha (shape aren't the same for each alpha)
    p_values: NDArray of shape (n_lambda, n_alpha)
        Contains the values of p_value for different alpha
    """
    p_values = hoefdding_bentkus_p_value(r_hat, n_obs, alpha_np)
    
    if p_values.shape[1] > 1:
        valid_index = []
        for i in range(len(alpha_np)):
            N_coarse = len(np.where(p_values[:, i] < delta/n_obs)[0])
            if N_coarse == 0:
                l_index = np.array([])
                valid_index.append(l_index)
            else:
                l_index = np.where(p_values[:, i] <= delta/n_obs)[0]
                valid_index.append(l_index)
        return np.array(valid_index), p_values
    else:
        N_coarse = len(np.where(p_values <= delta/n_obs)[0])
        if N_coarse == 0:
            valid_index = np.array([])
        else:
            valid_index = np.where(p_values <= delta/N_coarse)[0]
        return valid_index, p_values


def _find_lambda_control_star(
        r_hat: NDArray,
        valid_index: NDArray,
        lambdas: NDArray
) -> Tuple[NDArray, ArrayLike]:
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
    ----------
    l_lambda_star: NDArray of shape (n_alpha, )
        the lambda that give the highest precision
    r_star : NDArray of shape (n_alpha, )
        the value of lowest risk.
    """
    if len(valid_index) == 0:
        raise ValueError(
        """
        ERROR: The list of valid index is empty, use higher alpha or delta.
        """
        )
    if isinstance(valid_index[0], (np.int64)):

        lambda_star = lambdas[valid_index[np.argmin(r_hat[valid_index])]]
        r_star = r_hat[valid_index[np.argmin(r_hat[valid_index])]]

        return lambda_star, r_star

    else:

        l_lambda_star = []
        l_r_star = []
        for i in range(len(valid_index)):
            l_lambda_star.append(lambdas[valid_index[i][np.argmin(
                r_hat[valid_index[i]])]])
            l_r_star.append(
                r_hat[valid_index[i][np.argmin(r_hat[valid_index[i]])]])

        return l_lambda_star, l_r_star


def fixed_sequence_testing(p_values: NDArray,
                           delta: float,
                           downsample_factor: int
) -> NDArray:
    """
    Other technique for FWER control
    In LTT procedure we use by default Bonferonni
    correction.
    This one is another presented in LTT paper.
    """
    
    N = p_values.shape[0]
    N_coarse = max(int(p_values.shape[0] / downsample_factor), 1)

    coarse_indexes = np.arange(0, N, downsample_factor)
    coarse_indexes = np.append(coarse_indexes, [N-1])

    mask = p_values < delta / N_coarse
    mask_index = np.where(mask)[0]

    return mask_index
