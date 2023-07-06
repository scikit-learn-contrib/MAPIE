import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Union, cast, Optional
from scipy.stats import binom
from mapie.utils import check_alpha


def compute_hoefdding_bentkus_p_value(
    r_hat: NDArray,
    n: int,
    alpha: Union[float, NDArray]
) -> NDArray:
    """
    The method computes the p_values according to
    the Hoeffding_Bentkus inequality for each
    alpha.
    We return the minimum between the Hoeffding and
    Bentkus p-values (Note that it depends on
    scipy.stats). The p_value is introduced in
    learn then test paper.

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".

    Parameters
    ----------
    r_hat: NDArray of shape (n_lambdas, )
        Empirical risk of metric_control with respect
        to the lambdas.
        Note: r_hat is the empirical mean of a matrix of
            shape (n_samples, n_lambdas).

    n: Integer value
        Correspond to the number of observations in
        dataset.

    alpha: NDArray of shape (n_alpha) chosen by user.
        Correspond to the value that r_hat should not
        exceed.

    Returns
    -------
    hb_p_values: NDArray of shape
        (n_lambdas, n_alpha).
    """
    alpha_np = cast(NDArray, check_alpha(alpha))
    alpha_np = alpha_np[:, np.newaxis]
    r_hat_repeat = np.repeat(
        np.expand_dims(r_hat, axis=1),
        len(alpha_np),
        axis=1
    )
    alpha_repeat = np.repeat(
        alpha_np.reshape(1, -1),
        len(r_hat),
        axis=0
    )
    hoeffding_p_value = np.exp(-n * _h1(np.where(
        r_hat_repeat > alpha_repeat, alpha_repeat, r_hat_repeat),
        alpha_repeat))
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat_repeat),
                                       n, alpha_repeat)
    hb_p_value = np.where(bentkus_p_value > hoeffding_p_value,
                          hoeffding_p_value,
                          bentkus_p_value)
    return hb_p_value


def _h1(
    r_hat: NDArray,
    alpha: NDArray
) -> NDArray:
    """
    This function allow us to compute
    the tighter version of hoeffding inequality.
    This function is then used in the
    hoeffding_bentkus_p_value function.

    Parameters
    ----------
    r_hat: NDArray of shape (n_lambdas, ).
        Empirical risk of metric_control with respect
        to the lambdas.

    alpha: NDArray of alphas level of
        shape (n_alpha), determined by user.

    Returns
    -------
    NDArray of same shape as r_hat
        (n_lambdas, ).
    """

    return r_hat * np.log(r_hat/alpha) + (1-r_hat) * np.log(
                                        (1-r_hat)/(1-alpha))
