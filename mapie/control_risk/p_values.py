import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Union
from scipy.stats import binom


def hoefdding_bentkus_p_value(
    r_hat: NDArray,
    n: int,
    alpha: Union[float, NDArray]
) -> NDArray:
    """
    Parameters
    ----------
    r_hat : NDArray of shape (len(lambdas), )
        Empirical risk of metric_control with respect
        to the lambdas.
    n : Integer value
        Correspond to the number of observations in
        X_cal dataset.
    alpha: NDArray.
        Correspond to the value that r_hat should not
        exceed.

    Method
    -------
    The method computes the p_values according to
    the Hoeffding_Bentkus inequality for each
    alpha.
    We return the min between the Hoeffding and
    Bentkus p-values (Note that it depends on
    scipy.stats).
    Returns
    -------
    p_values: NDArray of shape
        (len(lambdas), len(alpha)).
    """
    # bentkus_p_value = np.e * binom.cdf(np.ceil(n*r_hat), n, alpha)
    if isinstance(alpha, float):
        alpha_np = np.array([alpha])
    elif isinstance(alpha, Iterable):
        alpha_np = np.array(alpha)
    else:
        raise ValueError(
            "Invalid alpha. Allowed values are float or NDArray."
        )
    if len(alpha_np.shape) != 1:
        raise ValueError(
            "Invalid alpha."
            "Please provide a one-dimensional list of values."
        )

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
    hoeffding_p_value = np.exp(-n * h1(np.where(
        r_hat_repeat > alpha_repeat, alpha_repeat, r_hat_repeat),
        alpha_repeat))
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat_repeat),
                                       n, alpha_repeat)
    hb_p_value = np.where(bentkus_p_value > hoeffding_p_value,
                          hoeffding_p_value,
                          bentkus_p_value)
    # return hoeffding_p_value
    return hb_p_value


def h1(
    r_hat: NDArray,
    alpha: NDArray
) -> NDArray:
    """
    Parameters
    ----------
    r_hat : NDArray of shape (n_samples, )
        Empirical risk of metric_control with respect
        to the lambdas.
    alpha : NDArray of alphas level.

    Returns
    -------
    NDArray of same shape as r_hat.
    """

    return r_hat * np.log(r_hat/alpha) + (1-r_hat) * np.log(
                                        (1-r_hat)/(1-alpha))
