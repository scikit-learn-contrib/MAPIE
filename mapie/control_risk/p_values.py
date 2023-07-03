import numpy as np
from numpy.typing import NDArray
from typing import Iterable, Union


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
    Note that we are only using for now Hoeffding
    part.
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
            "Invalid alpha. Allowed values are float or Iterable."
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
    return hoeffding_p_value


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


def binomial_cdf(
    k: int,
    n: int,
    p: float
) -> NDArray:
    """
    Computes the cumulative distribution function (CDF) of
    the binomial distribution for the given values of k, n, and p.

        THIS FUNCTION DOES NOT WORK

    """
    # Calculate the individual probabilities
    indices = np.arange(np.max(k) + 1)
    comb_func = np.frompyfunc(lambda x: np.prod(np.arange(n, n-x, -1)) //
                              np.prod(np.arange(1, x+1)), 1, 1)
    comb = comb_func(indices)
    probabilités = comb * (p**indices) * ((1 - p)**(n - indices))

    # Calculate the cumulative sum of probabilities
    cdf = np.cumsum(probabilités)
    return cdf


def bentkus_p_value(r_hat: NDArray,
                    n: int,
                    alpha: float
                    ) -> NDArray:
    """
    Compute the bentkus pvalue by
    taking the ceil of r_hat*n of binomial
    cdf.
        INVALID VALUE DU TO BINOM-CDF
    """
    p_values = []
    for i in range(len(r_hat)):
        p_values.append(binomial_cdf(
            np.ceil(r_hat[i] * n),
            n, alpha)[np.ceil(r_hat[i] * n).astype(int)])

    return np.array(p_values)
