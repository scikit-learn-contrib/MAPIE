from typing import Union, cast

import numpy as np
from scipy.stats import binom

from numpy.typing import NDArray
from mapie.utils import _check_alpha


def compute_hoeffdding_bentkus_p_value(
    r_hat: NDArray,
    n_obs: Union[int, NDArray],
    alpha: Union[float, NDArray],
    binary: bool = False,
) -> NDArray:
    """
    The method computes the p_values according to
    the Hoeffding_Bentkus inequality for each
    alpha.
    We return the minimum between the Hoeffding and
    Bentkus p-values (Note that it depends on
    scipy.stats). The p_value is introduced in
    learn then test paper [1].

    Parameters
    ----------
    r_hat: NDArray of shape (n_lambdas, )
        Empirical risk with respect
        to the lambdas.
        Here lambdas are thresholds that impact decision
        making and therefore empirical risk.

    n_obs: Union[int, NDArray]
        Correspond to the number of observations used to compute the risk.
        In the case of a conditional loss, n_obs must be the
        number of effective observations used to compute the empirical risk
        for each lambda, hence of shape (n_lambdas, ).

    alpha: Union[float, Iterable[float]].
        Contains the different alphas control level.
        The empirical risk must be less than alpha.
        If it is a iterable, it is a NDArray of shape
        (n_alpha, ).

    binary: bool, default=False
        Must be True if the loss associated to the risk is binary.
        If True, we use a tighter version of the Bentkus p-value, valid when the
        loss associated to the risk is binary. See section 3.2 of [1].

    Returns
    -------
    hb_p_values: NDArray of shape (n_lambda, n_alpha).

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".
    """
    alpha_np = cast(NDArray, _check_alpha(alpha))
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
    if isinstance(n_obs, int):
        n_obs = np.full_like(r_hat, n_obs, dtype=float)
    n_obs_repeat = np.repeat(
        np.expand_dims(n_obs, axis=1),
        len(alpha_np),
        axis=1
    )

    hoeffding_p_value = np.exp(
        -n_obs_repeat * _h1(
            np.where(
                r_hat_repeat > alpha_repeat,
                alpha_repeat,
                r_hat_repeat
            ),
            alpha_repeat
        )
    )
    factor = 1 if binary else np.e
    bentkus_p_value = factor * binom.cdf(
        np.ceil(n_obs_repeat * r_hat_repeat), n_obs_repeat, alpha_repeat
    )
    hb_p_value = np.where(
        bentkus_p_value > hoeffding_p_value,
        hoeffding_p_value,
        bentkus_p_value
    )
    return hb_p_value


def _h1(
    r_hats: NDArray, alphas: NDArray
) -> NDArray:
    """
    This function allow us to compute the tighter version of hoeffding inequality.
    When r_hat = 0, the log is undefined, but the limit is 0, so we set the result to 0.

    Parameters
    ----------
    r_hats: NDArray of shape (n_lambdas, n_alpha).
        Empirical risk with respect
        to the lambdas.
        Here lambdas are thresholds that impact decision
        making and therefore empirical risk.
        The value table has an extended dimension of
        shape (n_lambda, n_alpha).

    alphas: NDArray of shape (n_lambdas, n_alpha).
        Contains the different alphas control level.
        In other words, empirical risk must be less
        than each alpha in alphas.
        The value table has an extended dimension of
        shape (n_lambda, n_alpha).

    Returns
    -------
    NDArray of shape (n_lambdas, n_alpha).
    """
    elt1 = np.zeros_like(r_hats, dtype=float)
    mask = r_hats != 0
    elt1[mask] = r_hats[mask] * np.log(r_hats[mask] / alphas[mask])
    elt2 = (1 - r_hats) * np.log((1 - r_hats) / (1 - alphas))
    return elt1 + elt2
