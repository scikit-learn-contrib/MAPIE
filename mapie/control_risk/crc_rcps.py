from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def get_r_hat_plus(
    risks: NDArray,
    lambdas: NDArray,
    method: Optional[str],
    bound: Optional[str],
    delta: Optional[float],
    sigma_init: Optional[float]
) -> Tuple[NDArray, NDArray]:
    """
    Compute the upper bound of the loss for each lambda.
    The procedure here are the RCPS[1]
    and CRC [2] methods.

    Parameters
    ----------
    risks: ArrayLike of shape (n_samples_cal, n_lambdas)
        The risk for each observation for each threshold.

    lambdas: NDArray of shape (n_lambdas, )
        Array with all the values of lambda.
        Threshold that permit to compute score.

    method: Optional[str]
        Correspond to the method use to control recall
        score. Could be either CRC or RCPS.
        When method is RCPS and bound is None the default
        bound use wsr.

    bound: Optional[str]
        Bounds to compute. Either hoeffding, bernstein or wsr.
        When method is RCPS and bound in None, use wsr bound.

    delta: Optional[float]
        Level of confidence.

    sigma_init: Optional[float]
        First variance in the sigma_hat array. The default
        value is the same as in the paper implementation [1].

    Returns
    -------
    Tuple[NDArray, NDArray] of shape (n_lambdas, ) and (n_lambdas)
        Average risk over all the obervations and upper bound of the risk.

    References
    ----------
    [1] Bates, S., Angelopoulos, A., Lei, L., Malik, J., & Jordan, M.
    (2021).
    Distribution-free, risk-controlling prediction sets.

    [2] Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T.
    (2022).
    Conformal risk control.
    """
    n_lambdas = len(lambdas)
    r_hat = risks.mean(axis=0)
    n_obs = len(risks)

    if (method == "rcps") and (delta is not None):
        if bound == "hoeffding":
            r_hat_plus = (
                r_hat +
                np.sqrt((1 / (2 * n_obs)) * np.log(1 / delta))
            )

        elif bound == "bernstein":
            sigma_hat_bern = np.var(r_hat, axis=0, ddof=1)
            r_hat_plus = (
                r_hat +
                np.sqrt((sigma_hat_bern * 2 * np.log(2 / delta)) / n_obs) +
                (7 * np.log(2 / delta)) / (3 * (n_obs - 1))
            )

        else:
            mu_hat = (
                (.5 + np.cumsum(risks, axis=0)) /
                (np.repeat(
                    [range(1, n_obs + 1)],
                    n_lambdas,
                    axis=0
                ).T + 1)
            )
            sigma_hat = (
                (.25 + np.cumsum((risks - mu_hat)**2, axis=0)) /
                (np.repeat(
                    [range(1, n_obs + 1)],
                    n_lambdas,
                    axis=0
                ).T + 1)
            )
            sigma_hat = np.concatenate(
                [
                    np.full(
                        (1, n_lambdas), fill_value=sigma_init
                    ), sigma_hat[:-1]
                ]
            )
            nu = np.minimum(
                1,
                np.sqrt((2 * np.log(1 / delta)) / (n_obs * sigma_hat))
            )

            # Split the calculation in two to prevent memory issues
            batches = [
                range(int(n_obs / 2)),
                range(n_obs - int(n_obs / 2), n_obs)
            ]
            K_R_max = np.zeros((n_lambdas, n_lambdas))
            for batch in batches:
                nu_batch = nu[batch]
                losses_batch = risks[batch]

                nu_batch = np.repeat(
                    np.expand_dims(nu_batch, axis=2),
                    n_lambdas,
                    axis=2
                )
                losses_batch = np.repeat(
                    np.expand_dims(losses_batch, axis=2),
                    n_lambdas,
                    axis=2
                )

                R = lambdas
                K_R = np.cumsum(
                    np.log(
                        (
                            1 -
                            nu_batch *
                            (losses_batch - R)
                        ) +
                        np.finfo(np.float64).eps
                    ),
                    axis=0
                )
                K_R = np.max(K_R, axis=0)
                K_R_max += K_R

            r_hat_plus_tronc = lambdas[np.argwhere(
                np.cumsum(K_R_max > -np.log(delta), axis=1) == 1
            )[:, 1]]
            r_hat_plus = np.ones(n_lambdas)
            r_hat_plus[:len(r_hat_plus_tronc)] = r_hat_plus_tronc

    else:
        r_hat_plus = (n_obs / (n_obs + 1)) * r_hat + (1 / (n_obs + 1))

    return r_hat, r_hat_plus


def find_lambda_star(
    lambdas: NDArray,
    r_hat_plus: NDArray,
    alpha_np: NDArray
) -> NDArray:
    """Find the higher value of lambda such that for
    all smaller lambda, the risk is smaller, for each value
    of alpha.

    Parameters
    ----------
    lambdas: NDArray
        Array with all the values of lambda.
        Threshold that permit to compute score.

    r_hat_plus: NDArray of shape (n_lambdas, )
        Upper bounds computed in the `get_r_hat_plus` method.

    alphas: NDArray of shape (n_alphas, )
        Risk levels.

    Returns
    -------
    NDArray of shape (n_alphas, )
        Optimal lambdas which control the risks for each value
        of alpha.

    References
    ----------
    [1] Bates, S., Angelopoulos, A., Lei, L., Malik, J., & Jordan, M.
    (2021).
    Distribution-free, risk-controlling prediction sets.

    [2] Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T.
    (2022).
    Conformal risk control.
    """
    if len(alpha_np) > 1:
        alphas_np = alpha_np[:, np.newaxis]
    else:
        alphas_np = alpha_np

    bound_rep = np.repeat(
        np.expand_dims(r_hat_plus, axis=0),
        len(alphas_np),
        axis=0
    )
    bound_rep[:, np.argmax(bound_rep, axis=1)] = np.maximum(
        alphas_np,
        bound_rep[:, np.argmax(bound_rep, axis=1)]
    )  # to avoid an error if the risk is always higher than alpha
    lambdas_star = lambdas[np.argmin(
        - np.greater_equal(
            bound_rep,
            alphas_np
        ).astype(int),
        axis=1
    )]
    return lambdas_star
