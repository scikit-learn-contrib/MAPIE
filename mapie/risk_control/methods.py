import warnings
from typing import Any, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

from mapie.risk_control.fwer_control import control_fwer
from mapie.utils import _check_alpha


def get_r_hat_plus(
    risks: NDArray,
    lambdas: NDArray,
    method: Optional[str],
    bound: Optional[str],
    delta: Optional[float],
    sigma_init: Optional[float],
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
            r_hat_plus = r_hat + np.sqrt((1 / (2 * n_obs)) * np.log(1 / delta))

        elif bound == "bernstein":
            sigma_hat_bern = np.var(r_hat, axis=0, ddof=1)
            r_hat_plus = (
                r_hat
                + np.sqrt((sigma_hat_bern * 2 * np.log(2 / delta)) / n_obs)
                + (7 * np.log(2 / delta)) / (3 * (n_obs - 1))
            )

        else:
            mu_hat = (0.5 + np.cumsum(risks, axis=0)) / (
                np.repeat([range(1, n_obs + 1)], n_lambdas, axis=0).T + 1
            )
            sigma_hat = (0.25 + np.cumsum((risks - mu_hat) ** 2, axis=0)) / (
                np.repeat([range(1, n_obs + 1)], n_lambdas, axis=0).T + 1
            )
            sigma_hat = np.concatenate(
                [np.full((1, n_lambdas), fill_value=sigma_init), sigma_hat[:-1]]
            )
            nu = np.minimum(1, np.sqrt((2 * np.log(1 / delta)) / (n_obs * sigma_hat)))

            # Split the calculation in two to prevent memory issues
            batches = [range(int(n_obs / 2)), range(n_obs - int(n_obs / 2), n_obs)]
            K_R_max = np.zeros((n_lambdas, n_lambdas))
            for batch in batches:
                nu_batch = nu[batch]
                losses_batch = risks[batch]

                nu_batch = np.repeat(
                    np.expand_dims(nu_batch, axis=2), n_lambdas, axis=2
                )
                losses_batch = np.repeat(
                    np.expand_dims(losses_batch, axis=2), n_lambdas, axis=2
                )

                R = lambdas
                K_R = np.cumsum(
                    np.log(
                        (1 - nu_batch * (losses_batch - R)) + np.finfo(np.float64).eps
                    ),
                    axis=0,
                )
                K_R = np.max(K_R, axis=0)
                K_R_max += K_R

            r_hat_plus_tronc = lambdas[
                np.argwhere(np.cumsum(K_R_max > -np.log(delta), axis=1) == 1)[:, 1]
            ]
            r_hat_plus = np.ones(n_lambdas)
            r_hat_plus[: len(r_hat_plus_tronc)] = r_hat_plus_tronc

    else:
        r_hat_plus = (n_obs / (n_obs + 1)) * r_hat + (1 / (n_obs + 1))

    return r_hat, r_hat_plus


def _check_risk_monotonicity(arr: NDArray, tol: float = 1e-6) -> str:
    diffs = np.diff(arr)

    if np.all(diffs >= -tol):
        return "increasing"

    if np.all(diffs <= tol):
        return "decreasing"

    return "none"


def _is_increasing_risk(r_hat_plus: NDArray) -> bool:
    """
    Internal function checking if a risk is increasing or not.

    Parameters
    ----------
    r_hat_plus: NDArray of shape (n_lambdas, )
        Upper bounds computed in the `get_r_hat_plus` method.

    Returns
    -------
    bool
        True if array is increasing, False otherwise
    """
    return r_hat_plus[0] < r_hat_plus[-1]


def find_best_predict_param(
    lambdas: NDArray, r_hat_plus: NDArray, alpha_np: NDArray
) -> NDArray:
    """Find the higher value of lambda such that for
    all smaller lambda, the risk is smaller, for each value
    of alpha.

    Note on extreme cases:
    When all lambda values are valid:
    - if r_hat_plus is increasing, then return the last lambda
    - else return the first lambda.
    When no lambda values are valid:
    - if r_hat_plus is increasing, then return the first lambda
    - else return the last lambda.

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

    bound_rep = np.repeat(np.expand_dims(r_hat_plus, axis=0), len(alphas_np), axis=0)
    increasing_risk = _is_increasing_risk(r_hat_plus)

    arr = np.greater_equal(bound_rep, alphas_np).astype(int)
    if arr.min() == 1:
        warnings.warn(
            "The risk cannot be controlled. Returning the extreme value according to risk direction."
        )
    if not increasing_risk:
        arr = np.fliplr(arr)
    arr = arr.cumsum(axis=1)
    idx_last_zero = np.where(arr == 0, np.arange(arr.shape[1]), 0).max(axis=1)
    if not increasing_risk:
        idx_last_zero = len(lambdas) - 1 - idx_last_zero
    best_predict_param = lambdas[idx_last_zero]
    return best_predict_param


def ltt_procedure(
    r_hat: NDArray,
    alpha_np: NDArray,
    delta: float,
    n_obs: NDArray,
    binary: bool = False,
    fwer_method: Literal[
        "bonferroni",
        "fst_ascending",
        "sgt_bonferroni_holm",
    ] = "bonferroni",
    **fwer_kwargs,
) -> Tuple[List[List[Any]], NDArray]:
    """
    Apply the Learn-Then-Test procedure for risk control.
    Note that we will do a multiple test for ``r_hat`` that are
    less than level ``alpha_np``.
    The procedure follows the instructions in [1]:
        - Calculate p-values for each lambdas discretized
        - Apply a family wise error rate algorithm, here Bonferonni correction
        - Return the index lambdas that give you the control at alpha level

    Note that in the case of multi-risk, the arrays r_hat, alpha_np, and n_obs
    should have the same length for the first dimension which corresponds
    to the number of risks. In the case of a single risk, the length should be 1.

    Parameters
    ----------
    r_hat: NDArray of shape (n_risks, n_lambdas).
        Empirical risk with respect to the lambdas.
        Here lambdas are thresholds that impact decision-making,
        therefore empirical risk.

    alpha_np: NDArray of shape (n_risks, n_alpha).
        Contains the different alphas control level.
        The empirical risk should be less than alpha with
        probability 1-delta.
        Note: MAPIE 1.2 does not support multiple risks and multiple alphas
        simultaneously.
        For MultiLabelClassificationController, the shape should be (1, n_alpha).
        For BinaryClassificationController, the shape should be (n_risks, 1).

    delta: float.
        Probability of not controlling empirical risk.
        Correspond to proportion of failure we don't
        want to exceed.

    n_obs: NDArray of shape (n_risks, n_lambdas).
        Correspond to the number of observations used to compute the risk.
        In the case of a conditional loss, n_obs must be the
        number of effective observations used to compute the empirical risk
        for each lambda.

    binary: bool, default=False
        Must be True if the loss associated to the risk is binary.

    fwer_method : {"bonferroni", "fst_ascending", "sgt_bonferroni_holm"}, default="bonferroni"
        FWER control strategy.
    **fwer_kwargs
        Additional keyword arguments used only when ``fwer_method="fst_ascending"``.
        Currently supported keyword:
        - ``n_starts`` (int): number of equally spaced starting points used in
          the multi-start Fixed Sequence Testing procedure.

    Returns
    -------
    valid_index: List[List[Any]].
        Contain the valid index that satisfy FWER control
        for each alpha (length aren't the same for each alpha).

    p_values : NDArray of shape (n_lambdas, n_alpha)
        P-values associated with each tested parameter. In the multi-risk setting,
        they correspond to the maximum over the tested risks.

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".
    """
    if not (r_hat.shape[0] == n_obs.shape[0] == alpha_np.shape[0]):
        raise ValueError("r_hat, n_obs, and alpha_np must have the same length.")
    p_values = np.array(
        [
            compute_hoeffding_bentkus_p_value(r_hat_i, n_obs_i, alpha_np_i, binary)
            for r_hat_i, n_obs_i, alpha_np_i in zip(r_hat, n_obs, alpha_np)
        ]
    )
    p_values = p_values.max(axis=0)  # take max over risks (no effect if mono risk)

    # FST only supports a single monotonic risk.
    # - If non-monotonic: fallback to SGT when fwer_method="auto" in BCC, else error.
    # - If decreasing: reverse order so FST tests easiest→hardest;
    #   store permutation to remap indices afterward.
    order = None
    p_values_original = p_values
    _auto_selected = fwer_kwargs.pop("_auto_selected", False)
    if fwer_method == "fst_ascending":
        if r_hat.shape[0] > 1:
            raise ValueError("fst_ascending cannot be used with multiple risks.")

        direction = _check_risk_monotonicity(r_hat[0])

        if direction == "none":
            if _auto_selected:
                fwer_method = "sgt_bonferroni_holm"
            else:
                raise ValueError(
                    "fst_ascending requires a monotonic risk over lambdas."
                )

        if direction == "decreasing":
            order = np.arange(len(p_values))[::-1]
            p_values = p_values[order]

    valid_index = []
    for i in range(alpha_np.shape[1]):
        idx = control_fwer(p_values, delta, fwer_method=fwer_method, **fwer_kwargs)
        if order is not None:
            idx = order[idx]
        l_index = idx.tolist()
        valid_index.append(l_index)
    return valid_index, p_values_original


def compute_hoeffding_bentkus_p_value(
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
    r_hat_repeat = np.repeat(np.expand_dims(r_hat, axis=1), len(alpha_np), axis=1)
    alpha_repeat = np.repeat(alpha_np.reshape(1, -1), len(r_hat), axis=0)
    if isinstance(n_obs, int):
        n_obs = np.full_like(r_hat, n_obs, dtype=float)
    n_obs_repeat = np.repeat(np.expand_dims(n_obs, axis=1), len(alpha_np), axis=1)

    hoeffding_p_value = np.exp(
        -n_obs_repeat
        * _h1(
            np.where(r_hat_repeat > alpha_repeat, alpha_repeat, r_hat_repeat),
            alpha_repeat,
        )
    )
    factor = 1 if binary else np.e
    bentkus_p_value = factor * binom.cdf(
        np.ceil(n_obs_repeat * r_hat_repeat), n_obs_repeat, alpha_repeat
    )
    hb_p_value = np.where(
        bentkus_p_value > hoeffding_p_value, hoeffding_p_value, bentkus_p_value
    )
    return hb_p_value


def _h1(r_hats: NDArray, alphas: NDArray) -> NDArray:
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


def find_precision_best_predict_param(
    r_hat: NDArray, valid_index: List[List[Any]], lambdas: NDArray
) -> Tuple[NDArray, ArrayLike]:
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
        Contain the valid index that satisfy FWER control
        for each alpha (length aren't the same for each alpha).

    lambdas: NDArray of shape (n_lambda, )
        Discretize parameters use for ltt procedure.

    Returns
    -------
    l_best_predict_param: NDArray of shape (n_alpha, ).
        The lambda that gives the minimum precision
        for a given alpha.

    r_star: NDArray of shape (n_alpha, ).
        The value of lowest risk for a given alpha.
    """
    if [] in valid_index:
        warnings.warn(
            """
            Warning: the risk couldn't be controlled for at least one value of alpha.
            The corresponding lambdas have been set to np.nan.
            """
        )
    l_best_predict_param = []  # type: List[Any]
    l_r_star = []  # type: List[Any]
    for i in range(len(valid_index)):
        if len(valid_index[i]) == 0:
            l_best_predict_param.append(np.nan)
            l_r_star.append(np.nan)
        else:
            idx_min = int(np.min(valid_index[i]))
            best_lambda = lambdas[idx_min]
            best_risk = r_hat[idx_min]
            l_best_predict_param.append(best_lambda)
            l_r_star.append(best_risk)

    return np.array(l_best_predict_param), l_r_star
