import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammainc, gammaln


class GammaExponentialMixtureBound:
    """
    Python translation of the gamma-exponential mixture bound from ``confseq``.

    This implementation is adapted from the ``gamma_exponential_mixture_bound``
    functionality in the `confseq` package by Steven R. Howard, Ian
    Waudby-Smith, and Aaditya Ramdas. See ``THIRD_PARTY_NOTICES.md`` for the
    corresponding MIT license notice.

    References
    ----------
    Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021).
    "Time-uniform, nonparametric, nonasymptotic confidence sequences."
    The Annals of Statistics, 49(2), 1055-1080.

    Howard, S. R., Waudby-Smith, I., and Ramdas, A. (2021--).
    ``ConfSeq``: software for confidence sequences and uniform boundaries.
    https://github.com/gostevehoward/confseq
    """

    def __init__(self, v_opt: float, c: float, alpha_opt: float = 0.05) -> None:
        if v_opt <= 0:
            raise ValueError("v_opt must be > 0")
        if c <= 0:
            raise ValueError("c must be > 0")
        if not (0.0 < alpha_opt < 0.5):
            raise ValueError("alpha_opt must be in (0, 0.5)")

        self.v_opt = float(v_opt)
        self.c = float(c)
        self.alpha_opt = float(alpha_opt)
        self.rho: float = self._one_sided_best_rho(self.v_opt, self.alpha_opt)
        self.c_sq = self.c * self.c
        self.rho_c_sq = self.rho / self.c_sq
        self.leading_constant = (
            self.rho_c_sq * math.log(self.rho_c_sq)
            - gammaln(self.rho_c_sq)
            - self._log_lower_regularized_gamma(self.rho_c_sq, self.rho_c_sq)
        )

    @staticmethod
    def _two_sided_best_rho(v: float, alpha: float) -> float:
        if v <= 0:
            raise ValueError("v must be > 0")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        log_term = math.log(1.0 / alpha)
        return v / (2.0 * log_term + math.log1p(2.0 * log_term))

    @classmethod
    def _one_sided_best_rho(cls, v: float, alpha: float) -> float:
        return cls._two_sided_best_rho(v, 2.0 * alpha)

    @staticmethod
    def _log_lower_regularized_gamma(a: float, x: float) -> float:
        value = float(gammainc(a, x))
        if value <= 0.0:
            return float("-inf")
        return math.log(value)

    def log_supermg(self, s: float, v: float) -> float:
        if v < 0:
            raise ValueError("v must be >= 0")

        cs_v_csq = (self.c * s + v) / self.c_sq
        v_rho_csq = (v + self.rho) / self.c_sq
        x = cs_v_csq + self.rho_c_sq

        return float(
            self.leading_constant
            + gammaln(v_rho_csq)
            + self._log_lower_regularized_gamma(v_rho_csq, x)
            - v_rho_csq * math.log(x)
            + cs_v_csq
        )

    def bound(
        self, v: float, alpha: float, tol: float = 1e-12, max_iter: int = 200
    ) -> float:
        if v < 0:
            raise ValueError("v must be >= 0")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")

        log_threshold = math.log(1.0 / alpha)

        def root_fn(s: float) -> float:
            return self.log_supermg(s, v) - log_threshold

        upper = max(v, 1.0)
        for _ in range(50):
            if root_fn(upper) > 0.0:
                break
            upper *= 2.0
        else:
            raise RuntimeError("Failed to find an upper limit for the mixture bound")

        lo = 0.0
        hi = upper
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if root_fn(mid) > 0.0:
                hi = mid
            else:
                lo = mid

            if hi - lo <= tol * max(1.0, hi):
                break

        return 0.5 * (lo + hi)

    def __call__(
        self, v: float, alpha: float, tol: float = 1e-12, max_iter: int = 200
    ) -> float:
        return self.bound(v=v, alpha=alpha, tol=tol, max_iter=max_iter)


def hoeffding_bound(
    empirical_risk_sequence: ArrayLike,
    delta: float,
    bound_side: Literal["upper", "lower"] = "upper",
) -> float:
    """
    Predictably-mixed Hoeffding's (PM-H) risk bound.

    Parameters
    ----------
    empirical_risk_sequence : array-like
        Empirical risk estimates.
    delta : float
        Confidence level complement.
    bound_side : {"upper", "lower"}
        Side of the bound to return.
    Returns
    -------
    bound : float
        Lower or upper risk bound.
    """

    risk_sequence = np.asarray(empirical_risk_sequence, dtype=np.float64)
    num_observations = risk_sequence.shape[0]
    if num_observations == 0:
        raise ValueError("empirical_risk_sequence must contain at least one value.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")

    empirical_mean = float(np.mean(risk_sequence))

    radius = np.sqrt(np.log(1 / delta) / (2 * num_observations))

    if bound_side == "lower":
        return float(empirical_mean - radius)
    elif bound_side == "upper":
        return float(empirical_mean + radius)
    else:
        raise ValueError("bound_side must be either 'upper' or 'lower'.")


def conjugate_mixture_empirical_bernstein_bound(
    empirical_risk_sequence: ArrayLike,
    v_opt: float,
    alpha: float = 0.05,
    bound_side: Literal["upper", "lower"] = "upper",
    running_intersection: bool = True,
) -> NDArray[np.float64]:
    """
    Conjugate mixture empirical Bernstein (CM-EB) bound.

    Parameters
    ----------
    empirical_risk_sequence, array-like of reals
        Sequence of observed risks.
    v_opt, positive real
        Intrinsic time at which to optimize the confidence bound.
        For example, if the variance is given by sigma, and one
        wishes to optimize for time t, then v_opt = t*sigma^2.
    alpha, (0, 1)-valued real
        Significance level
    bound_side, {"upper", "lower"}
        Side of the bound to return.
    running_intersection, boolean
        Should the running intersection be taken?
    Returns
    -------
    bound, array-like of reals
        Lower or upper confidence bound sequence.
    """
    risk_sequence = np.asarray(empirical_risk_sequence, dtype=np.float64)
    if risk_sequence.size == 0:
        raise ValueError("empirical_risk_sequence must contain at least one value.")
    t = np.arange(1, len(risk_sequence) + 1)
    S_t = np.cumsum(risk_sequence)
    mu_hat_t = S_t / t
    mu_hat_tminus1 = np.append(1 / 2, mu_hat_t[0 : (len(mu_hat_t) - 1)])
    V_t = np.cumsum(np.power(risk_sequence - mu_hat_tminus1, 2))
    mixture_bound = GammaExponentialMixtureBound(v_opt=v_opt, c=1, alpha_opt=alpha / 2)
    boundary = (
        np.array([mixture_bound(v=float(v_t), alpha=alpha / 2) for v_t in V_t]) / t
    )
    if bound_side == "lower":
        bound = np.maximum(mu_hat_t - boundary, 0)
        if running_intersection:
            bound = np.maximum.accumulate(bound)
    elif bound_side == "upper":
        bound = np.minimum(mu_hat_t + boundary, 1)
        if running_intersection:
            bound = np.minimum.accumulate(bound)
    else:
        raise ValueError("bound_side must be either 'upper' or 'lower'.")

    return np.asarray(bound, dtype=np.float64)
