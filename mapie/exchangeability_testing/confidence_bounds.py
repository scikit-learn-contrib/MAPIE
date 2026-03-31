import math

import numpy as np
from confseq.boundaries import gamma_exponential_mixture_bound
from scipy.special import gammainc, gammaln


class GammaExponentialMixtureBound:
    """Standalone Python version of the gamma-exponential mixture bound."""

    def __init__(self, v_opt: float, c: float, alpha_opt: float = 0.05):
        if v_opt <= 0:
            raise ValueError("v_opt must be > 0")
        if c <= 0:
            raise ValueError("c must be > 0")
        if not (0.0 < alpha_opt < 0.5):
            raise ValueError("alpha_opt must be in (0, 0.5)")

        self.v_opt = float(v_opt)
        self.c = float(c)
        self.alpha_opt = float(alpha_opt)
        self.rho = self._one_sided_best_rho(self.v_opt, self.alpha_opt)
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

        return (
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


def hoeffding_upper_limit(seq, delta):
    """
    Predictably-mixed Hoeffding's (PM-H) confidence sequence.
    """
    n = len(seq)

    emp_mean = np.mean(seq)
    return emp_mean + np.sqrt(np.log(1 / delta) / (2 * n))


def conjmix_empbern_cs(x, v_opt, alpha=0.05, running_intersection=True):
    """
    Conjugate mixture empirical Bernstein (CM-EB) confidence sequence
    Parameters
    ----------
    x, array-like of reals
        The observed data
    v_opt, positive real
        Intrinsic time at which to optimize the confidence sequence.
        For example, if the variance is given by sigma, and one
        wishes to optimize for time t, then v_opt = t*sigma^2.
    alpha, (0, 1)-valued real
        Significance level
    running_intersection, boolean
        Should the running intersection be taken?
    Returns
    -------
    l, array-like of reals
        Lower confidence sequence
    u, array-like of reals
        Upper confidence sequence
    """
    x = np.array(x)
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)
    mu_hat_t = S_t / t
    mu_hat_tminus1 = np.append(1 / 2, mu_hat_t[0 : (len(mu_hat_t) - 1)])
    V_t = np.cumsum(np.power(x - mu_hat_tminus1, 2))
    bdry = (
        gamma_exponential_mixture_bound(
            V_t, alpha=alpha / 2, v_opt=v_opt, c=1, alpha_opt=alpha / 2
        )
        / t
    )
    lower, upper = mu_hat_t - bdry, mu_hat_t + bdry
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, 1)
    if running_intersection:
        lower = np.maximum.accumulate(lower)
        upper = np.minimum.accumulate(upper)

    return lower, upper
