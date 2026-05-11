from math import pi

import flintypy
import numpy as np
from numpy.typing import ArrayLike
from online_cp import ConformalRidgeRegressor, PluginMartingale
from online_cp.classifiers import ConformalNearestNeighboursClassifier
from online_cp.martingale import SimpleJumper
from scipy.spatial.distance import pdist
from scipy.stats import binom
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.stattools import durbin_watson

THRESHOLD = 0.05
### Scenarios ###


def generate_exchangeable_gaussian(
    n_samples=1000,
    mean0=np.array([0, 0]),
    mean1=np.array([2, 2]),
    rho=0.3,
    random_state=None,
):
    """
    Exchangeable Gaussian data with correlated features.

    Covariance has constant correlation rho.
    """

    is_exchangeable_ground_truth = True

    rng = np.random.RandomState(random_state)

    cov = np.array([[1, rho], [rho, 1]])

    # Train
    Xtr, ytr = [], []
    for _ in range(n_samples):
        label = rng.randint(0, 2)
        mu = mean0 if label == 0 else mean1
        Xtr.append(rng.multivariate_normal(mu, cov))
        ytr.append(label)

    # Test
    Xtt, ytt = [], []
    for _ in range(n_samples):
        label = rng.randint(0, 2)
        mu = mean0 if label == 0 else mean1
        Xtt.append(rng.multivariate_normal(mu, cov))
        ytt.append(label)

    return (
        is_exchangeable_ground_truth,
        np.array(Xtt),
        np.array(ytt),
        np.array(Xtr),
        np.array(ytr),
    )


def generate_two_gaussian_abrupt_shift(
    n_samples=1000,
    mean0_before=np.array([0, 0]),
    mean0_after=np.array([0, -5]),
    mean1_before=np.array([2, 2]),
    mean1_after=np.array([2, -3]),
    cov=None,
    prop_shift=0.5,
    random_state=None,
):
    """
    Generates two Gaussian classes. Train has no shift; to_test has abrupt shift in the middle.

    Parameters
    ----------
    n_samples : int
        Number of samples for both train and to_test (each gets n_samples).
    mean0_before, mean1_before : array
        Means for class 0 and 1 (before shift).
    mean0_after, mean1_after : array
        Means for class 0 and 1 (after shift, used in second half of to_test).
    cov : np.ndarray or None
        Covariance matrix.
    prop_shift : float
        Proportion of to_test that has the abrupt shift (second half). 0.5 = shift in the middle.
    random_state : int or None
        Seed for RNG.

    Returns
    -------
    is_exchangeable_ground_truth, X_to_test, y_to_test, X_train, y_train
    """
    is_exchangeable_ground_truth = False

    if cov is None:
        cov = np.eye(2) * 0.4
    rng = np.random.RandomState(random_state)
    n_train = n_samples
    n_to_test = n_samples
    n_no_shift = int(n_to_test * (1 - prop_shift))
    n_shift = n_to_test - n_no_shift

    # Train: no shift
    Xtr = []
    ytr = []
    for _ in range(n_train):
        label = rng.randint(0, 2)
        mu = mean0_before if label == 0 else mean1_before
        Xtr.append(rng.multivariate_normal(mu, cov))
        ytr.append(label)

    # To test: first part no shift, then abrupt shift (shift in the middle)
    Xtt = []
    ytt = []
    for _ in range(n_no_shift):
        label = rng.randint(0, 2)
        mu = mean0_before if label == 0 else mean1_before
        Xtt.append(rng.multivariate_normal(mu, cov))
        ytt.append(label)
    for _ in range(n_shift):
        label = rng.randint(0, 2)
        mu = mean0_after if label == 0 else mean1_after
        Xtt.append(rng.multivariate_normal(mu, cov))
        ytt.append(label)

    return (
        is_exchangeable_ground_truth,
        np.array(Xtt),
        np.array(ytt),
        np.array(Xtr),
        np.array(ytr),
    )


def generate_two_gaussian_slow_shift(
    n_samples=1000,
    mean0_start=np.array([0, 0]),
    mean0_end=np.array([0, -5]),
    mean1_start=np.array([2, 2]),
    mean1_end=np.array([2, -3]),
    cov=None,
    prop_shift=0.5,
    random_state=None,
):
    """
    Generates two-class Gaussian data. Train has no shift; to_test has slow linear drift in the middle.

    Parameters
    ----------
    n_samples : int
        Number of samples for both train and to_test (each gets n_samples).
    mean0_start, mean0_end, mean1_start, mean1_end : arrays
        Start and end means for each class; drift happens in second half of to_test.
    cov : np.ndarray or None
        Covariance matrix.
    prop_shift : float
        Proportion of to_test that has the slow drift (second half). 0.5 = drift in the middle.
    random_state : int or None
        RNG seed.

    Returns
    -------
    is_exchangeable_ground_truth, X_to_test, y_to_test, X_train, y_train
    """
    is_exchangeable_ground_truth = False

    if cov is None:
        cov = np.eye(2) * 0.4
    rng = np.random.RandomState(random_state)
    n_train = n_samples
    n_to_test = n_samples
    n_no_shift = int(n_to_test * (1 - prop_shift))
    n_shift = n_to_test - n_no_shift

    # Train: no shift
    Xtr = []
    ytr = []
    for _ in range(n_train):
        label = rng.randint(0, 2)
        mu = mean0_start if label == 0 else mean1_start
        Xtr.append(rng.multivariate_normal(mu, cov))
        ytr.append(label)

    # To test: first part no shift, then slow linear drift (shift in the middle)
    Xtt = []
    ytt = []
    for _ in range(n_no_shift):
        label = rng.randint(0, 2)
        mu = mean0_start if label == 0 else mean1_start
        Xtt.append(rng.multivariate_normal(mu, cov))
        ytt.append(label)
    for t in range(n_shift):
        frac = t / max(1, n_shift - 1)
        mean0_t = mean0_start * (1 - frac) + mean0_end * frac
        mean1_t = mean1_start * (1 - frac) + mean1_end * frac
        label = rng.randint(0, 2)
        mu = mean0_t if label == 0 else mean1_t
        Xtt.append(rng.multivariate_normal(mu, cov))
        ytt.append(label)

    return (
        is_exchangeable_ground_truth,
        np.array(Xtt),
        np.array(ytt),
        np.array(Xtr),
        np.array(ytr),
    )


def generate_var1_dependence(
    n_samples=1000,
    A=np.array([[0.7, 0.2], [0.1, 0.6]]),
    noise_scale=0.3,
    random_state=None,
):
    """
    Stationary VAR(1) process.
    """

    is_exchangeable_ground_truth = False
    rng = np.random.RandomState(random_state)

    burnin = 500
    n_train = int(n_samples * 0.3)

    total_steps = burnin + n_train + n_samples

    X_full = np.zeros((total_steps, 2))
    y_full = np.zeros(total_steps, dtype=int)

    x_t = rng.normal(size=2)

    for t in range(total_steps):
        noise = rng.normal(scale=noise_scale, size=2)
        x_t = A @ x_t + noise

        X_full[t] = x_t

        score = (
            np.sin(0.5 * x_t[0])
            + np.cos(0.5 * x_t[1])
            + 0.5 * np.sin(0.01 * t)
            + 5 * x_t[0]
        )
        prob = 1 / (1 + np.exp(-score))
        y_full[t] = rng.binomial(1, prob)

    # remove burn-in
    X_full = X_full[burnin:]
    y_full = y_full[burnin:]

    # split
    Xtr = X_full[:n_train]
    ytr = y_full[:n_train]

    Xtt = X_full[n_train:]
    ytt = y_full[n_train:]

    return (
        is_exchangeable_ground_truth,
        Xtt,
        ytt,
        Xtr,
        ytr,
    )


def generate_complex_dependence(
    n_samples=1000,
    A=np.array([[0.6, 0.2], [0.2, 0.5]]),
    shift_vector=np.array([0, 5]),
    prop_shift=0.5,
    noise_scale=0.3,
    random_state=None,
):
    """
    VAR(1) + nonlinear trend + abrupt shift (only in test).
    """

    is_exchangeable_ground_truth = False
    rng = np.random.RandomState(random_state)

    burnin = 500
    n_train = int(n_samples * 0.3)

    total_steps = burnin + n_train + n_samples

    shift_start = burnin + n_train + int(n_samples * (1 - prop_shift))

    X_full = np.zeros((total_steps, 2))
    y_full = np.zeros(total_steps, dtype=int)

    x_t = rng.normal(size=2)

    for t in range(total_steps):
        noise = rng.normal(scale=noise_scale, size=2)

        # VAR
        x_t = A @ x_t + noise

        # Add tendance
        x_t = x_t + 0.002 * t

        # shift only in test
        if t >= shift_start:
            x_t = x_t + shift_vector

        X_full[t] = x_t

        # Probabilistic label with noise
        score = np.sin(0.5 * x_t[0]) + np.cos(0.5 * x_t[1]) + 0.5 * np.sin(0.01 * t)
        prob = 1 / (1 + np.exp(-score))
        y_full[t] = rng.binomial(1, prob)

    # remove burn-in
    X_full = X_full[burnin:]
    y_full = y_full[burnin:]

    # split
    Xtr = X_full[:n_train]
    ytr = y_full[:n_train]

    Xtt = X_full[n_train:]
    ytt = y_full[n_train:]

    return (
        is_exchangeable_ground_truth,
        Xtt,
        ytt,
        Xtr,
        ytr,
    )


### Methods ###


def risk_monitoring(X_to_test, y_to_test, X_train, y_train, **kwargs):
    # Split train into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0
    )

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

    # Initialize parameters
    delta = 0.05  # desired type I error
    delta_source = delta / 2
    tol = 0

    # accuracy, risk, etc. on the source distribution
    empirical_source_risk = 1 - model.score(X_test, y_test)
    # PM-H bound
    upper_correction = np.sqrt(np.log(1 / delta_source) / (2 * len(X_test)))
    upper_bound_source_risk = empirical_source_risk + upper_correction

    # Run risk monitoring
    t_warmup = 100  # initialisation to avoid warnings in the early iterations: need to check the original code on how they handle that
    lower_bound_target_risk_history = []
    shift_detected_history = []
    for t in range(t_warmup, len(X_to_test)):
        if t == t_warmup:
            empirical_target_risk_t = 1 - model.score(X_to_test[:t], y_to_test[:t])

        else:
            empirical_target_risk_t = (t - 1) / t * empirical_target_risk_t + 1 / t * (
                1 - model.score(X_to_test[t : t + 1], y_to_test[t : t + 1])
            )
        lower_correction = 0  # should compute CM-EB (using an external package) which handles time-varying mean (bound adapted to online data slowly drifting to non i.i.d)
        lower_bound_target_risk_t = empirical_target_risk_t - lower_correction

        if lower_bound_target_risk_t > upper_bound_source_risk + tol:
            shift_detected_history.append(True)
        else:
            shift_detected_history.append(False)

        if t == t_warmup:
            lower_bound_target_risk_history[:t_warmup] = [lower_bound_target_risk_t] * (
                t_warmup + 1
            )
        else:
            lower_bound_target_risk_history.append(lower_bound_target_risk_t)

    # Find the first index t where shift_detected_history[t] is True and remains True for all subsequent t
    shift_point = None
    for idx, detected in enumerate(shift_detected_history):
        if detected and all(shift_detected_history[idx:]):
            shift_point = idx + t_warmup  # account for offset in t
            break
    # Return shift_point as the fourth output (may be None if never True and stays True)
    is_exchangeable = int(not any(shift_detected_history))
    stable_index = shift_point
    stable_value = np.nan
    if stable_index is not None and 0 <= stable_index < len(
        lower_bound_target_risk_history
    ):
        stable_value = lower_bound_target_risk_history[stable_index]
    return is_exchangeable, stable_index, stable_value


# Plug-in Maringale and Simple Jumper Martingale
def _first_stable_suffix_index(martingale_values, boundary, use_below):
    """
    Return the first index of the final suffix that stays on one side
    of the boundary.
    """
    values = np.asarray(martingale_values)
    if values.size == 0:
        return None

    comparator = np.less if use_below else np.greater
    condition = comparator(values, boundary)
    if not condition[-1]:
        return None

    idx = len(condition) - 1
    while idx > 0 and condition[idx - 1]:
        idx -= 1
    return int(idx)


def martingale_test(
    X_to_test,
    y_to_test,
    threshold=THRESHOLD,
    martingale_type="plugin_martingale",
    task="classification",
    **kwargs,
):
    dataset = list(zip(np.array(X_to_test), np.array(y_to_test)))

    if martingale_type == "plugin_martingale":
        M = PluginMartingale(warnings=False)
    elif martingale_type == "simple_jumper_martingale":
        M = SimpleJumper(warnings=False)
    else:
        raise ValueError

    martingale_values = []

    if task == "regression":
        cp = ConformalRidgeRegressor()

        for x_i, y_i in dataset:
            p = cp.compute_p_value(x_i, y_i)
            cp.learn_one(x_i, y_i)

            M.update_martingale_value(p)
            martingale_values.append(np.exp(M.logM))

    elif task == "classification":
        cp = ConformalNearestNeighboursClassifier(k=1, label_space=np.unique(y_to_test))

        for x_i, y_i in dataset:
            _, p_val, D = cp.predict(x_i, return_p_values=True, return_update=True)
            p = p_val[y_i]

            cp.learn_one(x_i, y_i, D)

            M.update_martingale_value(p)
            martingale_values.append(np.exp(M.logM))

    else:
        raise ValueError

    boundary = 1 / threshold
    is_exchangeable = int(martingale_values[-1] < boundary)
    stable_index = _first_stable_suffix_index(
        martingale_values,
        boundary=boundary,
        use_below=bool(is_exchangeable),
    )

    stable_value = (
        martingale_values[stable_index] if stable_index is not None else np.nan
    )
    return is_exchangeable, stable_index, stable_value


def plugin_martingale_test(X_to_test, y_to_test, **kwargs):
    is_exchangeable, stable_index, stable_value = martingale_test(
        X_to_test,
        y_to_test,
        threshold=THRESHOLD,
        martingale_type="plugin_martingale",
        task="classification",
        **kwargs,
    )
    return is_exchangeable, stable_index, stable_value


def simple_jumper_martingale_test(X_to_test, y_to_test, **kwargs):
    is_exchangeable, stable_index, stable_value = martingale_test(
        X_to_test,
        y_to_test,
        threshold=THRESHOLD,
        martingale_type="simple_jumper_martingale",
        task="classification",
        **kwargs,
    )
    return is_exchangeable, stable_index, stable_value


class ContinuousPairwiseBettingMartingale:
    def __init__(self):
        self.martingale = [1]
        self.num_coeff = 0
        self.denum_coeff = 0
        self.sum_sigma = 0
        self.last_observed = None

    @property
    def parameters(self):
        return {
            "coeff": self.num_coeff / self.denum_coeff,
            "var": 1 / (len(self.martingale) * 2 - 2) * self.sum_sigma,
        }

    @staticmethod
    def validate_sequence_shape(X: ArrayLike):
        if not (len(X.shape) == 1):
            raise ValueError("Sequence must be a one dimensionnal iterable")

    def update_parameters(self, X1, X2):
        self.num_coeff += X1 * X2
        self.denum_coeff += X2**2
        self.sum_sigma += (X2 - self.num_coeff / self.denum_coeff * X1) ** 2
        return self

    def null_hypothesis_likelihood(self, X1, X2):
        return 1 / 2

    def alternative_hypothesis_likelihood(self, X1, X2):
        params = self.parameters
        var = params["var"]
        coeff = params["coeff"]

        num = 1 / (2 * pi * var)
        expo1 = -1 / (2 * var) * (X1 - coeff * self.last_observed) ** 2
        expo2 = -1 / (2 * var) * (X2 - coeff * X1) ** 2
        num *= np.exp(expo1 + expo2)
        denum = 1 / (2 * pi * var)
        expo1 = -1 / (2 * var) * (X2 - coeff * self.last_observed) ** 2
        expo2 = -1 / (2 * var) * (X1 - coeff * X2) ** 2
        denum *= np.exp(expo1 + expo2)
        denum += num

        return num / denum

    def bet(self, X1, X2):
        if len(self.martingale) > 1:
            return self.alternative_hypothesis_likelihood(
                X1, X2
            ) / self.null_hypothesis_likelihood(X1, X2)
        return 1

    def run_martingale(self, X: ArrayLike):
        self.validate_sequence_shape(X)

        for i in np.arange(1, len(X) - 1, step=2):
            X2 = X[i]
            X1 = X[i - 1]
            m = 1
            if self.last_observed is not None:
                m = self.martingale[-1] * self.bet(X1, X2)
                self.update_parameters(self.last_observed, X1)
            self.martingale.append(m)
            self.update_parameters(X1, X2)
            self.last_observed = X2

        return self


def continuous_pairwise_betting_martingale_test(
    X_to_test,
    y_to_test,
    X_train,
    y_train,
    task="classification",
    threshold=THRESHOLD,
    **kwargs,
):
    scores = _compute_non_conformity_score(X_to_test, y_to_test, X_train, y_train, task)
    M = ContinuousPairwiseBettingMartingale()
    M = M.run_martingale(scores)
    martingale_values = np.array(M.martingale)
    boundary = 1 / threshold
    is_exchangeable = int(martingale_values[-1] < boundary)
    stable_index = _first_stable_suffix_index(
        martingale_values,
        boundary=boundary,
        use_below=bool(is_exchangeable),
    )
    stable_value = (
        martingale_values[stable_index] if stable_index is not None else np.nan
    )
    return is_exchangeable, stable_index, stable_value


def _compute_non_conformity_score(X_to_test, y_to_test, X_train, y_train, task):
    """
    Compute non-conformity score for each sample in X_to_test.
    """
    if task == "regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        scores = np.abs(model.predict(X_to_test) - y_to_test)
    elif task == "classification":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores = (
            1 - model.predict_proba(X_to_test)[np.arange(len(X_to_test)), y_to_test]
        )
    return scores


def v_test_distance(
    X_to_test, y_to_test, X_train, y_train, task="classification", threshold=THRESHOLD
):
    scores = _compute_non_conformity_score(X_to_test, y_to_test, X_train, y_train, task)
    scores_2d = np.expand_dims(scores, axis=1)  # shape (N, 1)

    # Compute pairwise distances (e.g. Minkowski p=2 for Euclidean)
    dist_vec = pdist(scores_2d, metric="minkowski", p=2) ** 2  # l_2^2
    dist_list = [dist_vec]  # one block

    p_value = flintypy.v_stat.dist_data_p_value(dist_list, num_perms=1000)
    return int(p_value > threshold), None, p_value


def v_test(
    X_to_test, y_to_test, X_train, y_train, task="classification", threshold=THRESHOLD
):
    scores = _compute_non_conformity_score(X_to_test, y_to_test, X_train, y_train, task)
    scores = np.expand_dims(scores, axis=1)

    p_value = flintypy.v_stat.get_p_value(scores, large_p=True, num_perms=1000)

    return int(p_value > threshold), None, p_value


def v_test_2d(
    X_to_test, y_to_test, X_train, y_train, task="classification", threshold=THRESHOLD
):
    scores = _compute_non_conformity_score(X_to_test, y_to_test, X_train, y_train, task)
    scores_2d = np.column_stack([scores, np.ones_like(scores)])

    p_value = flintypy.v_stat.get_p_value(scores_2d, large_p=True, num_perms=1000)

    return int(p_value > threshold), None, p_value


def durbin_watson_test(X_to_test, y_to_test, X_train, y_train, task="classification"):
    # Note: it supposed to be a test for autocorrelation in the residuals of a regression
    # Returns 2 when there is no autocorrelation, and 0 or 4when there is autocorrelation.
    # Here we use it for non-conformity scores and return the absolute difference between the statistic and 2.
    threshold = 1
    scores = _compute_non_conformity_score(X_to_test, y_to_test, X_train, y_train, task)
    dw_stat = durbin_watson(scores)
    return abs(dw_stat - 2) < threshold, None, abs(dw_stat - 2)


def yule_walker_test(X_to_test, y_to_test, X_train, y_train, task="classification"):
    threshold = 0.1
    scores = _compute_non_conformity_score(X_to_test, y_to_test, X_train, y_train, task)
    rho, _ = yule_walker(scores)
    return abs(rho.item()) < threshold, None, abs(rho.item())


def _mean_or_nan(array: ArrayLike):
    array = np.asarray(array)
    if array.size == 0:
        return np.nan
    return float(np.mean(array))


def _compute_test_statistic_for_mc_trial_fixed(X, y, n_blocks=10):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n = X.shape[0]

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    unique_labels = np.unique(y)
    d = X.shape[1]

    # Encode labels → index
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    y_idx = np.array([label_to_idx[val] for val in y])
    L = len(unique_labels)

    # cumulative sums per label
    cumsum = np.zeros((n, L, d))
    counts = np.zeros((n, L))

    for i in range(n):
        if i > 0:
            cumsum[i] = cumsum[i - 1]
            counts[i] = counts[i - 1]
        cumsum[i, y_idx[i]] += X[i]
        counts[i, y_idx[i]] += 1

    total_sum = cumsum[-1]
    total_count = counts[-1]

    max_stat = 0.0

    for k in range(1, n - 1):
        left_sum = cumsum[k]
        left_count = counts[k]

        right_sum = total_sum - left_sum
        right_count = total_count - left_count

        # avoid division by zero
        left_mean = np.divide(
            left_sum,
            left_count[:, None],
            out=np.zeros_like(left_sum),
            where=left_count[:, None] > 0,
        )

        right_mean = np.divide(
            right_sum,
            right_count[:, None],
            out=np.zeros_like(right_sum),
            where=right_count[:, None] > 0,
        )

        diff = left_mean - right_mean

        # CUSUM scaling
        scale = np.sqrt(k * (n - k) / n)

        stat_k = scale * np.linalg.norm(diff)

        if stat_k > max_stat:
            max_stat = stat_k

    return max_stat


def sequential_mc_trial(
    X,
    y,
    B=1000,
    alpha=0.05,
    strategy="binomial",
    c=None,
    p_zero=None,
    random_state=None,
):
    """
    Run one sequential Monte-Carlo trial on a fixed dataset (X, y).

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Fixed numeric observations.
    y : array-like of shape (n_samples,)
        Binary group indicators. Values are cast to bool.
    B : int
        Number of permutations.
    alpha : float
        Significance level.
    random_state : int or None
        RNG seed for permutation randomness.

    strategy : {"binomial", "aggressive", "binomial_mixture"}
        Which martingale-like wealth process to return.

    Returns
    -------
    is_exchangeable : int
        1 if martingale_values[-1] < 1 / alpha, else 0.
    threshold : float
        Decision threshold (alpha).
    martingale_values : np.ndarray
        Wealth trajectory for the selected strategy.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same length.")

    y_bool = y.astype(bool)
    if y_bool.sum() == 0 or (~y_bool).sum() == 0:
        raise ValueError("y must define two non-empty groups.")

    if c is None:
        c = alpha * 0.90
    if p_zero is None:
        p_zero = 1 / np.ceil(np.sqrt(2 * np.pi * np.exp(1 / 6)) / alpha)
    valid_strategies = {"binomial", "aggressive", "binomial_mixture"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {sorted(valid_strategies)}."
        )

    test_stat = _compute_test_statistic_for_mc_trial_fixed(X, y_bool)

    rank = 1
    wealth_bin = np.array([1.0])
    wealth_agg = np.array([1.0])
    wealth_bm = np.array([])
    n = len(X)

    for i in range(1, B + 1):
        perm = rng.permutation(n)
        X_perm = X[perm]
        y_bool_perm = y_bool[perm]
        test_stat_perm = _compute_test_statistic_for_mc_trial_fixed(X_perm, y_bool_perm)

        if wealth_bin[-1] * p_zero * (i + 1) / rank < alpha:
            pt = 0
        else:
            pt = p_zero

        if test_stat_perm >= test_stat:
            bet_bin_i = pt * (i + 1) / rank
            bet_agg_i = 0.0
            rank += 1
        else:
            bet_bin_i = (1 - pt) * (i + 1) / (i - (rank - 1))
            bet_agg_i = (i + 1) / i

        wealth_bin = np.append(wealth_bin, wealth_bin[-1] * bet_bin_i)

        wealth_agg = np.append(wealth_agg, wealth_agg[-1] * bet_agg_i)

        wealth_bm_i = (1 - binom.cdf(rank - 1, i + 1, c)) / c
        wealth_bm = np.append(wealth_bm, wealth_bm_i)

        # early stopping if possible
        if (
            strategy == "aggressive"
            and (wealth_agg[-1] < alpha or wealth_agg[-1] >= 1 / alpha)
            and (i > 50)
        ):
            break
        if (
            strategy == "binomial"
            and (wealth_bin[-1] < alpha or wealth_bin[-1] >= 1 / alpha)
            and (i > 50)
        ):
            break
        if (
            strategy == "binomial_mixture"
            and (wealth_agg[-1] < alpha or wealth_agg[-1] >= 1 / alpha)
            and (i > 50)
        ):
            break

    if strategy == "binomial":
        martingale_values = np.asarray(wealth_bin)
    elif strategy == "aggressive":
        martingale_values = np.asarray(wealth_agg)
    elif strategy == "binomial_mixture":
        # wealth_bm has length B while others have length B+1
        martingale_values = np.concatenate([[1.0], np.asarray(wealth_bm)])
    else:
        raise ValueError

    is_exchangeable = int(martingale_values[-1] < 1 / alpha)
    if is_exchangeable:
        stable_permutation_index = _first_stable_suffix_index(
            martingale_values,
            boundary=alpha,
            use_below=True,
        )
    else:
        stable_permutation_index = _first_stable_suffix_index(
            martingale_values,
            boundary=1 / alpha,
            use_below=False,
        )
    stable_value = (
        martingale_values[stable_permutation_index]
        if stable_permutation_index is not None
        else np.nan
    )
    return is_exchangeable, stable_permutation_index, stable_value


def binomial_martingale_test(X_to_test, y_to_test, X_train, y_train, **kwargs):
    """
    Fixed-dataset sequential MC test with binomial strategy.
    """
    return sequential_mc_trial(X_to_test, y_to_test, strategy="binomial", **kwargs)


def aggressive_martingale_test(X_to_test, y_to_test, X_train, y_train, **kwargs):
    """
    Fixed-dataset sequential MC test with aggressive strategy.
    """
    return sequential_mc_trial(X_to_test, y_to_test, strategy="aggressive", **kwargs)


def binomial_mixture_martingale_test(X_to_test, y_to_test, X_train, y_train, **kwargs):
    """
    Fixed-dataset sequential MC test with binomial mixture strategy.
    """
    return sequential_mc_trial(
        X_to_test, y_to_test, strategy="binomial_mixture", **kwargs
    )


def permutation_pvalue(
    X_to_test,
    y_to_test,
    X_train=None,
    y_train=None,
    B=1000,
    threshold=THRESHOLD,
    random_state=None,
):
    """
    Classical permutation test on fixed (X, y) with API-compatible output.

    Returns
    -------
    is_exchangeable : int
        1 if the final p-value is above threshold, else 0.
    threshold : float
        p-value decision threshold.
    p_values : np.ndarray
        Running permutation p-values p_t after t permutations.
    """
    rng = np.random.RandomState(random_state)
    X = np.asarray(X_to_test)
    y_bool = np.asarray(y_to_test).astype(bool)
    if X.shape[0] != y_bool.shape[0]:
        raise ValueError("X and y must have the same length.")
    if y_bool.sum() == 0 or (~y_bool).sum() == 0:
        raise ValueError("y must define two non-empty groups.")

    test_stat = _compute_test_statistic_for_mc_trial_fixed(X, y_bool)
    rank = 1
    p_values = np.empty(B + 1)
    p_values[0] = 1.0
    n = len(X)
    for t in range(1, B + 1):
        perm = rng.permutation(n)
        X_perm = X[perm]
        y_bool_perm = y_bool[perm]
        stat_perm = _compute_test_statistic_for_mc_trial_fixed(X_perm, y_bool_perm)
        if stat_perm >= test_stat:
            rank += 1
        p_values[t] = rank / (t + 1)

    is_exchangeable = int(p_values[-1] > threshold)
    if is_exchangeable:
        stable_permutation_index = _first_stable_suffix_index(
            p_values,
            boundary=threshold,
            use_below=False,
        )
    else:
        stable_permutation_index = _first_stable_suffix_index(
            p_values,
            boundary=threshold,
            use_below=True,
        )
    stable_value = (
        p_values[stable_permutation_index]
        if stable_permutation_index is not None
        else np.nan
    )
    return is_exchangeable, stable_permutation_index, stable_value
