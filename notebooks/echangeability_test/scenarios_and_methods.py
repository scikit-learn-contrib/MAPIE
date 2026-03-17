import numpy as np
from scipy.stats import gaussian_kde
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

### Scenarios ###


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


### Methods ###


def risk_monitoring(X_to_test, y_to_test, X_train, y_train):
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
    t_warmup = 10  # initialisation to avoid warnings in the early iterations: need to check the original code on how they handle that
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

    return (
        not any(shift_detected_history),
        upper_bound_source_risk + tol,
        lower_bound_target_risk_history,
    )


# SimpleMixitureMartingle and PlugInMartingale
def _compute_alpha(z_i, history):
    """
    Compute nonconformity score alpha_i.
    """
    x_i, y_i = z_i

    if len(history) == 0:
        return 1.0

    dist_same_label = []
    dist_diff_label = []

    for x_j, y_j in history:
        d = np.linalg.norm(x_i - x_j)
        if y_i == y_j:
            dist_same_label.append(d)
        else:
            dist_diff_label.append(d)

    min_same = min(dist_same_label) if dist_same_label else 1e-12
    min_diff = min(dist_diff_label) if dist_diff_label else 1e-12

    return min_same / (min_diff + 1e-12)


def _compute_p_value(alpha_i, previous_alphas):
    """
    Compute p-value according to Algorithm 1 of Vovk et al. 2012.
    p_i = ( #{alpha_j > alpha_i} + θ #{alpha_j = alpha_i} ) / n
    where theta ~ Uniform(0,1).
    """

    n = len(previous_alphas)

    if n == 0:
        return 0.5

    previous_alphas = np.asarray(previous_alphas)

    greater = np.sum(previous_alphas > alpha_i)
    equal = np.sum(previous_alphas == alpha_i)

    theta = np.random.rand()

    p_values = (greater + theta * equal) / n
    return np.clip(p_values, 1e-12, 1 - 1e-12)


def _estimate_density(p, p_history):
    """
    Estimate density of p using reflected KDE and normalize
    """

    # neutral start when insufficient data
    if len(p_history) < 50:
        return 1.0

    p_array = np.asarray(p_history)

    # reflection to reduce boundary bias
    reflected = np.concatenate([-p_array, p_array, 2 - p_array])

    kde = gaussian_kde(reflected, bw_method="silverman")

    # compute normalization constant over [0,1]
    grid = np.linspace(0, 1, 200)
    density_vals = kde(grid)
    normalization_val = np.trapezoid(density_vals, grid)

    # control against numerical failure
    if normalization_val <= 0 or not np.isfinite(normalization_val):
        return 1.0

    # density value at p
    density = kde([p])[0] / normalization_val

    # enforce support restriction
    if p < 0 or p > 1:
        density = 0.0

    return max(float(density), 1e-12)


def simple_mixture_martingale_test(X_to_test, y_to_test, threshold=0.01, **kwargs):
    X_train = kwargs.get("X_train", None)
    y_train = kwargs.get("y_train", None)

    epsilon_grid = np.linspace(0, 1, 200)
    power_martingale_by_eps = np.ones_like(epsilon_grid, dtype=float)
    current_martingale_value = 1.0

    dataset = list(zip(np.array(X_to_test), np.array(y_to_test)))

    history = []
    alpha_history = []

    martingale_values = []

    for z_i in dataset:
        # Step 1: compute the conformity score alpha_i
        alpha_i = _compute_alpha(z_i, history)

        # Step 2: compute p-value p_i
        p_i = _compute_p_value(alpha_i, alpha_history)
        p_i = np.clip(p_i, 1e-12, 1 - 1e-12)

        # Step 3: update martingales
        ## multiplicative update for each \\varepsilon
        if len(alpha_history) < 50:
            power_martingale_by_eps *= 1
        else:
            power_martingale_by_eps *= epsilon_grid * (p_i ** (epsilon_grid - 1))

        ## numerical integral over \\varepsilon
        current_martingale_value = np.trapezoid(power_martingale_by_eps, epsilon_grid)

        martingale_values.append(current_martingale_value)

        # Step 4: update histories
        history.append(z_i)
        alpha_history.append(alpha_i)

    is_exchangeable = int(martingale_values[-1] < 1 / threshold)

    return is_exchangeable, threshold, martingale_values


def plugin_martingale_test(X_to_test, y_to_test, threshold=0.01, **kwargs):
    X_train = kwargs.get("X_train", None)
    y_train = kwargs.get("y_train", None)

    p_history = []
    current_martingale_value = 1.0
    dataset = list(zip(np.array(X_to_test), np.array(y_to_test)))

    history = []
    alpha_history = []
    martingale_values = []

    for z_i in dataset:
        # Step 1: compute the conformity score alpha_i
        alpha_i = _compute_alpha(z_i, history)

        # Step 2: compute p-value p_i
        p_i = _compute_p_value(alpha_i, alpha_history)
        p_i = np.clip(p_i, 1e-12, 1 - 1e-12)

        # Step 3: update martingales

        rho_hat = _estimate_density(p_i, p_history)
        current_martingale_value *= rho_hat
        p_history.append(p_i)

        martingale_values.append(current_martingale_value)

        # Step 4: update histories
        history.append(z_i)
        alpha_history.append(alpha_i)

    is_exchangeable = int(martingale_values[-1] < 1 / threshold)

    return is_exchangeable, threshold, martingale_values
