import numpy as np


def sample_two_gaussians(
    n_samples=500,
    mean0=(0.0, 0.0),
    mean1=(1.8, 1.8),
    cov=None,
    random_state=None,
):
    if cov is None:
        cov = np.eye(2) * 0.5

    mean0 = np.asarray(mean0)
    mean1 = np.asarray(mean1)
    rng = np.random.RandomState(random_state)
    y = rng.randint(0, 2, size=n_samples)
    X = np.empty((n_samples, 2))

    mask0 = y == 0
    mask1 = ~mask0
    X[mask0] = rng.multivariate_normal(mean0, cov, size=mask0.sum())
    X[mask1] = rng.multivariate_normal(mean1, cov, size=mask1.sum())
    return X, y


def generate_gaussian_stream(
    n_samples=800,
    shift_type="stable",
    mean0_before=(0.0, 0.0),
    mean0_after=(0.0, -7.5),
    mean1_before=(1.8, 1.8),
    mean1_after=(1.8, -5.5),
    cov=None,
    prop_shift=0.5,
    random_state=None,
):
    if cov is None:
        cov = np.eye(2) * 0.5

    mean0_before = np.asarray(mean0_before)
    mean0_after = np.asarray(mean0_after)
    mean1_before = np.asarray(mean1_before)
    mean1_after = np.asarray(mean1_after)

    rng = np.random.RandomState(random_state)
    y = rng.randint(0, 2, size=n_samples)
    X = np.empty((n_samples, 2))
    shift_start = int(n_samples * (1 - prop_shift))

    for i, label in enumerate(y):
        mean0_t = mean0_before
        mean1_t = mean1_before

        if shift_type == "abrupt" and i >= shift_start:
            mean0_t = mean0_after
            mean1_t = mean1_after
        elif shift_type == "slow" and i >= shift_start:
            frac = (i - shift_start) / max(1, n_samples - shift_start - 1)
            mean0_t = (1 - frac) * mean0_before + frac * mean0_after
            mean1_t = (1 - frac) * mean1_before + frac * mean1_after
        elif shift_type != "stable" and shift_type not in {"abrupt", "slow"}:
            raise ValueError("shift_type must be 'stable', 'abrupt' or 'slow'.")

        mean = mean0_t if label == 0 else mean1_t
        X[i] = rng.multivariate_normal(mean, cov)

    return X, y
