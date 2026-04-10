import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(ax, X_online, y_online, title, shift_start=None):
    colors = {0: "tab:blue", 1: "tab:orange"}

    if shift_start is None:
        for label in [0, 1]:
            label_mask = y_online == label
            ax.scatter(
                X_online[label_mask, 0],
                X_online[label_mask, 1],
                color=colors[label],
                alpha=0.6,
                s=20,
                label=f"Class {label}",
            )
    else:
        before_shift = np.arange(len(y_online)) < shift_start
        after_shift = ~before_shift
        for label in [0, 1]:
            label_mask = y_online == label
            ax.scatter(
                X_online[label_mask & before_shift, 0],
                X_online[label_mask & before_shift, 1],
                color=colors[label],
                alpha=0.6,
                s=20,
                label=f"Class {label} before shift",
            )
            ax.scatter(
                X_online[label_mask & after_shift, 0],
                X_online[label_mask & after_shift, 1],
                color=colors[label],
                marker="x",
                alpha=0.8,
                s=35,
                label=f"Class {label} after shift",
            )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def plot_monitoring_results(
    X_online,
    y_online,
    monitor,
    threshold,
    title,
    shift_start=None,
):
    x_axis = np.arange(1, len(monitor.online_risk_lower_bound_sequence_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_dataset(axes[0], X_online, y_online, title, shift_start=shift_start)
    axes[0].legend()

    axes[1].plot(
        x_axis,
        monitor.online_risk_lower_bound_sequence_history,
        color="tab:purple",
        linewidth=2,
        label="Online lower confidence bound",
    )
    axes[1].axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Monitoring threshold",
    )
    if shift_start is not None:
        axes[1].axvline(
            shift_start,
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label="Shift starts",
        )
    axes[1].set_xlabel("Number of labeled online samples")
    axes[1].set_ylabel("Lower confidence bound on the misclassification risk")
    axes[1].set_title("Online monitoring")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


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
