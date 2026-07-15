"""Private data and plotting helpers used by MAPIE examples."""

import matplotlib.pyplot as plt
import numpy as np


def plot_dataset(X_online, y_online, title, shift_start=None, ax=None):
    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(6, 4.5))

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
    ax.legend()

    if created_ax:
        plt.tight_layout()
        plt.show()


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
    plot_dataset(
        X_online,
        y_online,
        title,
        shift_start=shift_start,
        ax=axes[0],
    )

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


def generate_gaussian_stream(
    n_samples=1000,
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


def plot_running_pvalues(tests, labels, test_level, title):
    """Plot the running p-values of several permutation tests."""
    styles = [
        {"linestyle": "-", "linewidth": 2.0, "zorder": 4},
        {"linestyle": "-", "linewidth": 3.0, "zorder": 1},
        {"linestyle": "--", "linewidth": 2.0, "zorder": 2},
        {"linestyle": ":", "linewidth": 2.5, "zorder": 3},
    ]

    plt.figure(figsize=(8, 4))
    for idx, (test, label) in enumerate(zip(tests, labels)):
        style = styles[idx % len(styles)]
        plt.plot(test.p_values, label=label, **style)
    plt.axhline(
        test_level,
        color="black",
        linestyle="--",
        label=f"test_level = {test_level:.2f}",
    )
    plt.xlabel("Number of permutations")
    plt.ylabel("Running p-value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_martingale_results_one_scenario(
    omt_jumper,
    omt_plugin,
    scenario_name,
    shift_start_time=None,
):
    """Plot martingales and the plug-in p-value histogram for one scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    threshold = omt_jumper.reject_threshold

    for ax, omt, method_label in [
        (axes[0], omt_jumper, "Jumper"),
        (axes[1], omt_plugin, "Plug-in"),
    ]:
        ax.plot(omt.martingale_value_history, label=f"{method_label} martingale")
        ax.axhline(threshold, linestyle="--", color="tab:red", label="Reject threshold")
        if shift_start_time is not None:
            ax.axvline(
                shift_start_time,
                linestyle="--",
                color="black",
                label="Shift start",
            )
        summary = omt.summary()
        if summary["is_exchangeable"] is False:
            ax.axvline(
                summary["stopping_time"],
                linestyle=":",
                color="red",
                label="Stopping time",
            )
        ax.set_title(f"{scenario_name} - {method_label}", fontsize=18)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Martingale value", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_yscale("log")
        ax.legend(fontsize=14)

    ax = axes[2]
    ax.hist(omt_plugin.pvalue_history, bins=20, density=True, alpha=0.7)
    ax.axhline(1.0, linestyle="--", color="tab:gray", label="Uniform density")
    ax.set_title(f"{scenario_name} - P-values", fontsize=18)
    ax.set_xlabel("P-value", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)

    plt.suptitle(f"Online martingale tests - {scenario_name}", fontsize=22)
    plt.tight_layout()
    plt.show()


def print_martingale_summary(results, test_level=0.05):
    """Print compact diagnostics for each stream and martingale method."""
    threshold = int(round(1 / test_level))
    print(f"\nSummary at test_level = {test_level} (threshold = {threshold}):")
    print(
        "Scenario        | Method  | Decision     | Stopping time | "
        "Value at decision | Final value"
    )
    print("-" * 99)

    for scenario_name, (omt_jumper, omt_plugin) in results.items():
        for method_name, omt in [
            ("Jumper", omt_jumper),
            ("Plug-in", omt_plugin),
        ]:
            summary = omt.summary()
            decision = summary["is_exchangeable"]
            if decision is None:
                decision_str = "Inconclusive"
            elif decision:
                decision_str = "No rejection"
            else:
                decision_str = "Rejected"

            print(
                f"{scenario_name:<15} | {method_name:<7} | {decision_str:<12} | "
                f"{summary['stopping_time']:<13} | "
                f"{summary['martingale_value_at_decision']:<17.3g} | "
                f"{summary['last_martingale_value']:<.3g}"
            )
