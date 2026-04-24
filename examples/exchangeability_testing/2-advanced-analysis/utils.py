"""Plotting and reporting helpers shared by the advanced examples."""

import matplotlib.pyplot as plt


def plot_running_pvalues(tests, labels, test_level, title):
    """Plot the running p-values of several permutation tests.

    Parameters
    ----------
    tests : list
        Fitted permutation-test instances exposing a ``p_values`` attribute.
    labels : list of str
        One label per test, used in the legend.
    test_level : float
        Horizontal reference line drawn at this level.
    title : str
        Plot title.
    """
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
    """Plot jumper and plug-in martingales and the plug-in p-value histogram.

    Parameters
    ----------
    omt_jumper, omt_plugin : OnlineMartingaleTest
        Fitted online martingale test instances (jumper and plug-in).
    scenario_name : str
        Used in per-axis titles and in the figure suptitle.
    shift_start_time : int or None, default=None
        If provided, draw a vertical line at this time index to mark the
        shift start.
    """
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
    """Print compact diagnostics for each stream and martingale method.

    Parameters
    ----------
    results : dict[str, tuple]
        Mapping ``scenario_name -> (omt_jumper, omt_plugin)``.
    test_level : float, default=0.05
        Only used to annotate the header (threshold = ``1 / test_level``).
    """
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
