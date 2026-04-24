"""
# Online martingale exchangeability tests for a deployed regressor

When a predictive model is deployed in production, a common question arises:
*are newly arriving labeled observations still exchangeable with the data used
for model development?*

This example answers that question with
:class:`~mapie.exchangeability_testing.OnlineMartingaleTest`,
a lightweight sequential test that converts each new observation into a
conformal p-value and accumulates evidence against exchangeability through a
martingale process.
When the martingale exceeds ``1 / test_level``, exchangeability is rejected
with false-alarm probability controlled by ``test_level``.
See [1], [2], and [3] for details and guarantees.

**MAPIE-style workflow.**
Following standard MAPIE practice, we generate one dataset and split it into:

1. **Train set** (30 %): fit the predictive model.
2. **Conformalization set** (20 %): held-out data used to calibrate conformity scores.
3. **Test set** (50 %): future monitoring data, never seen during training.

The monitoring stream given to
:class:`~mapie.exchangeability_testing.OnlineMartingaleTest`
is the concatenation of conformalization and test partitions.
This reflects the practical recommendation:
*run exchangeability diagnostics on held-out data only.*

**Three monitoring scenarios.**
We compare three test-set variants:

1. **Exchangeable**: same distribution as training.
2. **Subtle shift**: mild noisy location shift in the second half of test targets.
3. **Abrupt shift**: larger noisy location shift in the second half of test targets.

**Two martingale strategies.**
For each scenario we run:

- ``"jumper_martingale"``: bets against an excess of small p-values.
- ``"plugin_martingale"``: estimates p-value density and can react to broader
  departures from uniformity.

References
----------
 - [1] Angelopoulos, Barber, Bates (2026).
     "Theoretical Foundations of Conformal Prediction".
     arXiv preprint arXiv:2411.11824.
 - [2] Vovk, Gammerman, Shafer (2005).
     "Algorithmic Learning in a Random World".
     Boston, MA: Springer US. Section 7.1, page 169.
 - [3] Fedorova, Gammerman, Nouretdinov, Vovk (2012).
     "Plug-in Martingales for Testing Exchangeability on-line".
     In Proceedings of the 29th ICML. Algorithm 1, page 3.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from mapie.exchangeability_testing import OnlineMartingaleTest
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split

RANDOM_STATE = 7

warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
)

##############################################################################
# Data preparation
# ----------------
#
# We generate one simple linear regression dataset and apply the standard
# MAPIE train, conformalize, and test split. The linear regressor is fitted on
# the train partition only, then wrapped in
# :class:`~mapie.regression.SplitConformalRegressor` in prefit mode.
#

rng = np.random.default_rng(RANDOM_STATE)
X_full = np.linspace(0.1, 0.9, 2400).reshape(-1, 1)
y_full = 3.0 * X_full.ravel() + rng.normal(scale=0.1, size=X_full.shape[0])

(
    X_train,
    X_conformalize,
    X_test,
    y_train,
    y_conformalize,
    y_test,
) = train_conformalize_test_split(
    X_full,
    y_full,
    train_size=0.3,
    conformalize_size=0.2,
    test_size=0.5,
    shuffle=False,
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

mapie_regressor = SplitConformalRegressor(
    estimator=regressor,
    prefit=True,
)

##############################################################################
# We build three monitoring streams. The first half of each stream is the
# unchanged conformalization set. The second half (test set) is either left
# unchanged (exchangeable), mildly shifted with noise (subtle), or strongly
# shifted with noise (abrupt).
#

X_test_exch, y_test_exch = X_test.copy(), y_test.copy()
X_test_subtle, y_test_subtle = X_test.copy(), y_test.copy()
X_test_abrupt, y_test_abrupt = X_test.copy(), y_test.copy()

midpoint = len(y_test) // 2

# Subtle shift: mild noisy location shift in second half
y_test_subtle[midpoint:] += rng.normal(loc=0.1, scale=0.2, size=len(y_test) - midpoint)

# Abrupt shift: stronger noisy location shift in second half
y_test_abrupt[midpoint:] += rng.normal(loc=0.4, scale=0.2, size=len(y_test) - midpoint)

# Each monitoring stream = conformalize partition (clean) + test partition (scenario-specific)
X_exch = np.vstack([X_conformalize, X_test_exch])
y_exch = np.concatenate([y_conformalize, y_test_exch])
X_subtle = np.vstack([X_conformalize, X_test_subtle])
y_subtle = np.concatenate([y_conformalize, y_test_subtle])
X_abrupt = np.vstack([X_conformalize, X_test_abrupt])
y_abrupt = np.concatenate([y_conformalize, y_test_abrupt])

##############################################################################
# The figure below shows (Feature 1, target) for each test scenario before
# running online martingale monitoring.
# Marker style encodes temporal segment for shifted scenarios
# (dot = before shift, cross = after shift).
#

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)
for ax, title, X_data, y_data in zip(
    axes,
    ["Exchangeable test", "Subtle shift test", "Abrupt shift test"],
    [X_test_exch, X_test_subtle, X_test_abrupt],
    [y_test_exch, y_test_subtle, y_test_abrupt],
):
    if title == "Exchangeable test":
        ax.scatter(
            X_data[:, 0], y_data, s=16, alpha=0.65, marker="o", label="Observations"
        )
    else:
        before_mask = np.arange(len(y_data)) < midpoint
        after_mask = ~before_mask
        ax.scatter(
            X_data[before_mask, 0],
            y_data[before_mask],
            s=16,
            alpha=0.65,
            marker="o",
            label="Before shift",
        )
        ax.scatter(
            X_data[after_mask, 0],
            y_data[after_mask],
            s=26,
            alpha=0.8,
            marker="x",
            label="After shift",
        )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Feature 1", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
axes[0].set_ylabel("Target", fontsize=16)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=14)
plt.suptitle("Held-out test scenarios for exchangeability monitoring", fontsize=22)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()

##############################################################################
# We define one helper to visualize martingale trajectories and the plug-in
# p-value histogram for each scenario.
#

test_level = 0.05
burn_in = 100
shift_start_time = len(y_conformalize) + midpoint


def plot_results_one_scenario(
    omt_jumper,
    omt_plugin,
    scenario_name,
    shift_start_time=None,
):
    """Plot martingales and p-value histogram for one monitoring scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    threshold = omt_jumper.reject_threshold

    # Jumper martingale
    ax = axes[0]
    ax.plot(omt_jumper.martingale_value_history, label="Jumper martingale")
    ax.axhline(threshold, linestyle="--", color="tab:red", label="Reject threshold")
    if shift_start_time is not None:
        ax.axvline(
            shift_start_time,
            linestyle="--",
            color="black",
            label="Shift start",
        )
    ax.set_title(f"{scenario_name} - Jumper", fontsize=18)
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Martingale value", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_yscale("log")
    summary_jumper = omt_jumper.summary()
    if summary_jumper["is_exchangeable"] is False:
        ax.axvline(
            summary_jumper["stopping_time"],
            linestyle=":",
            color="red",
            label="Stopping time",
        )
    ax.legend(fontsize=14)

    # Plug-in martingale
    ax = axes[1]
    ax.plot(omt_plugin.martingale_value_history, label="Plug-in martingale")
    ax.axhline(threshold, linestyle="--", color="tab:red", label="Reject threshold")
    if shift_start_time is not None:
        ax.axvline(
            shift_start_time,
            linestyle="--",
            color="black",
            label="Shift start",
        )
    ax.set_title(f"{scenario_name} - Plug-in", fontsize=18)
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("Martingale value", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_yscale("log")
    summary_plugin = omt_plugin.summary()
    if summary_plugin["is_exchangeable"] is False:
        ax.axvline(
            summary_plugin["stopping_time"],
            linestyle=":",
            color="red",
            label="Stopping time",
        )
    ax.legend(fontsize=14)

    # P-value histogram (plug-in)
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


##############################################################################
# Scenario 1 - Exchangeable test set
# ----------------------------------
#
# The test set follows the same distribution as training data.
# Both martingales should stay below the rejection threshold.
#

omt_jumper_exch = OnlineMartingaleTest(
    mapie_estimator=mapie_regressor,
    task="regression",
    test_method="jumper_martingale",
    test_level=test_level,
    burn_in=burn_in,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_exch = OnlineMartingaleTest(
    mapie_estimator=mapie_regressor,
    task="regression",
    test_method="plugin_martingale",
    test_level=test_level,
    burn_in=burn_in,
    random_state=RANDOM_STATE,
    warn=False,
)

omt_jumper_exch.update(X_exch, y_exch)
omt_plugin_exch.update(X_exch, y_exch)

plot_results_one_scenario(
    omt_jumper_exch,
    omt_plugin_exch,
    "Exchangeable",
    shift_start_time=None,
)

##############################################################################
# Neither method should reject in this reference scenario.
#

##############################################################################
# Scenario 2 - Subtle shift
# -------------------------
#
# The second half of the test set receives a mild noisy target location shift.
#

omt_jumper_subtle_shift = OnlineMartingaleTest(
    mapie_estimator=mapie_regressor,
    task="regression",
    test_method="jumper_martingale",
    test_level=test_level,
    burn_in=burn_in,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_subtle_shift = OnlineMartingaleTest(
    mapie_estimator=mapie_regressor,
    task="regression",
    test_method="plugin_martingale",
    test_level=test_level,
    burn_in=burn_in,
    random_state=RANDOM_STATE,
    warn=False,
)

omt_jumper_subtle_shift.update(X_subtle, y_subtle)
omt_plugin_subtle_shift.update(X_subtle, y_subtle)

plot_results_one_scenario(
    omt_jumper_subtle_shift,
    omt_plugin_subtle_shift,
    "Subtle shift",
    shift_start_time=shift_start_time,
)

##############################################################################
# For this synthetic stream, both martingales are expected to accumulate
# evidence and can eventually reject exchangeability.
#

##############################################################################
# Scenario 3 - Abrupt shift
# -------------------------
#
# The second half of the test set receives a stronger noisy target location
# shift, creating a more obvious exchangeability violation.
#

omt_jumper_abrupt_shift = OnlineMartingaleTest(
    mapie_estimator=mapie_regressor,
    task="regression",
    test_method="jumper_martingale",
    test_level=test_level,
    burn_in=burn_in,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_abrupt_shift = OnlineMartingaleTest(
    mapie_estimator=mapie_regressor,
    task="regression",
    test_method="plugin_martingale",
    test_level=test_level,
    burn_in=burn_in,
    random_state=RANDOM_STATE,
    warn=True,
)

omt_jumper_abrupt_shift.update(X_abrupt, y_abrupt)
with warnings.catch_warnings(record=True) as raised_warnings:
    warnings.simplefilter("always")
    omt_plugin_abrupt_shift.update(X_abrupt, y_abrupt)
if raised_warnings:
    print(f"Raised warning: {raised_warnings[0].message}")

plot_results_one_scenario(
    omt_jumper_abrupt_shift,
    omt_plugin_abrupt_shift,
    "Abrupt shift",
    shift_start_time=shift_start_time,
)

##############################################################################
# Both methods should reject quickly in this stronger-shift regime.
#

##############################################################################
# Summary
# -------
#
# We collect and print diagnostics for each scenario and martingale method.
#

regression_results = {
    "Exchangeable": (omt_jumper_exch, omt_plugin_exch),
    "Subtle shift": (omt_jumper_subtle_shift, omt_plugin_subtle_shift),
    "Abrupt shift": (omt_jumper_abrupt_shift, omt_plugin_abrupt_shift),
}


def print_result_summary(results):
    """Print compact diagnostics for each stream and method."""
    print("\nSummary at test_level = 0.05 (threshold = 20):")
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


print_result_summary(regression_results)

##############################################################################
# The key takeaway is that online martingale tests integrate naturally into a
# standard MAPIE regression workflow and provide sequential exchangeability
# monitoring on held-out data without changing the predictive model itself.
#

# %%
