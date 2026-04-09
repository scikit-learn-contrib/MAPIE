"""
========================================================================
Online Martingale Exchangeability tests for binary classification models
========================================================================

In this example, we show how to use
:class:`~mapie.exhangeability_testing.OnlineMartingaleTest` to monitor exchangeability
on line after deployment of a model trained on a reference environment.
We illustrate the workflow with a binary classification task,
but the same principles apply to regression and other settings.

The tests consider the practical scenario where a model is trained on a reference environment
where there is no reason to expect exchangeability violations, then deployed in a monitoring environment where
a stream of labeled observations is received and the goal is to check if they are exchangeable with the training data.

Online martingale tests are a powerful tool for this problem, as they provide a lightweight,
model-agnostic way to monitor exchangeability over time.
They work by converting each new observation into a conformal p-value based on a non-conformity score,
then accumulating evidence against exchangeability with a martingale that bets against small p-values.
When the martingale value exceeds a threshold, exchangeability is rejected
with a user-chosen confidence level. See [1]_, [2]_, and [3]_ for theoretical
details and guarantees.


In the following, we implement a complete workflow for online exchangeability testing for stream data, including:

1. **Exchangeable**: same environment as the training data.
2. **Abrupt shift**: sudden covariate and label shift halfway through.
3. **Subtle shift**: smoother geometric drift halfway through.

For each stream, we:

1. We train a logistic regression model on a separate reference dataset,
2. Define a non-conformity score based on the model's predicted probabilities,
3. Initialize two online martingale tests (jumper and plug-in),
4. Process the stream sequentially, updating both martingales with each new observation,
5. Inspect martingale paths and p-value histograms to interpret the exchangeability decision.

References
----------
.. [1] Angelopoulos, Barber, Bates (2026).
    "Theoretical Foundations of Conformal Prediction".
    arXiv preprint arXiv:2411.11824.
.. [2] Vovk, Gammerman, Shafer (2005).
    "Algorithmic Learning in a Random World".
    Boston, MA: Springer US. Section 7.1, page 169.
.. [3] Fedorova, Gammerman, Nouretdinov, Vovk (2012).
    "Plug-in Martingales for Testing Exchangeability on-line".
    In Proceedings of the 29th ICML. Algorithm 1, page 3.
"""
# %%
# sphinx_gallery_thumbnail_number = 2

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mapie.exhangeability_testing import OnlineMartingaleTest

RANDOM_STATE = 42

warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
)

##############################################################################
# Step 1: Define the non-conformity score.
#
# We use:
#
# .. math::
#
#    s_i = 1 - \hat{P}(Y_i = y_i \mid X_i)
#
# so that confident correct predictions have small scores and surprising
# observations have large scores.
#
# These scores are then converted into conformal p-values using only the
# previously observed scores.


def classification_nonconformity_score(y_true, y_pred, X=None):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float)
    return 1.0 - y_pred[np.arange(len(y_true)), y_true]


##############################################################################
# Step 2: Generate reference and monitoring streams.
#
# We use one training distribution, then construct three
# monitoring streams:
#
# - one exchangeable stream,
# - one with an abrupt shift,
# - one with a subtler shift.


def make_classification_reference_data(n_samples=1500, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        flip_y=0.05,
        random_state=random_state,
    )
    return X, y


def make_classification_exchangeable_stream(n_samples=600, random_state=43):
    return make_classification_reference_data(
        n_samples=n_samples,
        random_state=random_state,
    )


def make_classification_abrupt_shift_stream(n_samples=600, random_state=44):
    X, y = make_classification_reference_data(
        n_samples=n_samples,
        random_state=random_state,
    )
    midpoint = n_samples // 2

    # Abrupt shift: covariates move and labels are partially flipped.
    # This creates a clear exchangeability break for the monitoring stage.
    X[midpoint:, 0] += 3.0
    X[midpoint:, 1] -= 2.0
    flip_mask = np.random.default_rng(random_state).random(n_samples - midpoint) < 0.35
    y[midpoint:][flip_mask] = 1 - y[midpoint:][flip_mask]
    return X, y


def make_classification_subtle_shift_stream(n_samples=600, random_state=45):
    X, y = make_classification_reference_data(
        n_samples=n_samples,
        random_state=random_state,
    )
    midpoint = n_samples // 2

    # Subtle shift: rotation + anisotropic scaling in the second half
    theta = np.deg2rad(25)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    X[midpoint:] = X[midpoint:] @ rotation.T
    X[midpoint:, 0] *= 1.8
    X[midpoint:, 1] *= 0.7
    X[midpoint:] += np.array([0.5, -0.2])

    return X, y


##############################################################################
# Visual check of the three streams.


X_exch, y_exch = make_classification_exchangeable_stream()
X_abrupt, y_abrupt = make_classification_abrupt_shift_stream()
X_subtle, y_subtle = make_classification_subtle_shift_stream()

streams = [
    (X_exch, y_exch, "Exchangeable"),
    (X_abrupt, y_abrupt, "Abrupt shift"),
    (X_subtle, y_subtle, "Subtle shift"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

for ax, (X, y, title) in zip(axes, streams):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.65, s=20, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.65, s=20, label="Class 1")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Feature 1", fontsize=11)
    ax.set_ylabel("Feature 2", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11)
plt.suptitle("Monitoring streams", fontsize=16)
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.show()

##############################################################################
# Step 3: Fit a probabilistic model on training data only.


X_train, y_train = make_classification_reference_data(n_samples=2000, random_state=10)

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
)
clf.fit(X_train, y_train)

##############################################################################
# Step 4: Initialize online martingale tests.
#
# We compare ``"jumper_martingale"`` and ``"plugin_martingale"``.
#
# We use confidence level 0.95 (test level 0.05), so the rejection threshold
# is ``1 / 0.05 = 20``. We also set ``min_sample_size_to_decide=100`` to avoid
# unstable early decisions.
#

confidence_level = 0.95
min_sample_size_to_decide = 100

omt_jumper_1 = OnlineMartingaleTest(
    non_conformity_score_function=classification_nonconformity_score,
    test_method="jumper_martingale",
    confidence_level=confidence_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_1 = OnlineMartingaleTest(
    non_conformity_score_function=classification_nonconformity_score,
    test_method="plugin_martingale",
    confidence_level=confidence_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_jumper_2 = OnlineMartingaleTest(
    non_conformity_score_function=classification_nonconformity_score,
    test_method="jumper_martingale",
    confidence_level=confidence_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_2 = OnlineMartingaleTest(
    non_conformity_score_function=classification_nonconformity_score,
    test_method="plugin_martingale",
    confidence_level=confidence_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_jumper_3 = OnlineMartingaleTest(
    non_conformity_score_function=classification_nonconformity_score,
    test_method="jumper_martingale",
    confidence_level=confidence_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_3 = OnlineMartingaleTest(
    non_conformity_score_function=classification_nonconformity_score,
    test_method="plugin_martingale",
    confidence_level=confidence_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)

##############################################################################
# Step 5: Process each stream sequentially and update both martingales.

y_proba_exch = clf.predict_proba(X_exch)
y_proba_abrupt = clf.predict_proba(X_abrupt)
y_proba_subtle = clf.predict_proba(X_subtle)

for i in range(len(y_exch)):
    omt_jumper_1.update(y_exch[i : i + 1], y_proba_exch[i : i + 1])
    omt_plugin_1.update(y_exch[i : i + 1], y_proba_exch[i : i + 1])

for i in range(len(y_abrupt)):
    omt_jumper_2.update(y_abrupt[i : i + 1], y_proba_abrupt[i : i + 1])
    omt_plugin_2.update(y_abrupt[i : i + 1], y_proba_abrupt[i : i + 1])

for i in range(len(y_subtle)):
    omt_jumper_3.update(y_subtle[i : i + 1], y_proba_subtle[i : i + 1])
    omt_plugin_3.update(y_subtle[i : i + 1], y_proba_subtle[i : i + 1])


##############################################################################
# Collect results for visualization and summary.


classification_results = {
    "Exchangeable": (omt_jumper_1, omt_plugin_1),
    "Abrupt shift": (omt_jumper_2, omt_plugin_2),
    "Subtle shift": (omt_jumper_3, omt_plugin_3),
}


def print_result_summary(results):
    """Print compact diagnostics for each stream and method."""
    print("\nSummary at confidence_level = 0.95 (threshold = 20):")
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


print_result_summary(classification_results)

##############################################################################
# Step 6: Visualize martingale trajectories and p-value distributions.
#
# Under exchangeability, p-values should be close to uniform. Persistent
# departures from uniformity accumulate as evidence against exchangeability.


def plot_results_grid(results, title):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14.5))
    threshold = next(iter(results.values()))[0].reject_threshold

    for row, (scenario_name, (omt_jumper, omt_plugin)) in enumerate(results.items()):
        # Jumper martingale
        ax = axes[row, 0]
        ax.plot(omt_jumper.martingale_value_history, label="Jumper martingale")
        ax.axhline(
            threshold,
            linestyle="--",
            color="tab:red",
            label="Reject threshold",
        )
        ax.set_title(f"{scenario_name} - Jumper")
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Martingale value", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_yscale("log")
        summary_jumper = omt_jumper.summary()
        if summary_jumper["is_exchangeable"] is False:
            ax.axvline(
                summary_jumper["stopping_time"],
                linestyle=":",
                color="black",
                label="Stopping time",
            )
        if row == 0:
            ax.legend()

        # Plug-in martingale
        ax = axes[row, 1]
        ax.plot(omt_plugin.martingale_value_history, label="Plug-in martingale")
        ax.axhline(
            threshold,
            linestyle="--",
            color="tab:red",
            label="Reject threshold",
        )
        ax.set_title(f"{scenario_name} - Plug-in")
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Martingale value", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_yscale("log")
        summary_plugin = omt_plugin.summary()
        if summary_plugin["is_exchangeable"] is False:
            ax.axvline(
                summary_plugin["stopping_time"],
                linestyle=":",
                color="black",
                label="Stopping time",
            )
        if row == 0:
            ax.legend()

        # P-value histogram
        ax = axes[row, 2]
        ax.hist(omt_plugin.pvalue_history, bins=20, density=True, alpha=0.7)
        ax.axhline(1.0, linestyle="--", color="tab:gray", label="Uniform density")
        ax.set_title(f"{scenario_name} - P-values")
        ax.set_xlabel("P-value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        if row == 0:
            ax.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


plot_results_grid(
    classification_results,
    "Online martingale tests for binary classification",
)

##############################################################################
# Interpretation of this run.
#
# In the exchangeable setting, p-values are approximately uniform and both
# martingales stay below threshold in this run.
#
# In the abrupt-shift stream, the plug-in martingale rejects while the jumper
# martingale does not. This illustrates that different betting strategies can
# have different sensitivity to the same departure from exchangeability.
#
# In this run, abrupt-shift p-values are roughly bimodal (mass near 0 and 1).
# The jumper martingale mainly bets against small p-values, so evidence carried
# by large p-values is not fully exploited. The plug-in martingale can react to
# both tails through density estimation, which explains why it rejects here.
#
# In the subtle-shift scenario, both martingales cross the rejection threshold,
# with different stopping times. The difference comes from how each method
# aggregates evidence from the p-value sequence.

##############################################################################
# Takeaway.
#
# Online martingale tests provide lightweight, model-agnostic monitoring of
# exchangeability in binary classification.
#
# In practice, the workflow is:
#
# 1. fit a predictive model on a training environment,
# 2. define a suitable non-conformity score,
# 3. convert future labeled observations into conformal p-values,
# 4. accumulate evidence against exchangeability with a martingale.
#
# The jumper martingale is often a strong default due to its robustness and
# simplicity, while the plug-in martingale can be more sensitive to richer
# departures from uniformity in the p-value distribution.
#
# Online martingale tests are particularly useful when monitoring whether
# predictive uncertainty assumptions remain valid after deployment.
#
# Practical interpretation tip: when no rejection is observed, this means
# "no violation detected so far at the chosen level and horizon", not proof
# that exchangeability truly holds.
#
# Final takeaway: these outcomes depend on several design choices, including
# the predictive model, the non-conformity score, the stream shift mechanism,
# and martingale settings (method, confidence level, warm-up size).

# %%
