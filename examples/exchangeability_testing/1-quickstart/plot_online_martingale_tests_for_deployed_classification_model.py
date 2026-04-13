"""
# Online Martingale Exchangeability tests for binary classification models

In this example, we show how to use `OnlineMartingaleTest` to monitor exchangeability
online after deployment of a model trained on a reference environment.
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
2. **Subtle shift**: smoother geometric drift halfway through.
3. **Abrupt shift**: sudden covariate and label shift halfway through.

For each stream, we:

1. Train a logistic regression model on a separate reference dataset,
2. Define a non-conformity score based on the model's predicted probabilities,
3. Initialize two online martingale tests (jumper and plug-in),
4. Process the stream sequentially, updating both martingales with each new observation,
5. Inspect martingale paths and p-value histograms to interpret the exchangeability decision.

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

# sphinx_gallery_thumbnail_number = 5

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie.exhangeability_testing import OnlineMartingaleTest

RANDOM_STATE = 42

warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
)


##############################################################################
# In this example, we consider that the deployed model is a logistic regression trained
# on a reference dataset where there is no reason to expect exchangeability violations.
# Therefore, we generate simple synthetic data using :func:`sklearn.datasets.make_classification`
# with two informative features and some label noise and use it as a reference environment
# for training the model. For stream monitoring, we generate three separate datasets with
# the same process but different random seeds and with different shift mechanisms to illustrate
# the behavior of online martingale tests under various scenarios.
#
# The reference data generation function is defined as follows.
#


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


##############################################################################
# We next generate a training dataset and fit a logistic regression model on it.
#

X_train, y_train = make_classification_reference_data(n_samples=2000, random_state=10)

clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)


###############################################################################
# Throughout the example, we use the same non-conformity score for both martingale tests.
# We consider:
#
# .. math::
#
#    s_i = 1 - \hat{\mathbb{P}}(Y_i = y_i \mid X_i)
#
# so that confident correct predictions have small scores and surprising
# observations have large scores.
#


def nonconformity_score(y_true, y_pred, X=None):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float)
    return 1.0 - y_pred[np.arange(len(y_true)), y_true]


###############################################################################
# Below, we plot the training data on the left and the associated non-conformity scores on the right.
#


def plot_data_and_score_histogram(
    X,
    y,
    scores,
    left_title="Training data",
    right_title="Histogram of non-conformity scores",
    figure_title="Reference training data and non-conformity scores",
):
    """Plot feature scatter (left) and class-wise score histogram (right)."""
    score_quantile = np.quantile(scores, 0.975)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    axes[0].scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        alpha=0.65,
        s=20,
        label="Class 0",
    )
    axes[0].scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        alpha=0.65,
        s=20,
        label="Class 1",
    )
    axes[0].set_title(left_title, fontsize=18)
    axes[0].set_xlabel("Feature 1", fontsize=16)
    axes[0].set_ylabel("Feature 2", fontsize=16)
    axes[0].tick_params(axis="both", labelsize=14)

    axes[1].hist(
        scores[y == 0],
        bins=25,
        alpha=0.65,
        label="Class 0",
    )
    axes[1].hist(
        scores[y == 1],
        bins=25,
        alpha=0.65,
        label="Class 1",
    )
    axes[1].axvline(
        score_quantile,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label="0.975 quantile",
    )
    axes[1].set_title(right_title, fontsize=18)
    axes[1].set_xlabel("Non-conformity score", fontsize=16)
    axes[1].set_ylabel("Count", fontsize=16)
    axes[1].tick_params(axis="both", labelsize=14)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=14)
    plt.suptitle(figure_title, fontsize=22)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.show()


y_proba_train = clf.predict_proba(X_train)
train_scores = nonconformity_score(y_train, y_proba_train)
plot_data_and_score_histogram(
    X_train,
    y_train,
    train_scores,
    left_title="Training data",
    right_title="Histogram of non-conformity scores",
    figure_title="Reference training data and non-conformity scores",
)

##############################################################################
# First, we consider the case where the model is deployed and a stream of labeled
# exchangeable observations is received sequentially.
# We want to monitor whether these observations are exchangeable with the training data.
#


def make_classification_exchangeable_stream(n_samples=600, random_state=43):
    return make_classification_reference_data(
        n_samples=n_samples,
        random_state=random_state,
    )


X_exch, y_exch = make_classification_exchangeable_stream()
test_exch_scores = nonconformity_score(y_exch, clf.predict_proba(X_exch))

plot_data_and_score_histogram(
    X_exch,
    y_exch,
    test_exch_scores,
    left_title="Exchangeable stream",
    right_title="Histogram of non-conformity scores",
    figure_title="Exchangeable stream and non-conformity scores",
)

#################################################################################
# We next initialize a `:class:~mapie.exhangeability_testing.OnlineMartingaleTest`
# for each method and process the exchangeable stream sequentially to update the martingales.
# We compare the "jumper_martingale" and "plugin_martingale" methods, passed as the ``test_method``
# argument to the constructor. These methods implement different betting strategies.
# The jumper martingale bets against small p-values with a simple,
# robust strategy that does not require density estimation,
# while the plug-in martingale estimates the p-value density and
# can react to richer departures from uniformity.
#
# We use a test level of 0.05, so the rejection threshold is ``1 / 0.05 = 20``.
# We also set ``min_sample_size_to_decide=100`` to avoid unstable early decisions.
#

test_level = 0.05
min_sample_size_to_decide = 100

omt_jumper_exch = OnlineMartingaleTest(
    non_conformity_score_function=nonconformity_score,
    test_method="jumper_martingale",
    test_level=test_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_exch = OnlineMartingaleTest(
    non_conformity_score_function=nonconformity_score,
    test_method="plugin_martingale",
    test_level=test_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)

y_proba_exch = clf.predict_proba(X_exch)
for i in range(len(y_exch)):
    omt_jumper_exch.update(y_exch[i : i + 1], y_proba_exch[i : i + 1])
    omt_plugin_exch.update(y_exch[i : i + 1], y_proba_exch[i : i + 1])

##############################################################################
# Visualize martingale trajectories and p-value distributions.
#


def plot_results_one_scenario(omt_jumper, omt_plugin, scenario_name):
    """Plot martingales and p-values for one monitoring scenario."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    threshold = omt_jumper.reject_threshold

    # Jumper martingale
    ax = axes[0]
    ax.plot(omt_jumper.martingale_value_history, label="Jumper martingale")
    ax.axhline(
        threshold,
        linestyle="--",
        color="tab:red",
        label="Reject threshold",
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
            color="black",
            label="Stopping time",
        )
    ax.legend(fontsize=14)

    # Plug-in martingale
    ax = axes[1]
    ax.plot(omt_plugin.martingale_value_history, label="Plug-in martingale")
    ax.axhline(
        threshold,
        linestyle="--",
        color="tab:red",
        label="Reject threshold",
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
            color="black",
            label="Stopping time",
        )
    ax.legend(fontsize=14)

    # P-value histogram
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


plot_results_one_scenario(omt_jumper_exch, omt_plugin_exch, "Exchangeable")

##############################################################################
# Both martingales remain stable and do not exceed the rejection threshold,
# so exchangeability is not rejected.
#

##############################################################################
# Second, we consider the case where the model is deployed and a stream of
# labeled observations is received sequentially, but there is a subtle shift
# in the data distribution halfway through the stream.
# We want to monitor whether these observations are exchangeable with the training data.
#


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


X_subtle, y_subtle = make_classification_subtle_shift_stream()
test_subtle_scores = nonconformity_score(y_subtle, clf.predict_proba(X_subtle))

plot_data_and_score_histogram(
    X_subtle,
    y_subtle,
    test_subtle_scores,
    left_title="Subtle shift stream",
    right_title="Histogram of non-conformity scores",
    figure_title="Subtle shift stream and non-conformity scores",
)

##############################################################################
# Using the same settings as before, we initialize two online martingale tests
# for the subtle-shift stream and process it sequentially to update the martingales.
#

omt_jumper_subtle_shift = OnlineMartingaleTest(
    non_conformity_score_function=nonconformity_score,
    test_method="jumper_martingale",
    test_level=test_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_subtle_shift = OnlineMartingaleTest(
    non_conformity_score_function=nonconformity_score,
    test_method="plugin_martingale",
    test_level=test_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)

y_proba_subtle = clf.predict_proba(X_subtle)

for i in range(len(y_subtle)):
    omt_jumper_subtle_shift.update(y_subtle[i : i + 1], y_proba_subtle[i : i + 1])
    omt_plugin_subtle_shift.update(y_subtle[i : i + 1], y_proba_subtle[i : i + 1])


plot_results_one_scenario(
    omt_jumper_subtle_shift,
    omt_plugin_subtle_shift,
    "Subtle shift",
)

##############################################################################
# Both martingales react to the shift and eventually exceed the rejection threshold,
# so exchangeability is rejected.
#

###############################################################################
# Third, we consider the case where the model is deployed and a stream of
# labeled observations is received sequentially, but there is an abrupt shift
# in the data distribution halfway through the stream.
# We want to monitor whether these observations are exchangeable with the training data.
#


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


X_abrupt, y_abrupt = make_classification_abrupt_shift_stream()
test_abrupt_scores = nonconformity_score(y_abrupt, clf.predict_proba(X_abrupt))

plot_data_and_score_histogram(
    X_abrupt,
    y_abrupt,
    test_abrupt_scores,
    left_title="Abrupt shift stream",
    right_title="Histogram of non-conformity scores",
    figure_title="Abrupt shift stream and non-conformity scores",
)


##############################################################################
# Using the same settings as before, we initialize two online martingale tests
# for the abrupt-shift stream and process it sequentially to update the martingales.
#

omt_jumper_abrupt_shift = OnlineMartingaleTest(
    non_conformity_score_function=nonconformity_score,
    test_method="jumper_martingale",
    test_level=test_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)
omt_plugin_abrupt_shift = OnlineMartingaleTest(
    non_conformity_score_function=nonconformity_score,
    test_method="plugin_martingale",
    test_level=test_level,
    min_sample_size_to_decide=min_sample_size_to_decide,
    random_state=RANDOM_STATE,
    warn=False,
)

y_proba_abrupt = clf.predict_proba(X_abrupt)

for i in range(len(y_abrupt)):
    omt_jumper_abrupt_shift.update(y_abrupt[i : i + 1], y_proba_abrupt[i : i + 1])
    omt_plugin_abrupt_shift.update(y_abrupt[i : i + 1], y_proba_abrupt[i : i + 1])

plot_results_one_scenario(
    omt_jumper_abrupt_shift,
    omt_plugin_abrupt_shift,
    "Abrupt shift",
)

##############################################################################
# The plug-in martingale reacts very quickly to the abrupt shift,
# while the jumper martingale does not react and does not reject exchangeability,
# which illustrates the different betting strategies of the two methods.
# In fact, the jumper bets against small p-values and in this scenario, the p-values
# histogram shows that there are many small p-values but also many large p-values,
# which does not create a clear signal for the jumper strategy.
# However, the plug-in martingale estimates the p-value density and can react to
# this departure from uniformity, which is why it reacts quickly and rejects exchangeability.
#

##############################################################################
# Finally, we collect results and print summary.
#

classification_results = {
    "Exchangeable": (omt_jumper_exch, omt_plugin_exch),
    "Abrupt shift": (omt_jumper_abrupt_shift, omt_plugin_abrupt_shift),
    "Subtle shift": (omt_jumper_subtle_shift, omt_plugin_subtle_shift),
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


print_result_summary(classification_results)


##############################################################################
# In this example, we illustrated how to use online martingale tests to monitor exchangeability
# for a binary classification model deployed in a monitoring environment where a stream of labeled
# data is continuously received. The online martingale tests allow us to detect shifts in the
# data distribution and assess whether the model's predictions remain reliable over time.
#
# The results show that both the jumper and plug-in martingales correctly do not reject exchangeability
# for the exchangeable stream, while they both reject exchangeability for the subtle shift stream.
# However, for the abrupt shift stream, only the plug-in martingale reacts and rejects exchangeability,
# while the jumper martingale does not react and does not reject exchangeability, which illustrates the
# different betting strategies of the two methods.
#
# Finally, the model and the choice of non-conformity score are important for the performance of the tests,
# as they determine the p-values and the martingale updates. In practice, it is recommended to use
# a well-performing model and a non-conformity score that captures the model's confidence in its predictions.
#
