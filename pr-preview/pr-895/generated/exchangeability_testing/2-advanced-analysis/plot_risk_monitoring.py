"""
# Detect harmful shifts with RiskMonitoring

In this example, we show how to use `RiskMonitoring` to track the
misclassification risk of a deployed binary classifier on an online stream.
The monitoring threshold is first estimated on a reference test set, then
the online lower confidence bound is updated as new labeled data arrives.

Instead of directly comparing risk estimations, computing confidence bounds
gives statistical guarantees to take evaluation uncertainty into account [1].

A limitation of the current implememted approach is that (at least some)
labeled data is necessary to update online risk estimation. For cases with
scarce or no labels, please refer to these extensions, [2] and [3] respectively.

### References:
- [1] Aleksandr Podkopaev and Aaditya Ramdas. Tracking the risk of a deployed
model and detecting harmful distribution shifts.
International Conference on Learning Representations, 2022.
- [2] Zhang, Guangyi, Cai, Yunlong, Yu, Guanding, et al. Prediction-Powered
Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts.
arXiv preprint arXiv:2602.02229, 2026.
- [3] Amoukou, Salim I., Bewley, Tom, Mishra, Saumitra, et al. Sequential
harmful shift detection without labels.
Advances in Neural Information Processing Systems, 2024.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from mapie.exchangeability_testing import RiskMonitoring
from utils import generate_gaussian_stream, sample_two_gaussians


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


##############################################################################
# We first fit a classifier on clean training data. Then, in the same workflow,
# we estimate the monitoring threshold on a clean reference set and update the
# monitor on a stable online stream. Here, `risk="accuracy"` means that
# `RiskMonitoring` tracks the misclassification risk `1 - accuracy`.

random_state = 42
batch_size = 25
prop_shift = 0.5

X_train, y_train = sample_two_gaussians(random_state=random_state)
X_reference, y_reference = sample_two_gaussians(random_state=random_state + 1)

clf = LogisticRegression(random_state=random_state)
clf.fit(X_train, y_train)

X_online_no_shift, y_online_no_shift = generate_gaussian_stream(
    shift_type="stable",
    prop_shift=prop_shift,
    random_state=random_state + 2,
)

monitor_no_shift = RiskMonitoring(risk="accuracy")
monitor_no_shift.compute_threshold(y_reference, clf.predict(X_reference))
threshold = monitor_no_shift.threshold

print(
    f"Reference upper bound on the misclassification risk: {monitor_no_shift.reference_risk_upper_bound:.3f}"
)
print(f"Monitoring threshold: {threshold:.3f}")

for start in range(0, len(X_online_no_shift), batch_size):
    stop = start + batch_size
    y_pred_batch = clf.predict(X_online_no_shift[start:stop])
    monitor_no_shift.update_online_risk(y_online_no_shift[start:stop], y_pred_batch)

print("\nNo shift scenario")
print(f"Harmful shift detected: {monitor_no_shift.harmful_shift_detected}")

plot_monitoring_results(
    X_online_no_shift,
    y_online_no_shift,
    monitor_no_shift,
    threshold,
    title="Stable online stream",
)

##############################################################################
# We now create an abrupt shift in the middle of the stream. The data
# distribution changes suddenly, and the monitored lower confidence bound
# eventually crosses the threshold. Note that there is no need to call
# `compute_threshold`: instead we can reuse the value computed earlier
# when initalizing `RiskMonitoring`. This also highlights that you can also
# define a custom threshold for the monitoring.

X_online_abrupt, y_online_abrupt = generate_gaussian_stream(
    shift_type="abrupt",
    prop_shift=prop_shift,
    random_state=random_state + 3,
)
shift_start_abrupt = int(len(y_online_abrupt) * (1 - prop_shift))

monitor_abrupt = RiskMonitoring(risk="accuracy", threshold=threshold)
for start in range(0, len(X_online_abrupt), batch_size):
    stop = start + batch_size
    y_pred_batch = clf.predict(X_online_abrupt[start:stop])
    monitor_abrupt.update_online_risk(y_online_abrupt[start:stop], y_pred_batch)

print("\nAbrupt shift scenario")
print(f"Harmful shift detected: {monitor_abrupt.harmful_shift_detected}")

plot_monitoring_results(
    X_online_abrupt,
    y_online_abrupt,
    monitor_abrupt,
    threshold,
    title="Abrupt shift stream",
    shift_start=shift_start_abrupt,
)

##############################################################################
# Finally, we create a slow drift. In that case the distribution evolves
# progressively, so the harmful shift is typically detected later than with the
# abrupt change.

X_online_slow, y_online_slow = generate_gaussian_stream(
    shift_type="slow",
    prop_shift=prop_shift,
    random_state=random_state + 4,
)
shift_start_slow = int(len(y_online_slow) * (1 - prop_shift))

monitor_slow = RiskMonitoring(risk="accuracy", threshold=threshold)
for start in range(0, len(X_online_slow), batch_size):
    stop = start + batch_size
    y_pred_batch = clf.predict(X_online_slow[start:stop])
    monitor_slow.update_online_risk(y_online_slow[start:stop], y_pred_batch)

print("\nSlow shift scenario")
print(f"Harmful shift detected: {monitor_slow.harmful_shift_detected}")

plot_monitoring_results(
    X_online_slow,
    y_online_slow,
    monitor_slow,
    threshold,
    title="Slow shift stream",
    shift_start=shift_start_slow,
)

##############################################################################
# This example highlights the intended workflow:
#
# 1. estimate an acceptable risk level on clean reference data,
# 2. update the online risk as new labeled observations become available,
# 3. flag a harmful shift once the lower confidence bound exceeds the threshold.
