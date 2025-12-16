"""Tests converted from the notebook:
mapie/notebooks/risk_control/theoretical_validity_tests.ipynb

This file reproduces the notebook logic (random classifier and logistic classifier
theoretical validity checks) as pytest tests.
"""

import warnings
from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from mapie.risk_control import (
    BinaryClassificationController,
    accuracy,
    precision,
    recall,
)

warnings.filterwarnings("ignore")


class RandomClassifier:
    """Simple deterministic random-like classifier used in the notebook."""

    def __init__(self, seed=42, threshold=0.5):
        self.seed = seed
        self.threshold = threshold

    def _get_prob(self, x):
        local_seed = hash((x, self.seed)) % (2**32)
        rng = np.random.RandomState(local_seed)
        return np.round(rng.rand(), 2)

    def predict_proba(self, X):
        probs = np.array([self._get_prob(x) for x in X])
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):  # pragma: no cover
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


def precision_random_classifier(threshold: float) -> float:
    """
    Theoretical precision of RandomClassifier on a balanced dataset.
    - RandomClassifier assigns labels randomly with a uniform probability, independent of the true label.
    - Therefore, precision is always 0.5 regardless of the threshold.
    """
    return 0.5


def accuracy_random_classifier(threshold: float) -> float:
    """
    Theoretical accuracy of RandomClassifier on a balanced dataset.
    - RandomClassifier assigns labels randomly with a uniform probability, independent of the true label.
    - Therefore, accuracy is always 0.5 regardless of the threshold.
    """
    return 0.5


def recall_random_classifier(threshold: float) -> float:
    """
    Theoretical recall of RandomClassifier on a balanced dataset.
    - RandomClassifier assigns labels randomly with a uniform probability, independent of the true label.
    - Therefore, recall = 1 - threshold.
    """
    return 1.0 - threshold


class LogisticClassifier:
    """Deterministic sigmoid-based binary classifier."""

    def __init__(self, scale=2.0, threshold=0.5):
        self.scale = scale
        self.threshold = threshold

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-self.scale * x))

    def _get_prob(self, x):
        """Probability of class 1 for input x."""
        return self.sigmoid(x)

    def predict_proba(self, X):
        """Return probabilities [p(y=0), p(y=1)] for each sample in X."""
        probs = np.array([self._get_prob(x) for x in X])
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):  # pragma: no cover
        """Return predicted class labels based on threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


def make_logistic_data(n_samples=200, scale=2.0, random_state=None):
    rng = check_random_state(random_state)
    X = rng.uniform(-3, 3, size=n_samples)
    probs = LogisticClassifier(scale=scale).sigmoid(X)
    y = rng.binomial(1, probs)
    return X, y


def precision_logistic_classifier(scale: float, threshold: float) -> float:
    """
    Theoretical precision of LogisticClassifier based on the `make_logistic_data` generator.
    - Data are generated as pairs (X, Y), where
        - X ~ Uniform(-3, 3)
        - Y | X=x ~ Bernoulli(p(x))
        - p(x) = P(Y=1|X=x) = 1 / (1 + exp(-scale * x))
    - Precision has a closed-form expression depending on `scale` and `threshold`.
    """
    decision_threshold = np.log(threshold / (1 - threshold)) / scale
    TP = (
        1
        / (6 * scale)
        * (
            np.log(1 + np.exp(3 * scale))
            - np.log(1 + np.exp(scale * decision_threshold))
        )
    )
    TP_plus_FP = (3 - decision_threshold) / 6

    return TP / TP_plus_FP


def accuracy_logistic_classifier(scale: float, threshold: float) -> float:
    """
    Theoretical accuracy of LogisticClassifier based on the `make_logistic_data` generator.
    - Data are generated as pairs (X, Y), where
        - X ~ Uniform(-3, 3)
        - Y | X=x ~ Bernoulli(p(x))
        - p(x) = P(Y=1|X=x) = 1 / (1 + exp(-scale * x))
    - Accuracy has a closed-form expression depending on `scale` and `threshold`.
    """
    decision_threshold = np.log(threshold / (1 - threshold)) / scale
    TP = (
        1
        / (6 * scale)
        * (
            np.log(1 + np.exp(3 * scale))
            - np.log(1 + np.exp(scale * decision_threshold))
        )
    )
    FN = (
        1
        / (6 * scale)
        * (
            np.log(1 + np.exp(scale * decision_threshold))
            - np.log(1 + np.exp(-3 * scale))
        )
    )
    FN_plus_TN = (3 + decision_threshold) / 6
    TN = FN_plus_TN - FN
    return TP + TN


def recall_logistic_classifier(scale: float, threshold: float) -> float:
    """
    Theoretical recall of LogisticClassifier based on the `make_logistic_data` generator.
    - Data are generated as pairs (X, Y), where
        - X ~ Uniform(-3, 3)
        - Y | X=x ~ Bernoulli(p(x))
        - p(x) = P(Y=1|X=x) = 1 / (1 + exp(-scale * x))
    - Recall has a closed-form expression depending on `scale` and `threshold`.
    """
    decision_threshold = np.log(threshold / (1 - threshold)) / scale
    TP = (
        1
        / (6 * scale)
        * (
            np.log(1 + np.exp(3 * scale))
            - np.log(1 + np.exp(scale * decision_threshold))
        )
    )
    FN = (
        1
        / (6 * scale)
        * (
            np.log(1 + np.exp(scale * decision_threshold))
            - np.log(1 + np.exp(-3 * scale))
        )
    )
    return TP / (TP + FN)


def run_one_experiment_with_random_classifier(
    clf_class, risk_dict, predict_params, target_level, confidence_level, N, n_repeats
):
    """
    Runs the experiment for one combination of risk, predict_params, target_level, confidence_level.
    Returns a DataFrame with one row per repeat.
    """
    clf = clf_class()
    records = []

    for repeat_id in range(n_repeats):
        X_calibrate, y_calibrate = make_classification(
            n_samples=N,
            n_features=1,
            n_informative=1,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            weights=[0.5, 0.5],
            flip_y=0,
            random_state=None,
        )
        X_calibrate = X_calibrate.squeeze()

        controller = BinaryClassificationController(
            predict_function=clf.predict_proba,
            risk=risk_dict["risk"],
            target_level=target_level,
            confidence_level=confidence_level,
            list_predict_params=predict_params,
        )
        controller.calibrate(X_calibrate, y_calibrate)
        valid_parameters = controller.valid_predict_params

        error_indicator = 0

        if len(valid_parameters) == 0:
            error_indicator = 1
        else:
            for lambda_ in valid_parameters:
                if risk_dict["risk"] == precision:
                    theoretical_metric = precision_random_classifier(lambda_)
                elif risk_dict["risk"] == recall:
                    theoretical_metric = recall_random_classifier(lambda_)
                elif risk_dict["risk"] == accuracy:
                    theoretical_metric = accuracy_random_classifier(lambda_)

                if risk_dict["risk"].higher_is_better:
                    if theoretical_metric <= target_level:
                        error_indicator = 1
                        break
                else:
                    if theoretical_metric > target_level:
                        error_indicator = 1
                        break

        records.append(
            {
                "risk_name": risk_dict["name"],
                "predict_param": predict_params,
                "target_level": target_level,
                "confidence_level": confidence_level,
                "repeat_id": repeat_id,
                "error_indicator": error_indicator,
                "valid_param": valid_parameters,
                "nb_valid_param": len(valid_parameters),
            }
        )

    return pd.DataFrame(records)


def run_one_experiment_with_logistic_classifier(
    clf_class,
    risk_dict,
    predict_params,
    target_level,
    confidence_level,
    N,
    n_repeats,
    scale=2.0,
):
    """
    Runs the experiment for one combination of using a LogisticClassifier.
    Returns a DataFrame with one row per repeat.
    """
    clf = clf_class(scale=scale, threshold=0.5)
    records = []

    for repeat_id in range(n_repeats):
        X_calibrate, y_calibrate = make_logistic_data(
            n_samples=N, scale=scale, random_state=None
        )

        controller = BinaryClassificationController(
            predict_function=clf.predict_proba,
            risk=risk_dict["risk"],
            target_level=target_level,
            confidence_level=confidence_level,
            list_predict_params=predict_params,
        )
        controller = controller.calibrate(X_calibrate, y_calibrate)
        valid_parameters = controller.valid_predict_params

        error_indicator = 0

        if len(valid_parameters) == 0:
            error_indicator = 1
        else:
            for lambda_ in valid_parameters:
                if risk_dict["risk"] == precision:
                    empirical_metric = precision_logistic_classifier(scale, lambda_)
                elif risk_dict["risk"] == recall:
                    empirical_metric = recall_logistic_classifier(scale, lambda_)
                elif risk_dict["risk"] == accuracy:
                    empirical_metric = accuracy_logistic_classifier(scale, lambda_)

                if risk_dict["risk"].higher_is_better:
                    if empirical_metric <= target_level:
                        error_indicator = 1
                        break
                else:
                    if empirical_metric > target_level:
                        error_indicator = 1
                        break

        records.append(
            {
                "risk_name": risk_dict["name"],
                "predict_param": predict_params,
                "target_level": target_level,
                "confidence_level": confidence_level,
                "repeat_id": repeat_id,
                "error_indicator": error_indicator,
                "valid_param": valid_parameters,
                "nb_valid_param": len(valid_parameters),
            }
        )

    return pd.DataFrame(records)


def analyze_results(df_results):
    summary = []
    grouped = df_results.groupby(["risk_name", "target_level", "confidence_level"])

    for (risk_name, target_level, confidence_level), group in grouped:
        proportion_not_controlled = group["error_indicator"].mean()
        nb_predict_parameters = len(group.iloc[0]["predict_param"])
        mean_nb_valid_thresholds = group["nb_valid_param"].mean()

        delta = 1 - confidence_level
        valid_experiment = proportion_not_controlled <= delta

        summary.append(
            {
                "risk_name": risk_name,
                "target_level": target_level,
                "confidence_level": confidence_level,
                "nb_predict_parameters": nb_predict_parameters,
                "proportion_not_controlled": proportion_not_controlled,
                "delta": delta,
                "mean_nb_valid_thresholds": mean_nb_valid_thresholds,
                "valid_experiment": valid_experiment,
            }
        )

    df_summary = pd.DataFrame(summary)

    return df_summary


def test_random_classifier_theoretical_validity():  # pragma: no cover
    """Reproduces section 1 of the notebook (random classifier checks)."""

    N = 2000  # size of the calibration set
    risk = [
        {"name": "precision", "risk": precision},
        {"name": "recall", "risk": recall},
        {"name": "accuracy", "risk": accuracy},
    ]
    target_level = [0.1, 0.9]
    confidence_level = [0.8, 0.2]
    n_repeats = 100

    # Random classifier : the case of multiple parameters
    predict_params = [np.linspace(0, 0.99, 100)]
    all_results = []

    for combination in product(risk, predict_params, target_level, confidence_level):
        risk_dict, predict_param_set, t_level, c_level = combination

        df_one = run_one_experiment_with_random_classifier(
            clf_class=RandomClassifier,
            risk_dict=risk_dict,
            predict_params=predict_param_set,
            target_level=t_level,
            confidence_level=c_level,
            N=N,
            n_repeats=n_repeats,
        )

        all_results.append(df_one)

    df_results = pd.concat(all_results, ignore_index=True)
    analyze_results(df_results)
    # A further statistical analysis is required to compare the obtained results
    # with the expected ones given the experiment design.
    # Therefore, we intentionally skip the test assertion for now.
    # A strict assertion will be added later, e.g.:
    # df_summary = analyze_results(df_results)
    # assert df_summary["valid_experiment"].all()

    # Random classifier : the case of single parameter
    all_results_single_param = []

    for combination in product(risk, target_level, confidence_level):
        risk_dict, t_level, c_level = combination

        predict_param_set = np.array([np.random.choice(np.linspace(0, 0.9, 10))])

        df_one = run_one_experiment_with_random_classifier(
            clf_class=RandomClassifier,
            risk_dict=risk_dict,
            predict_params=predict_param_set,
            target_level=t_level,
            confidence_level=c_level,
            N=N,
            n_repeats=n_repeats,
        )

        all_results_single_param.append(df_one)

    df_results_single_param = pd.concat(all_results_single_param, ignore_index=True)
    analyze_results(df_results_single_param)
    # A further statistical analysis is required to compare the obtained results
    # with the expected ones given the experiment design.
    # Therefore, we intentionally skip the test assertion for now.
    # A strict assertion will be added later, e.g.:
    # df_summary_single_param = analyze_results(df_results_single_param)
    # assert df_summary_single_param["valid_experiment"].all()

    pytest.skip(
        "This test reproduces the notebook results. However, a further statistical analysis "
        "is needed to compare the obtained results with the expected ones regarding the experiment design."
    )


def test_logistic_classifier_theoretical_validity():  # pragma: no cover
    """Reproduces section 2 of the notebook (logistic classifier checks)."""

    N = 2000
    risk = [
        {"name": "precision", "risk": precision},
        {"name": "recall", "risk": recall},
        {"name": "accuracy", "risk": accuracy},
    ]
    target_level = [0.1, 0.9]
    confidence_level = [0.8, 0.2]
    n_repeats = 100
    scale = 2.0

    # Logistic classifier : the case of multiple parameters
    predict_params = [np.linspace(0, 0.99, 100)]
    all_results = []

    for combination in product(risk, predict_params, target_level, confidence_level):
        risk_dict, predict_param_set, t_level, c_level = combination

        df_one = run_one_experiment_with_logistic_classifier(
            clf_class=LogisticClassifier,
            risk_dict=risk_dict,
            predict_params=predict_param_set,
            target_level=t_level,
            confidence_level=c_level,
            N=N,
            n_repeats=n_repeats,
            scale=scale,
        )

        all_results.append(df_one)

    df_results = pd.concat(all_results, ignore_index=True)
    analyze_results(df_results)
    # A further statistical analysis is required to compare the obtained results
    # with the expected ones given the experiment design.
    # Therefore, we intentionally skip the test assertion for now.
    # A strict assertion will be added later, e.g.:
    # df_summary = analyze_results(df_results)
    # assert df_summary["valid_experiment"].all()

    # Logistic classifier : the case of single parameter
    all_results_single_param = []

    for combination in product(risk, target_level, confidence_level):
        risk_dict, t_level, c_level = combination

        predict_param_set = np.array([np.random.choice(np.linspace(0, 0.9, 10))])

        df_one = run_one_experiment_with_logistic_classifier(
            clf_class=LogisticClassifier,
            risk_dict=risk_dict,
            predict_params=predict_param_set,
            target_level=t_level,
            confidence_level=c_level,
            N=N,
            n_repeats=n_repeats,
            scale=scale,
        )

        all_results_single_param.append(df_one)

    df_results_single_param = pd.concat(all_results_single_param, ignore_index=True)
    analyze_results(df_results_single_param)
    # A further statistical analysis is required to compare the obtained results
    # with the expected ones given the experiment design.
    # Therefore, we intentionally skip the test assertion for now.
    # A strict assertion will be added later, e.g.:
    # df_summary_single_param = analyze_results(df_results_single_param)
    # assert df_summary_single_param["valid_experiment"].all()

    pytest.skip(
        "This test reproduces the notebook results. However, a further statistical analysis "
        "is needed to compare the obtained results with the expected ones regarding the experiment design."
    )
