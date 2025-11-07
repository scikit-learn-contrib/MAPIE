"""Tests converted from the notebook:
mapie/notebooks/risk_control/theoretical_validity_tests.ipynb

This file reproduces the notebook logic (random classifier and logistic classifier
theoretical validity checks) as pytest tests.
"""

from itertools import product
from decimal import Decimal

import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import check_random_state
from sklearn.metrics import precision_score, recall_score, accuracy_score

from mapie.risk_control import (
    BinaryClassificationController,
    precision,
    accuracy,
    recall,
)


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

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


class LogisticClassifier:
    """Deterministic sigmoid-based binary classifier."""

    def __init__(self, scale=2.0, threshold=0.5):
        self.scale = scale
        self.threshold = threshold

    def _get_prob(self, x):
        return 1 / (1 + np.exp(-self.scale * x))

    def predict_proba(self, X):
        probs = np.array([self._get_prob(x) for x in X])
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


def make_logistic_data(n_samples=200, scale=2.0, random_state=None):
    rng = check_random_state(random_state)
    X = rng.uniform(-3, 3, size=n_samples)
    probs = 1 / (1 + np.exp(-scale * X))
    y = rng.binomial(1, probs)
    return X, y


def test_random_classifier_theoretical_validity():
    """Reproduces section 1 of the notebook (random classifier checks)."""

    N = 2000  # size of the calibration set (same as notebook)
    risk = [
        {"name": "precision", "risk": precision},
        {"name": "recall", "risk": recall},
        {"name": "accuracy", "risk": accuracy},
    ]
    predict_params = [np.linspace(0, 0.99, 100), np.empty(1)]
    target_level = [0.1, 0.9]
    confidence_level = [0.8, 0.2]

    n_repeats = 100
    invalid_experiment = False

    for combination in product(risk, predict_params, target_level, confidence_level):
        risk_item, predict_params_item, target_level_item, confidence_level_item = (
            combination
        )
        if len(predict_params_item) == 1:
            predict_params_item = np.array([np.random.choice(np.linspace(0, 0.9, 10))])
        alpha = float(Decimal("1") - Decimal(str(target_level_item)))
        delta = float(Decimal("1") - Decimal(str(confidence_level_item)))

        clf = RandomClassifier()
        nb_errors = 0
        total_nb_valid_params = 0

        for _ in range(n_repeats):
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
                risk=risk_item["risk"],
                target_level=target_level_item,
                confidence_level=confidence_level_item,
            )
            controller._predict_params = predict_params_item
            controller.calibrate(X_calibrate, y_calibrate)
            valid_parameters = controller.valid_predict_params
            total_nb_valid_params += len(valid_parameters)

            # Using theoretical risk knowledge for the random classifier (balanced generator)
            if risk_item["risk"] == precision or risk_item["risk"] == accuracy:
                if target_level_item > 0.5 and len(valid_parameters) >= 1:
                    nb_errors += 1
            elif risk_item["risk"] == recall:
                if (
                    any(x > alpha for x in valid_parameters)
                    and len(valid_parameters) >= 1
                ):
                    nb_errors += 1

        # Basic checks mirroring the notebook prints (but use assertions)
        proportion_not_controlled = nb_errors / n_repeats
        assert proportion_not_controlled <= delta or proportion_not_controlled >= 0
        # Keep track if any experiment was invalid (mirrors notebook behaviour)
        if proportion_not_controlled > delta:
            invalid_experiment = True

    assert not invalid_experiment


def test_logistic_classifier_theoretical_validity():
    """Reproduces section 2 of the notebook (logistic classifier checks)."""

    N = 2000
    risk = [
        {"name": "precision", "risk": precision},
        {"name": "recall", "risk": recall},
        {"name": "accuracy", "risk": accuracy},
    ]
    predict_params = [np.linspace(0, 0.99, 100), np.empty(1)]
    target_level = [0.1, 0.9]
    confidence_level = [0.8, 0.2]

    n_repeats = 100
    invalid_experiment = False

    for combination in product(risk, predict_params, target_level, confidence_level):
        risk_item, predict_params_item, target_level_item, confidence_level_item = (
            combination
        )
        if len(predict_params_item) == 1:
            predict_params_item = np.array([np.random.choice(np.linspace(0, 0.9, 10))])
        delta = float(Decimal("1") - Decimal(str(confidence_level_item)))

        clf = LogisticClassifier(scale=2.0, threshold=0.5)
        nb_errors = 0
        total_nb_valid_params = 0

        for _ in range(n_repeats):
            X_calibrate, y_calibrate = make_logistic_data(
                n_samples=N, scale=2.0, random_state=None
            )

            controller = BinaryClassificationController(
                predict_function=clf.predict_proba,
                risk=risk_item["risk"],
                target_level=target_level_item,
                confidence_level=confidence_level_item,
            )
            controller._predict_params = predict_params_item
            controller = controller.calibrate(X_calibrate, y_calibrate)
            valid_parameters = controller.valid_predict_params
            total_nb_valid_params += len(valid_parameters)

            # Estimate empirical risk on a fresh test set
            X_test, y_test = make_logistic_data(
                n_samples=N, scale=2.0, random_state=None
            )
            probs = clf.predict_proba(X_test)[:, 1]

            if len(valid_parameters) >= 1:
                for lambda_ in valid_parameters:
                    y_pred = (probs >= lambda_).astype(int)

                    if risk_item["risk"] == precision:
                        empirical_metric = precision_score(
                            y_test, y_pred, zero_division=0
                        )
                    elif risk_item["risk"] == recall:
                        empirical_metric = recall_score(y_test, y_pred, zero_division=0)
                    elif risk_item["risk"] == accuracy:
                        empirical_metric = accuracy_score(y_test, y_pred)

                    # Check if the risk control fails according to higher_is_better flag
                    if risk_item["risk"].higher_is_better:
                        if empirical_metric <= target_level_item:
                            nb_errors += 1
                            break
                    else:
                        if empirical_metric > target_level_item:
                            nb_errors += 1
                            break

        proportion_not_controlled = nb_errors / n_repeats
        assert proportion_not_controlled <= delta or proportion_not_controlled >= 0
        if proportion_not_controlled > delta:
            invalid_experiment = True

    assert not invalid_experiment
