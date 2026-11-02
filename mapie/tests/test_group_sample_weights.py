import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold

from mapie.classification import CrossConformalClassifier
from mapie.regression import CrossConformalRegressor


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_group_sample_weights_match_explicit_filtering(task):
    if task == "classification":
        X, y = make_classification(n_samples=36, n_features=4, random_state=7)
        conformal_class = CrossConformalClassifier
        estimator = LogisticRegression()
        prediction_method = "predict_set"
    else:
        X, y = make_regression(n_samples=36, n_features=4, random_state=7)
        conformal_class = CrossConformalRegressor
        estimator = LinearRegression(n_jobs=1)
        prediction_method = "predict_interval"

    groups = np.repeat(np.arange(6), 6)
    weights = np.linspace(1.0, 2.0, len(y))
    weights[[0, 7, 14]] = 0
    keep = weights != 0

    expected = conformal_class(
        estimator=estimator, cv=GroupKFold(3), confidence_level=0.8, n_jobs=1
    ).fit_conformalize(
        X[keep],
        y[keep],
        groups=groups[keep],
        fit_params={"sample_weight": weights[keep]},
    )
    expected_predictions = getattr(expected, prediction_method)(X[:3])
    assert expected_predictions[0].shape == (3,)
    assert expected.conformity_scores.shape[0] == keep.sum()

    actual = conformal_class(
        estimator=estimator, cv=GroupKFold(3), confidence_level=0.8, n_jobs=1
    ).fit_conformalize(X, y, groups=groups, fit_params={"sample_weight": weights})

    np.testing.assert_allclose(actual.conformity_scores, expected.conformity_scores)
    for actual_values, expected_values in zip(
        getattr(actual, prediction_method)(X[:3]), expected_predictions
    ):
        np.testing.assert_allclose(actual_values, expected_values)
