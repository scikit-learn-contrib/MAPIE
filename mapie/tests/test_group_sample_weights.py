import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold

from mapie.classification import CrossConformalClassifier
from mapie.regression import CrossConformalRegressor


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("group_type", [np.asarray, list, pd.Series])
@pytest.mark.parametrize("weight_mode", ["none", "positive", "zero"])
def test_group_sample_weights_match_explicit_filtering(task, group_type, weight_mode):
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
    if weight_mode == "zero":
        weights[[0, 7, 14]] = 0
    keep = weights != 0
    fit_params = {} if weight_mode == "none" else {"sample_weight": weights}
    filtered_fit_params = (
        {} if weight_mode == "none" else {"sample_weight": weights[keep]}
    )

    expected = conformal_class(
        estimator=estimator, cv=GroupKFold(3), confidence_level=0.8, n_jobs=1
    ).fit_conformalize(
        X[keep], y[keep], groups=groups[keep], fit_params=filtered_fit_params
    )
    expected_predictions = getattr(expected, prediction_method)(X[:3])
    assert expected_predictions[0].shape == (3,)
    assert expected.conformity_scores.shape[0] == keep.sum()

    actual = conformal_class(
        estimator=estimator, cv=GroupKFold(3), confidence_level=0.8, n_jobs=1
    ).fit_conformalize(X, y, groups=group_type(groups), fit_params=fit_params)

    np.testing.assert_allclose(actual.conformity_scores, expected.conformity_scores)
    for actual_values, expected_values in zip(
        getattr(actual, prediction_method)(X[:3]), expected_predictions
    ):
        np.testing.assert_allclose(actual_values, expected_values)
