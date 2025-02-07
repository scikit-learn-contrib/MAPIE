import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mapie.classification import MapieClassifier
from mapie_v1.classification import SplitConformalClassifier
from tests_v1.integration_tests.utils import (
    DummyClassifierWithFitAndPredictParams,
    train_test_split_shuffle,
)
from mapie._typing import ArrayLike

pytestmark = pytest.mark.classification

RANDOM_STATE = 1


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(
        n_samples=200,
        n_informative=5,
        n_classes=4,
        random_state=RANDOM_STATE
    )
    return {
        "X": X,
        "y": y,
        "sample_weight": RandomState(RANDOM_STATE).random(len(X)),
    }


@pytest.fixture()
def params_split_test_1(dataset):
    return {
        "v1": {},
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": "lac",
                "cv": "prefit"
            },
            "predict": {
                "alpha": 0.1,
            }}}


@pytest.fixture()
def params_split_test_2(dataset):
    return {
        "v1": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "confidence_level": 0.8,
                "prefit": False,
                "conformity_score": "top_k",
            },
            "fit": {
                "fit_params": {"fit_param": True},
            },
            "conformalize": {
                "predict_params": {"predict_param": True},
            }},
        "v0": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "conformity_score": "top_k",
                "cv": "split"
            },
            "fit": {
                "fit_params": {"fit_param": True},
            },
            "predict": {
                "alpha": 0.2,
                "predict_params": {"predict_param": True},
            }}}


@pytest.fixture()
def params_split_test_3(dataset):
    return {
        "v1": {
            "__init__": {
                "estimator": RandomForestClassifier(),
                "confidence_level": [0.8, 0.9],
                "prefit": False,
                "conformity_score": "aps",
            },
            "fit": {
                "fit_params": {"sample_weight": dataset["sample_weight"]},
            },
            "predict_set": {
                "conformity_score_params": {"include_last_label": False}
            }},
        "v0": {
            "__init__": {
                "estimator": RandomForestClassifier(),
                "conformity_score": "top_k",
                "cv": "split"
            },
            "fit": {
                "sample_weight": dataset["sample_weight"],
            },
            "predict": {
                "alpha": [0.2, 0.1],
                "include_last_label": False,
            }}}


@pytest.fixture()
def params_split_test_4(dataset):
    return {
        "v1": {
            "__init__": {
                "conformity_score": "raps",
            }},
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": "raps",
                "cv": "prefit"
            },
            "predict": {
                "alpha": 0.1,
            }}}


@pytest.mark.parametrize(
    "params", [
        "params_split_test_1",
        "params_split_test_2",
        "params_split_test_3",
        "params_split_test_4"
    ]
)
def test_intervals_and_predictions_exact_equality_split(dataset, params, request):
    params_ = request.getfixturevalue(params)
    v0_params = params_["v0"]
    v1_params = params_["v1"]

    X, y = dataset["X"], dataset["y"]
    X_train, X_conformalize, y_train, y_conformalize = train_test_split_shuffle(
        X, y, random_state=RANDOM_STATE
    )

    v0 = MapieClassifier(**v0_params.get("__init__", {}))
    v1 = SplitConformalClassifier(**v1_params.get("__init__", {}))

    if v1_params.get("__init__", {}).get("prefit", True):
        v0_params["estimator"].fit(X_train, y_train)
        v1_params["estimator"].fit(X_train, y_train)
    else:
        v0.fit(X_train, y_train)
        v1.fit(X_train, y_train)

    v1.conformalize(X_conformalize, y_conformalize, **v1_params.get("conformalize", {}))

    v0_preds, v0_pred_sets = v0.predict(X_conformalize, **v0_params.get("predict", {}))
    v1_preds, v1_pred_sets = v1.predict_set(
        X_conformalize,
        **v1_params.get("predict_set", {})
    )

    v1_preds_using_predict: ArrayLike = v1.predict(X_conformalize)

    np.testing.assert_array_equal(v0_preds, v1_preds)
    np.testing.assert_array_equal(v0_pred_sets, v1_pred_sets)
    np.testing.assert_array_equal(v1_preds_using_predict, v1_preds)
    assert v1_pred_sets.shape == (
        len(X_conformalize),
        len(np.unique(y)),
        len(v1_params.get("confidence_level", [0.9]))
    )
