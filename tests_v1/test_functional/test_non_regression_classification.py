import numpy as np
import pytest
from numpy.random import RandomState
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, GroupKFold

from mapie.classification import MapieClassifier
from mapie.conformity_scores import (
    RAPSConformityScore,
    APSConformityScore,
    TopKConformityScore,
    LACConformityScore,
)
from mapie_v1.classification import SplitConformalClassifier, CrossConformalClassifier
from tests_v1.test_functional.utils import (
    DummyClassifierWithFitAndPredictParams,
    train_test_split_shuffle,
)
from numpy.typing import ArrayLike

RANDOM_STATE = 1


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(
        n_samples=1000,
        n_informative=5,
        n_classes=4,
        random_state=RANDOM_STATE
    )
    sample_weight = RandomState(RANDOM_STATE).random(len(X))
    groups = np.array([i % 5 for i in range(len(X))])

    (
        X_train,
        X_conformalize,
        y_train,
        y_conformalize,
        sample_weight_train,
        sample_weight_conformalize,
    ) = train_test_split_shuffle(
        X, y, random_state=RANDOM_STATE, sample_weight=sample_weight
    )

    return {
        "X": X,
        "y": y,
        "sample_weight": sample_weight,
        "groups": groups,
        "X_train": X_train,
        "X_conformalize": X_conformalize,
        "y_train": y_train,
        "y_conformalize": y_conformalize,
        "sample_weight_train": sample_weight_train,
        "sample_weight_conformalize": sample_weight_conformalize,
    }


@pytest.fixture()
def params_split_test_1():
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
            },
        },
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": LACConformityScore(),
                "cv": "prefit"
            },
            "predict": {
                "alpha": 0.1,
            }}}


@pytest.fixture()
def params_split_test_2():
    return {
        "v1": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "confidence_level": 0.8,
                "prefit": False,
                "conformity_score": "top_k",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "fit_params": {"dummy_fit_param": True},
            },
            "conformalize": {
                "predict_params": {"dummy_predict_param": True},
            }},
        "v0": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "conformity_score": TopKConformityScore(),
                "cv": "split",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "fit_params": {"dummy_fit_param": True},
                "predict_params": {"dummy_predict_param": True},
            },
            "predict": {
                "alpha": 0.2,
                "dummy_predict_param": True,
            }}}


@pytest.fixture()
def params_split_test_3(dataset):
    return {
        "v1": {
            "__init__": {
                "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
                "confidence_level": [0.8, 0.9],
                "prefit": False,
                "conformity_score": "aps",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "fit_params": {"sample_weight": dataset["sample_weight_train"]},
            },
            "predict_set": {
                "conformity_score_params": {"include_last_label": False}
            }},
        "v0": {
            "__init__": {
                "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
                "conformity_score": APSConformityScore(),
                "cv": "split",
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "sample_weight": dataset["sample_weight"],
            },
            "predict": {
                "alpha": [0.2, 0.1],
                "include_last_label": False,
            }}}


@pytest.fixture()
def params_split_test_4():
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": "raps",
                "random_state": RANDOM_STATE,
            }},
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": RAPSConformityScore(),
                "cv": "prefit",
                "random_state": RANDOM_STATE,
            },
            "predict": {
                "alpha": 0.1,
            }}}


@pytest.fixture()
def params_split_test_5():
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": RAPSConformityScore(size_raps=0.4),
                "random_state": RANDOM_STATE,
            }},
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": RAPSConformityScore(size_raps=0.4),
                "cv": "prefit",
                "random_state": RANDOM_STATE,
            },
            "predict": {
                "alpha": 0.1,
            }}}


@pytest.mark.parametrize(
    "params_", [
        "params_split_test_1",
        "params_split_test_2",
        "params_split_test_3",
        "params_split_test_4",
        "params_split_test_5",
    ]
)
def test_split(dataset, params_, request):
    X, y, X_train, X_conformalize, y_train, y_conformalize = (
        dataset["X"],
        dataset["y"],
        dataset["X_train"],
        dataset["X_conformalize"],
        dataset["y_train"],
        dataset["y_conformalize"],
    )

    params = extract_params(request.getfixturevalue(params_))

    prefit = params["v1_init"].get("prefit", True)

    if prefit:
        params["v0_init"]["estimator"].fit(X_train, y_train)
        params["v1_init"]["estimator"].fit(X_train, y_train)

    v0 = MapieClassifier(**params["v0_init"])
    v1 = SplitConformalClassifier(**params["v1_init"])

    if prefit:
        v0.fit(X_conformalize, y_conformalize, **params["v0_fit"])
    else:
        v0.fit(X, y, **params["v0_fit"])
        v1.fit(X_train, y_train, **params["v1_fit"])
    v1.conformalize(X_conformalize, y_conformalize, **params["v1_conformalize"])

    v0_preds, v0_pred_sets = v0.predict(X_conformalize, **params["v0_predict"])
    v1_preds, v1_pred_sets = v1.predict_set(X_conformalize, **params["v1_predict_set"])

    v1_preds_using_predict: ArrayLike = v1.predict(X_conformalize)

    np.testing.assert_array_equal(v0_preds, v1_preds)
    np.testing.assert_array_equal(v0_pred_sets, v1_pred_sets)
    np.testing.assert_array_equal(v1_preds_using_predict, v1_preds)

    n_confidence_level = get_number_of_confidence_levels(params["v1_init"])

    assert v1_pred_sets.shape == (
        len(X_conformalize),
        len(np.unique(y)),
        n_confidence_level,
    )


@pytest.fixture()
def params_cross_test_1(dataset):
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
                "confidence_level": 0.8,
                "conformity_score": "lac",
                "cv": 4,
                "random_state": RANDOM_STATE,
            },
            "fit_conformalize": {
                "fit_params": {"sample_weight": dataset["sample_weight"]},
            },
        },
        "v0": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": LACConformityScore(),
                "cv": 4,
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "sample_weight": dataset["sample_weight"],
            },
            "predict": {
                "alpha": 0.2,
            }}}


@pytest.fixture()
def params_cross_test_2():
    return {
        "v1": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "confidence_level": [0.9, 0.8],
                "conformity_score": "aps",
                "cv": LeaveOneOut(),
                "random_state": RANDOM_STATE,
            },
            "fit_conformalize": {
                "predict_params": {"dummy_predict_param": True},
            },
            "predict_set": {
                "conformity_score_params": {"include_last_label": False}
            },
        },
        "v0": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "conformity_score": APSConformityScore(),
                "cv": LeaveOneOut(),
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "predict_params": {"dummy_predict_param": True},
            },
            "predict": {
                "alpha": [0.1, 0.2],
                "include_last_label": False,
                "dummy_predict_param": True,
            }}}


@pytest.fixture()
def params_cross_test_3(dataset):
    return {
        "v1": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "cv": GroupKFold(),
                "random_state": RANDOM_STATE,
            },
            "fit_conformalize": {
                "groups": dataset["groups"],
                "fit_params": {"dummy_fit_param": True},
            },
            "predict_set": {
                "agg_scores": "crossval",
            },
        },
        "v0": {
            "__init__": {
                "estimator": DummyClassifierWithFitAndPredictParams(),
                "cv": GroupKFold(),
                "random_state": RANDOM_STATE,
            },
            "fit": {
                "groups": dataset["groups"],
                "fit_params": {"dummy_fit_param": True},
            },
            "predict": {
                "alpha": 0.1,
                "agg_scores": "crossval",
            }}}


@pytest.fixture()
def params_cross_test_4():
    return {
        "v1": {
            "__init__": {
                "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
                "confidence_level": 0.7,
                "conformity_score": LACConformityScore(),
                "random_state": RANDOM_STATE,
            },
        },
        "v0": {
            "__init__": {
                "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
                "cv": 5,
                "random_state": RANDOM_STATE,
            },
            "predict": {
                "alpha": 0.3,
            }}}


@pytest.mark.parametrize(
    "params_", [
        "params_cross_test_1",
        "params_cross_test_2",
        "params_cross_test_3",
        "params_cross_test_4",
    ]
)
def test_cross(dataset, params_, request):
    X, y = dataset["X"], dataset["y"]

    params = extract_params(request.getfixturevalue(params_))

    v0 = MapieClassifier(**params["v0_init"])
    v1 = CrossConformalClassifier(**params["v1_init"])

    v0.fit(X, y, **params["v0_fit"])
    v1.fit_conformalize(X, y, **params["v1_fit_conformalize"])

    v0_preds, v0_pred_sets = v0.predict(X, **params["v0_predict"])
    v1_preds, v1_pred_sets = v1.predict_set(X, **params["v1_predict_set"])

    v1_preds_using_predict: ArrayLike = v1.predict(X)

    np.testing.assert_array_equal(v0_preds, v1_preds)
    np.testing.assert_array_equal(v0_pred_sets, v1_pred_sets)
    np.testing.assert_array_equal(v1_preds_using_predict, v1_preds)

    n_confidence_level = get_number_of_confidence_levels(params["v1_init"])
    assert v1_pred_sets.shape == (
        len(X),
        len(np.unique(y)),
        n_confidence_level,
    )


def extract_params(params):
    return {
        "v0_init": params["v0"].get("__init__", {}),
        "v0_fit": params["v0"].get("fit", {}),
        "v0_predict": params["v0"].get("predict", {}),
        "v1_init": params["v1"].get("__init__", {}),
        "v1_fit": params["v1"].get("fit", {}),
        "v1_conformalize": params["v1"].get("conformalize", {}),
        "v1_predict_set": params["v1"].get("predict_set", {}),
        "v1_fit_conformalize": params["v1"].get("fit_conformalize", {})
    }


def get_number_of_confidence_levels(v1_init_params):
    confidence_level = v1_init_params.get("confidence_level", 0.9)
    return 1 if isinstance(confidence_level, float) else len(confidence_level)
