import inspect
from typing import Dict, Optional, Callable, Any, Tuple

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from numpy._typing import ArrayLike, NDArray
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, QuantileRegressor
from sklearn.model_selection import LeaveOneOut, GroupKFold, train_test_split, \
    ShuffleSplit
from typing_extensions import Self

from mapie.classification import SplitConformalClassifier, \
    CrossConformalClassifier
from mapie.conformity_scores import LACConformityScore, RAPSConformityScore, \
    GammaConformityScore, ResidualNormalisedScore
from mapie.regression import CrossConformalRegressor, JackknifeAfterBootstrapRegressor
from mapie.regression.quantile_regression import ConformalizedQuantileRegressor
from mapie.regression.regression import SplitConformalRegressor
from mapie.subsample import Subsample

RANDOM_STATE = 1
K_FOLDS = 3
N_BOOTSTRAPS = 30
N_SAMPLES = 200
N_GROUPS = 5


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
    }


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
    }


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
    }


@pytest.fixture()
def params_split_test_4():
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": "raps",
                "random_state": RANDOM_STATE,
            }},
    }


@pytest.fixture()
def params_split_test_5():
    return {
        "v1": {
            "__init__": {
                "estimator": LogisticRegression(),
                "conformity_score": RAPSConformityScore(size_raps=0.4),
                "random_state": RANDOM_STATE,
            }},
    }


@pytest.mark.parametrize(
    "params_", [
        "params_split_test_1",
        "params_split_test_2",
        "params_split_test_3",
        "params_split_test_4",
        "params_split_test_5",
    ]
)
def test_split(
    dataset: Dict[str, Any],
    params_: str,
    request: FixtureRequest
) -> None:
    _, y, X_train, X_conformalize, y_train, y_conformalize = (
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
        params["v1_init"]["estimator"].fit(X_train, y_train)

    v1 = SplitConformalClassifier(**params["v1_init"])

    if not prefit:
        v1.fit(X_train, y_train, **params["v1_fit"])
    v1.conformalize(X_conformalize, y_conformalize, **params["v1_conformalize"])

    v1_preds, v1_pred_sets = v1.predict_set(X_conformalize, **params["v1_predict_set"])

    v1_preds_using_predict: NDArray = v1.predict(X_conformalize)

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
    }


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
    }


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
    }


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
    }


@pytest.mark.parametrize(
    "params_", [
        "params_cross_test_1",
        "params_cross_test_2",
        "params_cross_test_3",
        "params_cross_test_4",
    ]
)
def test_cross(
    dataset: Dict[str, Any],
    params_: str,
    request: FixtureRequest
):
    X, y = dataset["X"], dataset["y"]

    params = extract_params(request.getfixturevalue(params_))

    v1 = CrossConformalClassifier(**params["v1_init"])

    v1.fit_conformalize(X, y, **params["v1_fit_conformalize"])

    v1_preds, v1_pred_sets = v1.predict_set(X, **params["v1_predict_set"])

    v1_preds_using_predict: NDArray = v1.predict(X)

    np.testing.assert_array_equal(v1_preds_using_predict, v1_preds)

    n_confidence_level = get_number_of_confidence_levels(params["v1_init"])
    assert v1_pred_sets.shape == (
        len(X),
        len(np.unique(y)),
        n_confidence_level,
    )


def extract_params(params):
    return {
        "v1_init": params["v1"].get("__init__", {}),
        "v1_fit": params["v1"].get("fit", {}),
        "v1_conformalize": params["v1"].get("conformalize", {}),
        "v1_predict_set": params["v1"].get("predict_set", {}),
        "v1_fit_conformalize": params["v1"].get("fit_conformalize", {})
    }


def get_number_of_confidence_levels(v1_init_params):
    confidence_level = v1_init_params.get("confidence_level", 0.9)
    return 1 if isinstance(confidence_level, float) else len(confidence_level)


X, y_signed = make_regression(
    n_samples=N_SAMPLES,
    n_features=10,
    noise=1.0,
    random_state=RANDOM_STATE
)
y = np.abs(y_signed)
sample_weight = RandomState(RANDOM_STATE).random(len(X))
groups = [j for j in range(N_GROUPS) for i in range((N_SAMPLES//N_GROUPS))]
positive_predictor = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=lambda y_: np.log(y_ + 1),
    inverse_func=lambda X_: np.exp(X_) - 1
)

sample_weight_train = train_test_split(
    X,
    y,
    sample_weight,
    test_size=0.4,
    random_state=RANDOM_STATE
)[-2]


params_test_cases_cross = [
    {
        "v1": {
            "class": CrossConformalRegressor,
            "__init__": {
                "confidence_level": 0.8,
                "conformity_score": "absolute",
                "cv": 4,
                "method": "base",
                "random_state": RANDOM_STATE,
            },
            "fit_conformalize": {
                "fit_params": {"sample_weight": sample_weight},
            },
            "predict_interval": {
                "aggregate_predictions": "median",
            },
            "predict": {
                "aggregate_predictions": "median",
            }
        },
    },
    {
        "v1": {
            "class": CrossConformalRegressor,
            "__init__": {
                "estimator": positive_predictor,
                "confidence_level": [0.5, 0.5],
                "conformity_score": "gamma",
                "cv": LeaveOneOut(),
                "method": "plus",
                "random_state": RANDOM_STATE,
            },
            "predict_interval": {
                "minimize_interval_width": True,
            },
        },
    },
    {
        "v1": {
            "class": CrossConformalRegressor,
            "__init__": {
                "cv": GroupKFold(),
                "method": "minmax",
                "random_state": RANDOM_STATE,
            },
            "fit_conformalize": {
                "groups": groups,
            },
            "predict_interval": {
                "allow_infinite_bounds": True,
                "aggregate_predictions": None,
            },
            "predict": {
                "aggregate_predictions": None,
            },
        },
    },
]

params_test_cases_jackknife = [
    {
        "v1": {
            "class": JackknifeAfterBootstrapRegressor,
            "__init__": {
                "confidence_level": 0.8,
                "conformity_score": "absolute",
                "resampling": Subsample(
                    n_resamplings=15, random_state=RANDOM_STATE
                ),
                "aggregation_method": "median",
                "method": "plus",
                "random_state": RANDOM_STATE,
            },
            "fit_conformalize": {
                "fit_params": {"sample_weight": sample_weight},
            },
        },
    },
    {
        "v1": {
            "class": JackknifeAfterBootstrapRegressor,
            "__init__": {
                "estimator": positive_predictor,
                "confidence_level": [0.5, 0.5],
                "aggregation_method": "mean",
                "conformity_score": "gamma",
                "resampling": Subsample(
                    n_resamplings=20,
                    replace=True,
                    random_state=RANDOM_STATE
                ),
                "method": "plus",
                "random_state": RANDOM_STATE,
            },
            "predict_interval": {
                "minimize_interval_width": True,
            },
        },
    },
    {
        "v1": {
            "class": JackknifeAfterBootstrapRegressor,
            "__init__": {
                "confidence_level": 0.9,
                "resampling": Subsample(
                    n_resamplings=30, random_state=RANDOM_STATE
                ),
                "method": "minmax",
                "aggregation_method": "mean",
                "random_state": RANDOM_STATE,
            },
            "predict_interval": {
                "allow_infinite_bounds": True,
            },
        },
    },
]


def run_v1_pipeline_cross_or_jackknife(params):
    params_ = params["v1"]
    init_params = params_.get("__init__", {})
    confidence_level = init_params.get("confidence_level", 0.9)
    confidence_level_length = 1 if isinstance(confidence_level, float) else len(
        confidence_level
    )
    minimize_interval_width = params_.get("predict_interval", {}).get(
        "minimize_interval_width"
    )

    mapie_regressor = params_["class"](**init_params)
    mapie_regressor.fit_conformalize(X, y, **params_.get("fit_conformalize", {}))

    X_test = X
    preds, pred_intervals = mapie_regressor.predict_interval(
        X_test,
        **params_.get("predict_interval", {})
    )
    preds_using_predict = mapie_regressor.predict(
        X_test,
        **params_.get("predict", {})
    )

    return (
        preds,
        pred_intervals,
        preds_using_predict,
        len(X_test),
        confidence_level_length,
        minimize_interval_width,
    )


@pytest.mark.parametrize(
    "params",
    params_test_cases_cross + params_test_cases_jackknife
)
def test_cross_and_jackknife(params: dict) -> None:
    (
        v1_preds,
        v1_pred_intervals,
        v1_preds_using_predict,
        X_test_length,
        confidence_level_length,
        minimize_interval_width,
    ) = run_v1_pipeline_cross_or_jackknife(params)

    np.testing.assert_array_equal(v1_preds_using_predict, v1_preds)

    if not minimize_interval_width:
        # condition to remove when optimize_beta/minimize_interval_width works
        # but keep assertion to check shapes
        assert v1_pred_intervals.shape == (X_test_length, 2, confidence_level_length)


params_test_cases_split = [
    {
        "v1": {
            "confidence_level": 0.8,
            "conformity_score": "absolute",
            "prefit": False,
            "test_size": 0.4,
            "fit_params": {"sample_weight": sample_weight_train},
            "class": SplitConformalRegressor
        }
    },
    {
        "v1": {
            "estimator": positive_predictor,
            "test_size": 0.2,
            "confidence_level": [0.5, 0.5],
            "conformity_score": "gamma",
            "prefit": False,
            "class": SplitConformalRegressor
        }
    },
    {
        "v1": {
            "estimator": LinearRegression(),
            "confidence_level": 0.9,
            "prefit": True,
            "test_size": 0.2,
            "conformity_score": ResidualNormalisedScore(
                random_state=RANDOM_STATE
            ),
            "allow_infinite_bounds": True,
            "class": SplitConformalRegressor
        }
    },
    {
        "v1": {
            "estimator": positive_predictor,
            "confidence_level": 0.9,
            "prefit": False,
            "conformity_score": GammaConformityScore(),
            "test_size": 0.3,
            "minimize_interval_width": True,
            "class": SplitConformalRegressor
        }
    },
]

split_model = QuantileRegressor(
                solver="highs-ds",
                alpha=0.0,
            )

gbr_models = []
gbr_alpha = 0.1

for alpha_ in [gbr_alpha / 2, (1 - (gbr_alpha / 2)), 0.5]:
    estimator_ = GradientBoostingRegressor(
        loss='quantile',
        alpha=alpha_,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )
    gbr_models.append(estimator_)

params_test_cases_quantile = [
    {
        "v1": {
            "confidence_level": 0.8,
            "prefit": False,
            "test_size": 0.4,
            "fit_params": {"sample_weight": sample_weight_train},
            "class": ConformalizedQuantileRegressor
        },
    },
    {
        "v1": {
            "estimator": gbr_models,
            "confidence_level": 1-gbr_alpha,
            "prefit": True,
            "test_size": 0.2,
            "fit_params": {"sample_weight": sample_weight},
            "minimize_interval_width": True,
            "symmetric_correction": True,
            "class": ConformalizedQuantileRegressor
        },
    },
    {
        "v1": {
            "estimator": split_model,
            "confidence_level": 0.5,
            "prefit": False,
            "test_size": 0.3,
            "allow_infinite_bounds": True,
            "class": ConformalizedQuantileRegressor
        },
    },
    {
        "v1": {
            "confidence_level": 0.9,
            "prefit": False,
            "test_size": 0.3,
            "symmetric_correction": True,
            "class": ConformalizedQuantileRegressor
        },
    },
]


def run_v1_pipeline_split_or_quantile(params):
    params_ = params["v1"]
    test_size = params_["test_size"]
    prefit = params_["prefit"]
    minimize_interval_width = params_.get("minimal_interval_width", False)

    if isinstance(params_["confidence_level"], float):
        n_confidence_level = 1
    else:
        n_confidence_level = len(params_["confidence_level"])

    (
        X_train,
        X_conf,
        y_train,
        y_conf,
        sample_weight_train,
        sample_weight_conf,
    ) = train_test_split_shuffle(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    if prefit:
        estimator = params_["estimator"]
        if isinstance(estimator, list):
            for single_estimator in estimator:
                single_estimator.fit(X_train, y_train)
        else:
            estimator.fit(X_train, y_train)

        params_["estimator"] = estimator

    init_params = filter_params(params_["class"].__init__, params_)

    mapie_regressor = params_["class"](**init_params)

    fit_params = filter_params(mapie_regressor.fit, params_)
    conformalize_params = filter_params(mapie_regressor.conformalize, params_)

    if not prefit:
        mapie_regressor.fit(X_train, y_train, **fit_params)

    mapie_regressor.conformalize(X_conf, y_conf, **conformalize_params)

    predict_params = filter_params(mapie_regressor.predict, params_)
    predict_interval_params = filter_params(mapie_regressor.predict_interval, params_)

    preds, pred_intervals = mapie_regressor.predict_interval(
        X_conf, **predict_interval_params
    )

    preds_using_predict: ArrayLike = mapie_regressor.predict(X_conf, **predict_params)

    return (
        n_confidence_level,
        minimize_interval_width,
        X_conf,
        preds,
        pred_intervals,
        preds_using_predict
    )


@pytest.mark.parametrize(
    "params",
    params_test_cases_split + params_test_cases_quantile
)
def test_intervals_and_predictions_exact_equality_split_and_quantile(
        params: dict) -> None:
    (
        n_confidence_level,
        minimize_interval_width,
        X_conf,
        v1_preds,
        v1_pred_intervals,
        v1_preds_using_predict
    ) = run_v1_pipeline_split_or_quantile(params)

    np.testing.assert_array_equal(v1_preds_using_predict, v1_preds)
    if not minimize_interval_width:
        # condition to remove when optimize_beta/minimize_interval_width works
        # but keep assertion to check shapes
        assert v1_pred_intervals.shape == (len(X_conf), 2, n_confidence_level)


def train_test_split_shuffle(
    X: NDArray,
    y: NDArray,
    test_size: Optional[float] = None,
    random_state: int = 42,
    sample_weight: Optional[NDArray] = None,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    splitter = ShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(X))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if sample_weight is not None:
        sample_weight_train = sample_weight[train_idx]
        sample_weight_test = sample_weight[test_idx]
    else:
        sample_weight_train = None
        sample_weight_test = None

    return X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test


def filter_params(
    function: Callable,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    model_params = inspect.signature(function).parameters
    return {k: v for k, v in params.items() if k in model_params}


class DummyClassifierWithFitAndPredictParams(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self._dummy_fit_param = None

    def fit(self, X: NDArray, y: NDArray, dummy_fit_param: bool = False) -> Self:
        self.classes_ = np.unique(y)
        self._dummy_fit_param = dummy_fit_param
        return self

    def predict_proba(self, X: NDArray, dummy_predict_param: bool = False) -> NDArray:
        probas = np.zeros((len(X), len(self.classes_)))
        probas[:, 0] = 0.1
        probas[:, 1] = 0.9
        return probas
