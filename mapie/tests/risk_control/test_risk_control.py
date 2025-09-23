from copy import deepcopy
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import precision_score, recall_score
from sklearn.dummy import DummyClassifier
from typing_extensions import TypedDict

from numpy.typing import NDArray
from mapie.risk_control import (
    PrecisionRecallController,
    precision,
    recall,
    BinaryClassificationRisk, false_positive_rate,
    BinaryClassificationController, accuracy,
)

Params = TypedDict(
    "Params",
    {
        "method": str,
        "bound": Optional[str],
        "random_state": Optional[int],
        "metric_control": Optional[str]
    }
)

METHODS = ["crc", "rcps", "ltt"]
METRICS = ['recall', 'precision']

BOUNDS = ["wsr", "hoeffding", "bernstein"]
random_state = 42

WRONG_METHODS = ["rpcs", "rcr", "test", "llt"]
WRONG_BOUNDS = ["wrs", "hoeff", "test", "", 1, 2.5, (1, 2)]
WRONG_METRICS = ["presicion", "recal", ""]


STRATEGIES = {
    "crc": (
        Params(
            method="crc",
            bound=None,
            random_state=random_state,
            metric_control="recall"
        ),
    ),
    "rcps_wsr": (
        Params(
            method="rcps",
            bound="wsr",
            random_state=random_state,
            metric_control='recall'
        ),
    ),
    "rcps_hoeffding": (
        Params(
            method="rcps",
            bound="hoeffding",
            random_state=random_state,
            metric_control='recall'
        ),
    ),
    "rcps_bernstein": (
        Params(
            method="rcps",
            bound="bernstein",
            random_state=random_state,
            metric_control='recall'
        ),
    ),
    "ltt": (
        Params(
            method="ltt",
            bound=None,
            random_state=random_state,
            metric_control='precision'
        ),
    ),
}


y_toy_mapie = {
    "crc": [
        [True, False, True],
        [True, False, True],
        [True, False, True],
        [True, False, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [False, True, True]
    ],
    "rcps_bernstein": [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ],
    "rcps_wsr": [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ],
    "rcps_hoeffding": [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ],
    "ltt": [
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False],
    ]
}


class WrongOutputModel:

    def __init__(self):
        pass

    def predict_proba(self, *args: Any):
        """Dummy predict_proba."""

    def predict(self, *args: Any):
        """Dummy predict."""


class ArrayOutputModel:

    def __init__(self):
        self.trained_ = True

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict_proba(self, X: NDArray, *args: Any) -> NDArray:
        probas = np.array([[.9, .05, .05]])
        proba_out = np.repeat(probas, len(X), axis=0)
        return proba_out

    def predict(self, X: NDArray, *args: Any) -> NDArray:
        return self.predict_proba(X) >= .3

    def __sklearn_is_fitted__(self):
        return True


X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.stack(
    [
        [1, 0, 1], [1, 0, 0], [0, 1, 1],
        [0, 1, 0], [0, 0, 1], [1, 1, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ]
)

X, y = make_multilabel_classification(
    n_samples=1000,
    n_classes=5,
    random_state=random_state,
    allow_unlabeled=False
)

X_no_label, y_no_label = make_multilabel_classification(
    n_samples=1000,
    n_classes=5,
    random_state=random_state,
    allow_unlabeled=True
)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    PrecisionRecallController()


def test_valid_estimator() -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        estimator=clf,
        random_state=random_state
    )
    mapie_clf.fit(X_toy, y_toy)
    assert isinstance(mapie_clf.single_estimator_, MultiOutputClassifier)


def test_valid_method() -> None:
    """Test that valid methods raise no errors."""
    mapie_clf = PrecisionRecallController(random_state=random_state)
    mapie_clf.fit(X_toy, y_toy)
    check_is_fitted(mapie_clf, mapie_clf.fit_attributes)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_metric_method(strategy: str) -> None:
    """Test that valid metric raise no errors"""
    args = STRATEGIES[strategy][0]
    mapie_clf = PrecisionRecallController(
        random_state=random_state,
        metric_control=args["metric_control"]
    )
    mapie_clf.fit(X_toy, y_toy)
    check_is_fitted(mapie_clf, mapie_clf.fit_attributes)


@pytest.mark.parametrize("bound", BOUNDS)
def test_valid_bound(bound: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_clf = PrecisionRecallController(
        random_state=random_state, method="rcps"
    )
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(X_toy, bound=bound, delta=.1)
    check_is_fitted(mapie_clf, mapie_clf.fit_attributes)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [0.2, [0.2, 0.3], (0.2, 0.3)])
@pytest.mark.parametrize("delta", [0.2, 0.1, 0.05])
def test_predict_output_shape(
    strategy: str, alpha: Any, delta: Any
) -> None:
    """Test predict output shape."""
    args = STRATEGIES[strategy][0]
    mapie_clf = PrecisionRecallController(
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"]
    )
    mapie_clf.fit(X, y)
    y_pred, y_ps = mapie_clf.predict(
        X,
        alpha=alpha,
        bound=args["bound"],
        delta=delta
    )
    n_alpha = len(alpha) if hasattr(alpha, "__len__") else 1
    assert y_pred.shape == y.shape
    assert y_ps.shape == (y.shape[0], y.shape[1], n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args = STRATEGIES[strategy][0]
    mapie_clf = PrecisionRecallController(
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"]
    )
    mapie_clf.fit(X, y)
    _, y_ps = mapie_clf.predict(
        X,
        alpha=[0.1, 0.1],
        bound=args["bound"],
        delta=.1
    )
    np.testing.assert_allclose(y_ps[:, 0, 0], y_ps[:, 0, 1])
    np.testing.assert_allclose(y_ps[:, 1, 0], y_ps[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_partial_fit(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args = STRATEGIES[strategy][0]
    clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    mapie_clf = PrecisionRecallController(
        estimator=clf,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"]
    )
    mapie_clf.fit(X, y)

    mapie_clf_partial = PrecisionRecallController(
        estimator=clf,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"]
    )
    for i in range(len(X)):
        mapie_clf_partial.partial_fit(
            X[i][np.newaxis, :],
            y[i][np.newaxis, :]
        )

    y_pred, y_ps = mapie_clf.predict(
        X,
        alpha=[0.1, 0.1],
        bound=args["bound"],
        delta=.1
    )

    y_pred_partial, y_ps_partial = mapie_clf_partial.predict(
        X,
        alpha=[0.1, 0.1],
        bound=args["bound"],
        delta=.1
    )
    np.testing.assert_allclose(y_pred, y_pred_partial)
    np.testing.assert_allclose(y_ps, y_ps_partial)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize(
    "alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)]
)
def test_results_for_alpha_as_float_and_arraylike(
    strategy: str, alpha: Any
) -> None:
    """Test that output values do not depend on type of alpha."""
    args = STRATEGIES[strategy][0]
    mapie_clf = PrecisionRecallController(
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"]
    )
    mapie_clf.fit(X, y)
    y_pred_float1, y_ps_float1 = mapie_clf.predict(
        X,
        alpha=alpha[0],
        bound=args["bound"],
        delta=.9
    )
    y_pred_float2, y_ps_float2 = mapie_clf.predict(
        X,
        alpha=alpha[1],
        bound=args["bound"],
        delta=.9
    )
    y_pred_array, y_ps_array = mapie_clf.predict(
        X,
        alpha=alpha,
        bound=args["bound"],
        delta=.9
    )
    np.testing.assert_allclose(y_pred_float1, y_pred_array)
    np.testing.assert_allclose(y_pred_float2, y_pred_array)
    np.testing.assert_allclose(y_ps_float1[:, :, 0], y_ps_array[:, :, 0])
    np.testing.assert_allclose(y_ps_float2[:, :, 0], y_ps_array[:, :, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_single_and_multi_jobs(strategy: str) -> None:
    """
    Test that _MapieClassifier gives equal predictions
    regardless of number of parallel jobs.
    """
    args = STRATEGIES[strategy][0]
    mapie_clf_single = PrecisionRecallController(
        n_jobs=1,
        metric_control=args["metric_control"],
        random_state=args["random_state"]
    )
    mapie_clf_multi = PrecisionRecallController(
        n_jobs=-1,
        metric_control=args["metric_control"],
        random_state=args["random_state"]
    )
    mapie_clf_single.fit(X, y)
    mapie_clf_multi.fit(X, y)
    y_pred_single, y_ps_single = mapie_clf_single.predict(
        X,
        alpha=0.2,
        bound=args["bound"],
        delta=.9
    )
    y_pred_multi, y_ps_multi = mapie_clf_multi.predict(
        X,
        alpha=0.2,
        bound=args["bound"],
        delta=.9
    )
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_ps_single, y_ps_multi)


@pytest.mark.parametrize(
    "alpha", [[0.2, 0.8], (0.2, 0.8), np.array([0.2, 0.8]), None],
)
@pytest.mark.parametrize(
    "delta", [.1, .2, .5, .9, .001],
)
@pytest.mark.parametrize(
    "bound", BOUNDS,
)
def test_valid_prediction(alpha: Any, delta: Any, bound: Any) -> None:
    """Test fit and predict."""
    model = MultiOutputClassifier(
        LogisticRegression()
    )
    model.fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
      estimator=model, method="rcps",
      random_state=random_state
    )

    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(
        X_toy,
        alpha=alpha,
        bound=bound,
        delta=delta
    )


@pytest.mark.parametrize(
    "alpha", [[0.2, 0.8], (0.2, 0.8), np.array([0.2, 0.8]), None],
)
@pytest.mark.parametrize(
    "delta", [.1, .2, .5, .9, .001],
)
@pytest.mark.parametrize(
    "bound", BOUNDS,
)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_array_output_model(strategy: str, alpha: Any, delta: Any, bound: Any):
    args = STRATEGIES[strategy][0]
    model = ArrayOutputModel()
    mapie_clf = PrecisionRecallController(
        estimator=model,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=random_state
    )
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.predict(
        X_toy,
        alpha=alpha,
        bound=bound,
        delta=delta
    )


def test_reinit_new_fit():
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        estimator=clf, random_state=random_state
    )
    mapie_clf.fit(X_toy, y_toy)
    mapie_clf.fit(X_toy, y_toy)
    assert len(mapie_clf.risks) == len(X_toy)


@pytest.mark.parametrize("method", WRONG_METHODS)
def test_method_error_in_fit(method: str) -> None:
    """Test error for wrong method"""
    mapie_clf = PrecisionRecallController(
      random_state=random_state, method=method
    )

    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        mapie_clf.fit(X_toy, y_toy)


def test_method_error_if_no_label_fit() -> None:
    """Test error for wrong method"""
    mapie_clf = PrecisionRecallController(random_state=random_state)
    with pytest.raises(ValueError, match=r".*Invalid y.*"):
        mapie_clf.fit(X_no_label, y_no_label)


def test_method_error_if_no_label_partial_fit() -> None:
    """Test error for wrong method"""
    clf = MultiOutputClassifier(LogisticRegression()).fit(
        X_no_label,
        y_no_label
    )
    mapie_clf = PrecisionRecallController(
        estimator=clf, random_state=random_state
    )
    with pytest.raises(ValueError, match=r".*Invalid y.*"):
        mapie_clf.partial_fit(X_no_label, y_no_label)


@pytest.mark.parametrize("bound", WRONG_BOUNDS)
def test_bound_error_in_predict(bound: str) -> None:
    """Test error for wrong bounds"""
    mapie_clf = PrecisionRecallController(
      random_state=random_state, method='rcps'
    )

    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*bound must be in.*"):
        mapie_clf.predict(X_toy, bound=bound, delta=.1)


@pytest.mark.parametrize("metric_control", WRONG_METRICS)
def test_metric_error_in_fit(metric_control: str) -> None:
    """Test error for wrong metrics"""
    mapie_clf = PrecisionRecallController(
        random_state=random_state,
        metric_control=metric_control
    )
    with pytest.raises(ValueError, match=r".*Invalid metric. *"):
        mapie_clf.fit(X_toy, y_toy)


def test_error_rcps_delta_null() -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = PrecisionRecallController(
      random_state=random_state, method='rcps'
    )

    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta cannot be ``None``*"):
        mapie_clf.predict(X_toy)


def test_error_ltt_delta_null() -> None:
    """Test error for LTT method and delta None"""
    mapie_clf = PrecisionRecallController(
        random_state=random_state,
        metric_control='precision'
    )
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Invalid delta. *"):
        mapie_clf.predict(X_toy)


@pytest.mark.parametrize("delta", [-1., 0, 1, 4, -3])
def test_error_delta_wrong_value(delta: Any) -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = PrecisionRecallController(
      random_state=random_state, method='rcps'
    )
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta must be*"):
        mapie_clf.predict(X_toy, delta=delta)


@pytest.mark.parametrize("delta", [-1., 0, 1, 4, -3])
def test_error_delta_wrong_value_ltt(delta: Any) -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = PrecisionRecallController(
        random_state=random_state,
        metric_control='precision'
    )

    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta must be*"):
        mapie_clf.predict(X_toy, delta=delta)


def test_bound_none_crc() -> None:
    """Test that a warning is raised when bound is not None with CRC method."""
    mapie_clf = PrecisionRecallController(
      random_state=random_state, method="crc"
    )

    mapie_clf.fit(X_toy, y_toy)
    with pytest.warns(UserWarning, match=r"WARNING: you are using crc*"):
        mapie_clf.predict(X_toy, bound="wsr")


def test_delta_none_crc() -> None:
    """Test that a warning is raised nound is not None with CRC method."""
    mapie_clf = PrecisionRecallController(
      random_state=random_state, method="crc"
    )
    mapie_clf.fit(X_toy, y_toy)
    with pytest.warns(UserWarning, match=r"WARNING: you are using crc*"):
        mapie_clf.predict(X_toy, bound=None, delta=.1)


def test_warning_estimator_none() -> None:
    """Test that a warning is raised nound is not None with CRC method."""
    mapie_clf = PrecisionRecallController(random_state=random_state)
    with pytest.warns(UserWarning, match=r"WARNING: To avoid overfitting,*"):
        mapie_clf.fit(X_toy, y_toy)


@pytest.mark.parametrize("delta", [np.arange(0, 1, 0.01), (.1, .2), [.4, .5]])
def test_error_delta_wrong_type(delta: Any) -> None:
    """Test error for RCPS method and delta None"""
    mapie_clf = PrecisionRecallController(
      random_state=random_state, method="rcps"
    )
    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta must be a float*"):
        mapie_clf.predict(X_toy, delta=delta)


@pytest.mark.parametrize("delta", [np.arange(0, 1, 0.01), (.1, .2), [.4, .5]])
def test_error_delta_wrong_type_ltt(delta: Any) -> None:
    """Test error for LTT method and delta None"""
    mapie_clf = PrecisionRecallController(
        random_state=random_state,
        metric_control="precision"
    )

    mapie_clf.fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*delta must be a float*"):
        mapie_clf.predict(X_toy, delta=delta)


def test_error_partial_fit_different_size() -> None:
    """Test error for partial_fit with different size"""
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        estimator=clf, random_state=random_state
    )
    mapie_clf.partial_fit(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*Number of features*"):
        mapie_clf.partial_fit(X, y)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_pipeline_compatibility(strategy: str) -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    args = STRATEGIES[strategy][0]
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5],
        }
    )
    y = np.array(
        [
            [0, 0, 1], [0, 0, 1],
            [1, 1, 0], [1, 0, 1],
            [1, 0, 1], [1, 1, 1]
        ]
    )
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[
            ("encoding", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_preprocessor, ["x_cat"]),
            ("num", numeric_preprocessor, ["x_num"])
        ]
    )
    pipe = make_pipeline(
        preprocessor,
        MultiOutputClassifier(LogisticRegression())
    )
    pipe.fit(X, y)
    mapie = PrecisionRecallController(
        estimator=pipe,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=random_state
    )

    mapie.fit(X, y)
    mapie.predict(X, bound=args["bound"], delta=.1)


def test_error_no_fit() -> None:
    """Test error for no fit"""
    clf = WrongOutputModel()
    mapie_clf = PrecisionRecallController(
        estimator=clf, random_state=random_state
    )
    with pytest.raises(
        ValueError,
        match=r".*Please provide a classifier with*"
    ):
        mapie_clf.fit(X_toy, y_toy)


def test_error_estimator_none_partial() -> None:
    """Test error estimator none partial"""
    mapie_clf = PrecisionRecallController(random_state=random_state)
    with pytest.raises(
        ValueError,
        match=r".*Invalid estimator with partial_fit*"
    ):
        mapie_clf.partial_fit(X_toy, y_toy)


def test_partial_fit_first_time():
    mclf = PrecisionRecallController(random_state=random_state)
    assert mclf._check_partial_fit_first_call()


def test_partial_fit_second_time():
    clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    mclf = PrecisionRecallController(
        estimator=clf, random_state=random_state
    )
    mclf.partial_fit(X, y)
    assert not mclf._check_partial_fit_first_call()


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_toy_dataset_predictions(strategy: str) -> None:
    """
    Test toy_dataset_predictions.
    """
    args = STRATEGIES[strategy][0]
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        clf,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=random_state
    )
    mapie_clf.fit(X_toy, y_toy)
    _, y_ps = mapie_clf.predict(
        X_toy,
        alpha=.2,
        bound=args["bound"],
        delta=.1
    )
    np.testing.assert_allclose(
            y_ps[:, :, 0],
            y_toy_mapie[strategy],
            rtol=1e-6
        )


@pytest.mark.parametrize("method", ["rcps", "crc"])
def test_error_wrong_method_metric_precision(method: str) -> None:
    """
    Test that an error is returned when using a metric
    with invalid method .
    """
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        clf,
        method=method,
        metric_control="precision"
    )
    with pytest.raises(
        ValueError,
        match=r".*Invalid method for metric*"
    ):
        mapie_clf.fit(X_toy, y_toy)


@pytest.mark.parametrize("method", ["ltt"])
def test_check_metric_control(method: str) -> None:
    """
    Test that an error is returned when using a metric
    with invalid method .
    """
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        clf,
        method=method,
        metric_control="recall"
    )
    with pytest.raises(
        ValueError,
        match=r".*Invalid method for metric*"
    ):
        mapie_clf.fit(X_toy, y_toy)


def test_method_none_precision() -> None:
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        clf,
        metric_control="precision"
    )
    mapie_clf.fit(X_toy, y_toy)
    assert mapie_clf.method == "ltt"


def test_method_none_recall() -> None:
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = PrecisionRecallController(
        clf,
        metric_control="recall"
    )
    mapie_clf.fit(X_toy, y_toy)
    assert mapie_clf.method == "crc"


def fpr_func(y_true: NDArray, y_pred: NDArray) -> float:
    """Computes false positive rate."""
    tn: int = np.sum((y_true == 0) & (y_pred == 0))
    fp: int = np.sum((y_true == 0) & (y_pred == 1))
    return fp / (tn + fp)


# The following test is voluntarily agnostic
# to the specific binary classification risk control implementation.
@pytest.mark.parametrize(
    "risk_instance, metric_func, effective_sample_func",
    [
        (precision, precision_score, lambda y_true, y_pred: np.sum(y_pred == 1)),
        (recall, recall_score, lambda y_true, y_pred: np.sum(y_true == 1)),
        (false_positive_rate, fpr_func, lambda y_true, y_pred: np.sum(y_true == 0)),
    ],
)
@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])),
        (np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1])),
    ],
)
def test_binary_classification_risk(
    risk_instance: BinaryClassificationRisk,
    metric_func,
    effective_sample_func,
    y_true,
    y_pred
):
    value, n = risk_instance.get_value_and_effective_sample_size(y_true, y_pred)
    effective_sample_size = effective_sample_func(y_true, y_pred)

    if effective_sample_size != 0:
        expected_value = metric_func(y_true, y_pred)
        expected_n = effective_sample_size
    else:
        expected_value = 1
        expected_n = -1
    if risk_instance.higher_is_better:
        expected_value = 1 - expected_value
    assert np.isclose(value, expected_value)
    assert n == expected_n


class TestBinaryClassificationControllerBestPredictParamChoice:
    @pytest.mark.parametrize(
        "risk_instance, expected",
        [
            (precision, recall),
            (recall, precision),
            (accuracy, accuracy),
            (false_positive_rate, recall),
        ],
    )
    def test_auto(
        self,
        risk_instance: BinaryClassificationRisk,
        expected
    ):
        controller = BinaryClassificationController(
            predict_function=lambda X: np.random.rand(1, 2),
            risk=risk_instance,
            target_level=0.8,
            best_predict_param_choice="auto"
        )

        result = controller._best_predict_param_choice
        assert result is expected

    def test_custom(self):
        """Test _set_best_predict_param_choice with a custom risk instance."""
        custom_risk = accuracy

        controller = BinaryClassificationController(
            predict_function=lambda X: np.random.rand(1, 2),
            risk=precision,
            target_level=0.8,
            best_predict_param_choice=custom_risk
        )

        result = controller._set_best_predict_param_choice(custom_risk)
        assert result is custom_risk

    def test_auto_unknown_risk(self):
        """Test _set_best_predict_param_choice with 'auto' mode for unknown risk."""
        unknown_risk = deepcopy(accuracy)

        with pytest.raises(ValueError):
            BinaryClassificationController(
                predict_function=lambda X: np.random.rand(1, 2),
                risk=unknown_risk,
                target_level=0.8,
                best_predict_param_choice="auto"
            )


@pytest.mark.parametrize(
    "risk_instance,target_level,expected_alpha",
    [
        (recall, 0.6, 0.4),  # higher_is_better=True
        (false_positive_rate, 0.6, 0.6),  # higher_is_better=False
    ],
)
def test_binary_classification_controller_alpha(
    risk_instance: BinaryClassificationRisk,
    target_level: float,
    expected_alpha: float,
) -> None:
    controller = BinaryClassificationController(
        predict_function=lambda X: np.random.rand(1, 2),
        risk=risk_instance,
        target_level=target_level,
    )
    assert np.isclose(controller._alpha, expected_alpha)


def test_binary_classification_controller_sklearn_pipeline_with_dataframe() -> None:
    X_df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([1, 1, 0, 1], dtype=int)

    pipe = Pipeline(steps=[("clf", DummyClassifier(random_state=random_state))])
    pipe.fit(X_df, y)

    controller = BinaryClassificationController(
        predict_function=pipe.predict_proba,
        risk=precision,
        target_level=0.1,
        confidence_level=0.1,
    )

    controller.calibrate(X_df, y)
    controller.predict(X_df)


def test_set_risk_not_controlled():
    controller = BinaryClassificationController(
        predict_function=lambda X: np.random.rand(1, 2),
        risk=precision,
        target_level=0.9,
    )
    with pytest.warns(
        UserWarning,
        match=r"No predict parameters were found to control the risk"
    ):
        controller._set_risk_not_controlled()
    assert controller.best_predict_param is None


class TestBinaryClassificationControllerSetBestPredictParam:
    @pytest.mark.parametrize("best_predict_param_choice", ["auto", precision, recall])
    def test_only_one_param(self, best_predict_param_choice):
        """
        Expected: should always set this param
        """
        controller = BinaryClassificationController(
            predict_function=lambda X: np.random.rand(1, 2),
            risk=precision,
            target_level=0.9,
            best_predict_param_choice=best_predict_param_choice
        )

        dummy_param = 0.5
        y_calibrate = np.array([1, 0])
        dummy_predictions = np.array([[True, False]])
        valid_params_index = [0]
        controller.valid_predict_params = np.array([dummy_param])

        controller._set_best_predict_param(
            y_calibrate_=y_calibrate,
            predictions_per_param=dummy_predictions,
            valid_params_index=valid_params_index
        )

        assert controller.best_predict_param == dummy_param

    @pytest.mark.parametrize(
        "best_predict_param_choice, expected",
        [[precision, 0.5], [recall, 0.7]]
    )
    def test_correct_param_out_of_two(self, best_predict_param_choice, expected):
        dummy_param = 0.5
        dummy_param_2 = 0.7

        controller = BinaryClassificationController(
            predict_function=lambda X: np.random.rand(1, 2),
            risk=precision,
            target_level=0.9,
            best_predict_param_choice=best_predict_param_choice
        )

        y_calibrate = np.array([1, 1, 0])
        predictions_per_param = np.array(
            [
                [True, False, False],
                [True, True, True]
            ]
        )
        valid_params_index = [0, 1]

        controller.valid_predict_params = np.array(
            [dummy_param, dummy_param_2]
        )

        controller._set_best_predict_param(
            y_calibrate_=y_calibrate,
            predictions_per_param=predictions_per_param,
            valid_params_index=valid_params_index
        )

        assert controller.best_predict_param == expected

    def test_secondary_risk_undefined(self):
        """
        Expected: should set the param even though precision is not defined
        """
        controller = BinaryClassificationController(
            predict_function=lambda X: np.random.rand(1, 2),
            risk=precision,
            target_level=0.9,
            best_predict_param_choice=precision
        )

        y_calibrate = np.array([1, 0])
        predictions_per_param = np.array([[False, False]])  # precision undefined
        valid_params_index = [0]
        dummy_param = 0.5
        controller.valid_predict_params = np.array([dummy_param])

        controller._set_best_predict_param(
            y_calibrate_=y_calibrate,
            predictions_per_param=predictions_per_param,
            valid_params_index=valid_params_index
        )
        assert controller.best_predict_param == dummy_param


def deterministic_predict_function(X):
    probs1 = np.array([0.2, 0.5, 0.9])
    probs0 = 1.0 - probs1
    return np.stack([probs0, probs1], axis=1)


@pytest.fixture
def bcc_deterministic():
    return BinaryClassificationController(
        predict_function=deterministic_predict_function,
        risk=precision,
        target_level=0.9,
    )


class TestBinaryClassificationControllerGetPredictionsPerParam:
    def test_single_parameter(self, bcc_deterministic):
        result = bcc_deterministic._get_predictions_per_param(
            X=[],
            params=np.array([0.5])
        )

        expected = np.array([[False, True, True]])
        assert result.shape == (1, 3)
        assert result.dtype == int
        np.testing.assert_array_equal(result, expected)

    def test_multiple_parameters(self, bcc_deterministic):
        result = bcc_deterministic._get_predictions_per_param(
            X=[],
            params=np.array([0.0, 0.5, 0.8])
        )

        expected = np.array([
            [True, True, True],
            [False, True, True],
            [False, False, True],
        ])
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape_consistency(self):
        def predict_fn(X):
            return np.array([[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])

        controller = BinaryClassificationController(
            predict_function=predict_fn,
            risk=precision,
            target_level=0.9,
        )

        params = np.array([0.2, 0.5, 0.8])
        result = controller._get_predictions_per_param(X=[], params=params)

        assert result.shape == (len(params), 3)

    def test_error_passing_classifier(self):
        """
        Test when the user provides a classifier instead of a predict_proba
        method
        """
        clf = LogisticRegression().fit([[0], [1]], [0, 1])
        bcc = BinaryClassificationController(
            predict_function=clf,
            risk=precision,
            target_level=0.9
        )
        X_test = [[0]]
        params = np.array([0.5])

        with pytest.raises(
            TypeError,
            match=r"Maybe you provided a binary classifier"
        ):
            bcc._get_predictions_per_param(X_test, params)

    def test_error_incorrect_predict_shape(self):
        """
        Test when the user provides a predict function that outputs only
        the positive class.
        """
        clf = LogisticRegression().fit([[0], [1]], [0, 1])

        def pred_func(X):
            return clf.predict_proba(X)[:, 0]

        bcc = BinaryClassificationController(
            predict_function=pred_func,
            risk=precision,
            target_level=0.9
        )
        X_test = [[0]]
        params = np.array([0.5])

        with pytest.raises(
            IndexError,
            match=r"Maybe the predict function you provided returns only the "
                  r"probability of the positive class."
        ):
            bcc._get_predictions_per_param(X_test, params)

    @pytest.mark.parametrize(
        "error,expected_error_type,expected_error_message",
        [
            (ValueError("Hey"), ValueError, "Hey"),
            (IndexError("Gloups"), IndexError, "Gloups"),
            (TypeError("I'm hungry"), TypeError, "I'm hungry"),
        ],
    )
    def test_other_error(self, error, expected_error_type, expected_error_message):
        """Test that other errors are re-raised without modification"""

        def failing_predict_function(X):
            raise error

        bcc = BinaryClassificationController(
            predict_function=failing_predict_function,
            risk=precision,
            target_level=0.9
        )

        X_test = [[0]]
        params = np.array([0.5])

        with pytest.raises(expected_error_type, match=expected_error_message):
            bcc._get_predictions_per_param(X_test, params)


class TestBinaryClassificationControllerPredict:
    def test_output_shape(self, bcc_deterministic):
        controller = bcc_deterministic
        controller.best_predict_param = 0.5
        predictions = controller.predict([])

        assert predictions.shape == (3,)
        assert predictions.dtype == int

    def test_error(self, bcc_deterministic):
        controller = bcc_deterministic
        controller.best_predict_param = None

        with pytest.raises(
            ValueError,
            match=r"Cannot predict"
        ):
            controller.predict(X)
