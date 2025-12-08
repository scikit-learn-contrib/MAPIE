from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from numpy.typing import ArrayLike, NDArray
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import TypedDict

from mapie.risk_control import MultiLabelClassificationController
from mapie.utils import check_is_fitted

Params = TypedDict(
    "Params",
    {
        "method": str,
        "rcps_bound": Optional[str],
        "random_state": Optional[int],
        "metric_control": Optional[str],
    },
)

BOUNDS = ["wsr", "hoeffding", "bernstein"]
random_state = 42

WRONG_METHODS = ["rpcs", "rcr", "test", "llt"]
WRONG_BOUNDS = ["wrs", "hoeff", "test", "", 1, 2.5, (1, 2)]
WRONG_METRICS = ["presicion", "recal", ""]


STRATEGIES = {
    "crc": (
        Params(
            method="crc",
            rcps_bound=None,
            random_state=random_state,
            metric_control="recall",
        ),
    ),
    "rcps_wsr": (
        Params(
            method="rcps",
            rcps_bound="wsr",
            random_state=random_state,
            metric_control="recall",
        ),
    ),
    "rcps_hoeffding": (
        Params(
            method="rcps",
            rcps_bound="hoeffding",
            random_state=random_state,
            metric_control="recall",
        ),
    ),
    "rcps_bernstein": (
        Params(
            method="rcps",
            rcps_bound="bernstein",
            random_state=random_state,
            metric_control="recall",
        ),
    ),
    "ltt": (
        Params(
            method="ltt",
            rcps_bound=None,
            random_state=random_state,
            metric_control="precision",
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
        [False, True, True],
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
    ],
}


class ArrayOutputModel:
    def __init__(self):
        self.trained_ = True

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict_proba(self, X: ArrayLike) -> NDArray:
        X = np.asarray(X)  # fix mypy issue for minimum supported requirements
        probas = np.array([[0.9, 0.05, 0.05]])
        proba_out = np.repeat(probas, len(X), axis=0)
        return proba_out


X_toy = np.arange(9).reshape(-1, 1)
y_toy = np.stack(
    [
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]
)

X, y = make_multilabel_classification(
    n_samples=1000, n_classes=5, random_state=random_state, allow_unlabeled=False
)

X_no_label, y_no_label = make_multilabel_classification(
    n_samples=1000, n_classes=5, random_state=random_state, allow_unlabeled=True
)

toy_estimator = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, random_state=random_state)
).fit(X_toy, y_toy)
toy_predict_function = toy_estimator.predict_proba

multilabel_estimator = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, random_state=random_state)
).fit(X, y)
multilabel_predict_function = multilabel_estimator.predict_proba

toy_estimator = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, random_state=random_state)
).fit(X_toy, y_toy)
toy_predict_function = toy_estimator.predict_proba

multilabel_estimator = MultiOutputClassifier(
    LogisticRegression(max_iter=1000, random_state=random_state)
).fit(X, y)
multilabel_predict_function = multilabel_estimator.predict_proba


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MultiLabelClassificationController(predict_function=toy_predict_function)


def test_valid_method() -> None:
    """Test that valid methods raise no errors."""
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function, random_state=random_state
    )
    mapie_clf.calibrate(X_toy, y_toy)
    check_is_fitted(mapie_clf)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_metric_method(strategy: str) -> None:
    """Test that valid metric raise no errors"""
    args = STRATEGIES[strategy][0]
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function,
        random_state=random_state,
        metric_control=args["metric_control"],
        confidence_level=0.9,
    )
    mapie_clf.calibrate(X_toy, y_toy)
    check_is_fitted(mapie_clf)


@pytest.mark.parametrize("bound", BOUNDS)
def test_valid_bound(bound: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function,
        random_state=random_state,
        method="rcps",
        rcps_bound=bound,
        confidence_level=0.9,
    )
    mapie_clf.calibrate(X_toy, y_toy)
    mapie_clf.predict(X_toy)
    check_is_fitted(mapie_clf)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("target_level", [0.8, [0.8, 0.7], (0.8, 0.7)])
@pytest.mark.parametrize("confidence_level", [0.8, 0.9, 0.95])
def test_predict_output_shape(
    strategy: str, target_level: Any, confidence_level: Any
) -> None:
    """Test predict output shape."""
    args = STRATEGIES[strategy][0]
    mapie_clf = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=target_level,
        confidence_level=confidence_level,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf.calibrate(X, y)
    y_pred, y_ps = mapie_clf.predict(X)
    n_alpha = len(target_level) if hasattr(target_level, "__len__") else 1
    assert y_pred.shape == y.shape
    assert y_ps.shape == (y.shape[0], y.shape[1], n_alpha)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_same_alpha(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args = STRATEGIES[strategy][0]
    mapie_clf = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=[0.9, 0.9],
        confidence_level=0.9,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf.calibrate(X, y)
    _, y_ps = mapie_clf.predict(X)
    np.testing.assert_allclose(y_ps[:, 0, 0], y_ps[:, 0, 1])
    np.testing.assert_allclose(y_ps[:, 1, 0], y_ps[:, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_for_partial_calibrate(strategy: str) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    args = STRATEGIES[strategy][0]
    mapie_clf = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=[0.9, 0.9],
        confidence_level=0.9,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf.calibrate(X, y)

    mapie_clf_partial = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=[0.9, 0.9],
        confidence_level=0.9,
        rcps_bound=args["rcps_bound"],
    )
    for i in range(len(X)):
        mapie_clf_partial.partial_calibrate(X[i][np.newaxis, :], y[i][np.newaxis, :])

    y_pred, y_ps = mapie_clf.predict(X)

    y_pred_partial, y_ps_partial = mapie_clf_partial.predict(X)
    np.testing.assert_allclose(y_pred, y_pred_partial)
    np.testing.assert_allclose(y_ps, y_ps_partial)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("alpha", [np.array([0.05, 0.1]), [0.05, 0.1], (0.05, 0.1)])
def test_results_for_alpha_as_float_and_arraylike(strategy: str, alpha: Any) -> None:
    """Test that output values do not depend on type of alpha."""
    args = STRATEGIES[strategy][0]
    mapie_clf = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=alpha[0],
        confidence_level=0.1,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf.calibrate(X, y)
    y_pred_float1, y_ps_float1 = mapie_clf.predict(X)

    mapie_clf = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=alpha[1],
        confidence_level=0.1,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf.calibrate(X, y)
    y_pred_float2, y_ps_float2 = mapie_clf.predict(X)

    mapie_clf = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=alpha,
        confidence_level=0.1,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf.calibrate(X, y)
    y_pred_array, y_ps_array = mapie_clf.predict(X)

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
    mapie_clf_single = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        n_jobs=1,
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=0.8,
        confidence_level=0.1,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf_multi = MultiLabelClassificationController(
        predict_function=multilabel_predict_function,
        n_jobs=-1,
        metric_control=args["metric_control"],
        random_state=args["random_state"],
        target_level=0.8,
        confidence_level=0.1,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf_single.calibrate(X, y)
    mapie_clf_multi.calibrate(X, y)
    y_pred_single, y_ps_single = mapie_clf_single.predict(X)
    y_pred_multi, y_ps_multi = mapie_clf_multi.predict(X)
    np.testing.assert_allclose(y_pred_single, y_pred_multi)
    np.testing.assert_allclose(y_ps_single, y_ps_multi)


@pytest.mark.parametrize(
    "target_level",
    [[0.8, 0.2], (0.8, 0.2), np.array([0.8, 0.2])],
)
@pytest.mark.parametrize(
    "confidence_level",
    [0.9, 0.8, 0.5, 0.1, 0.999],
)
@pytest.mark.parametrize(
    "bound",
    BOUNDS,
)
def test_valid_prediction(target_level: Any, confidence_level: Any, bound: Any) -> None:
    """Test fit and predict."""
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function,
        method="rcps",
        random_state=random_state,
        target_level=target_level,
        confidence_level=confidence_level,
        rcps_bound=bound,
    )

    mapie_clf.calibrate(X_toy, y_toy)
    mapie_clf.predict(X_toy)


@pytest.mark.parametrize(
    "target_level",
    [[0.8, 0.2], (0.8, 0.2), np.array([0.8, 0.2])],
)
@pytest.mark.parametrize(
    "confidence_level",
    [0.9, 0.8, 0.5, 0.1, 0.999],
)
@pytest.mark.parametrize(
    "bound",
    BOUNDS,
)
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_array_output_model(
    strategy: str, target_level: Any, confidence_level: Any, bound: Any
):
    args = STRATEGIES[strategy][0]
    model = ArrayOutputModel()
    mapie_clf = MultiLabelClassificationController(
        predict_function=model.predict_proba,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=random_state,
        target_level=target_level,
        confidence_level=confidence_level,
        rcps_bound=bound if args["method"] == "rcps" else args["rcps_bound"],
    )
    mapie_clf.calibrate(X_toy, y_toy)
    mapie_clf.predict(X_toy)


def test_reinit_new_fit():
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function, random_state=random_state
    )
    mapie_clf.calibrate(X_toy, y_toy)
    mapie_clf.calibrate(X_toy, y_toy)
    assert len(mapie_clf.risks) == len(X_toy)


@pytest.mark.parametrize("method", WRONG_METHODS)
def test_method_error_in_init(method: str) -> None:
    """Test error for wrong method"""
    with pytest.raises(ValueError, match=r".*Invalid method.*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            method=method,
        )


def test_method_error_if_no_label_fit() -> None:
    """Test error for wrong method"""
    mapie_clf = MultiLabelClassificationController(
        predict_function=multilabel_predict_function, random_state=random_state
    )
    with pytest.raises(ValueError, match=r".*Invalid y.*"):
        mapie_clf.calibrate(X_no_label, y_no_label)


def test_method_error_if_no_label_partial_calibrate() -> None:
    """Test error for wrong method"""
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_no_label, y_no_label)
    mapie_clf = MultiLabelClassificationController(
        predict_function=clf.predict_proba, random_state=random_state
    )
    with pytest.raises(ValueError, match=r".*Invalid y.*"):
        mapie_clf.partial_calibrate(X_no_label, y_no_label)


@pytest.mark.parametrize("bound", WRONG_BOUNDS)
def test_bound_error(bound: str) -> None:
    """Test error for wrong bounds"""
    with pytest.raises(ValueError, match=r".*bound must be in.*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            method="rcps",
            rcps_bound=bound,
            confidence_level=0.9,
        )


@pytest.mark.parametrize("metric_control", WRONG_METRICS)
def test_metric_error_in_init(metric_control: str) -> None:
    """Test error for wrong metrics"""
    with pytest.raises(ValueError, match=r".*Invalid metric. *"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            metric_control=metric_control,
        )


def test_error_rcps_confidence_level_null() -> None:
    """Test error for RCPS method and confidence_level None"""
    with pytest.raises(ValueError, match=r".*confidence_level cannot be ``None``*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            method="rcps",
            confidence_level=None,
        )


def test_error_ltt_confidence_level_null() -> None:
    """Test error for LTT method and confidence_level None"""
    with pytest.raises(ValueError, match=r".*confidence_level cannot be ``None``*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            metric_control="precision",
            confidence_level=None,
        )


@pytest.mark.parametrize("confidence_level", [-1.0, 0, 1, 4, -3])
def test_error_confidence_level_wrong_value(confidence_level: Any) -> None:
    """Test error for RCPS method and confidence_level wrong value"""
    with pytest.raises(ValueError, match=r".*confidence_level must be*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            method="rcps",
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize("confidence_level", [-1.0, 0, 1, 4, -3])
def test_error_confidence_level_wrong_value_ltt(confidence_level: Any) -> None:
    """Test error for LTT method and confidence_level wrong value"""
    with pytest.raises(ValueError, match=r".*confidence_level must be*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            metric_control="precision",
            confidence_level=confidence_level,
        )


def test_bound_none_crc() -> None:
    """Test that a warning is raised when bound is not None with CRC method."""
    with pytest.warns(UserWarning, match=r"WARNING: you are using crc*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            method="crc",
            rcps_bound="wsr",
        )


def test_confidence_level_none_crc() -> None:
    """Test that a warning is raised when confidence_level is not none with CRC method."""
    with pytest.warns(UserWarning, match=r"WARNING: you are using crc*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            method="crc",
            confidence_level=0.9,
        )


@pytest.mark.parametrize(
    "confidence_level", [np.arange(0, 1, 0.99), (0.9, 0.8), [0.6, 0.5]]
)
def test_error_confidence_level_wrong_type(confidence_level: Any) -> None:
    """Test error for RCPS method and confidence_level wrong type"""

    with pytest.raises(ValueError, match=r".*confidence_level must be a float*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            method="rcps",
            confidence_level=confidence_level,
        )


@pytest.mark.parametrize(
    "confidence_level", [np.arange(0, 1, 0.01), (0.1, 0.2), [0.4, 0.5]]
)
def test_error_confidence_level_wrong_type_ltt(confidence_level: Any) -> None:
    """Test error for LTT method and confidence_level wrong type"""
    with pytest.raises(ValueError, match=r".*confidence_level must be a float*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            random_state=random_state,
            metric_control="precision",
            confidence_level=confidence_level,
        )


def test_error_partial_calibrate_different_size() -> None:
    """Test error for partial_calibrate with different size"""
    clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    mapie_clf = MultiLabelClassificationController(
        predict_function=clf.predict_proba, random_state=random_state
    )
    mapie_clf.partial_calibrate(X_toy, y_toy)
    with pytest.raises(ValueError, match=r".*features, but*"):
        mapie_clf.partial_calibrate(X, y)


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
    y = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1]])
    numeric_preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_preprocessor = Pipeline(
        steps=[("encoding", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_preprocessor, ["x_cat"]),
            ("num", numeric_preprocessor, ["x_num"]),
        ]
    )
    pipe = make_pipeline(preprocessor, MultiOutputClassifier(LogisticRegression()))
    pipe.fit(X, y)
    mapie_clf = MultiLabelClassificationController(
        predict_function=pipe.predict_proba,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=random_state,
        confidence_level=0.9,
        rcps_bound=args["rcps_bound"],
    )

    mapie_clf.calibrate(X, y)
    mapie_clf.predict(X)


def test_partial_calibrate_first_time():
    mclf = MultiLabelClassificationController(
        predict_function=toy_predict_function, random_state=random_state
    )
    assert mclf._check_partial_calibrate_first_call()


def test_partial_calibrate_second_time():
    clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
    mclf = MultiLabelClassificationController(
        predict_function=clf.predict_proba, random_state=random_state
    )
    mclf.partial_calibrate(X, y)
    assert not mclf._check_partial_calibrate_first_call()


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_toy_dataset_predictions(strategy: str) -> None:
    """
    Test toy_dataset_predictions.
    """
    args = STRATEGIES[strategy][0]
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function,
        method=args["method"],
        metric_control=args["metric_control"],
        random_state=random_state,
        target_level=0.8,
        confidence_level=0.9,
        rcps_bound=args["rcps_bound"],
    )
    mapie_clf.calibrate(X_toy, y_toy)
    _, y_ps = mapie_clf.predict(X_toy)
    np.testing.assert_allclose(y_ps[:, :, 0], y_toy_mapie[strategy], rtol=1e-6)


@pytest.mark.parametrize("method", ["rcps", "crc"])
def test_error_wrong_method_metric_precision(method: str) -> None:
    """
    Test that an error is returned when using a metric
    with invalid method .
    """
    with pytest.raises(ValueError, match=r".*Invalid method for metric*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            method=method,
            metric_control="precision",
        )


@pytest.mark.parametrize("method", ["ltt"])
def test_check_metric_control(method: str) -> None:
    """
    Test that an error is returned when using a metric
    with invalid method .
    """
    with pytest.raises(ValueError, match=r".*Invalid method for metric*"):
        MultiLabelClassificationController(
            predict_function=toy_predict_function,
            method=method,
            metric_control="recall",
        )


def test_method_none_precision() -> None:
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function,
        metric_control="precision",
        confidence_level=0.9,
    )
    mapie_clf.calibrate(X_toy, y_toy)
    assert mapie_clf.method == "ltt"


def test_method_none_recall() -> None:
    mapie_clf = MultiLabelClassificationController(
        predict_function=toy_predict_function, metric_control="recall"
    )
    mapie_clf.calibrate(X_toy, y_toy)
    assert mapie_clf.method == "crc"
