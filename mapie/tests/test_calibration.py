from typing import Union
import numpy as np
from sklearn.calibration import _SigmoidCalibration
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mapie.calibration import MapieCalibrator
from inspect import signature
import pytest

random_state = 20

CALIBRATORS = [
    "sigmoid", "isotonic", _SigmoidCalibration(), LinearRegression()
]

ESTIMATORS = [
    LogisticRegression(),
    RandomForestClassifier(random_state=random_state),
]

results = {
    "normal": [
    [0.        , 0.66666667, 0.        ],
    [0.33333333, 0.        , 0.        ],
    [0.        , 0.66666667, 0.        ],
    [0.        , 0.66666667, 0.        ],
    [0.        , 0.66666667, 0.        ],
    [0.        , 0.        , 0.2       ],
    [0.        , 0.        , 0.2       ]
    ],
}

X, y = make_classification(
    n_samples=20,
    n_classes=3,
    n_informative=4,
    random_state=random_state
)

X_, X_test, y_, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_state
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_, y_, test_size=0.33, random_state=random_state
)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieCalibrator()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_cal = MapieCalibrator()
    assert mapie_cal.method == "top_label"
    assert mapie_cal.calibration_method is None
    assert mapie_cal.cv == "split"


def test_default_fit_params() -> None:
    """Test default sample weights."""
    mapie_cal = MapieCalibrator()
    assert (
        signature(mapie_cal.fit).parameters["sample_weight"].default
        is None
    )
    assert (
        signature(mapie_cal.fit).parameters["calib_size"].default
        == 0.3
    )
    assert (
        signature(mapie_cal.fit).parameters["random_state"].default
        is None
    )
    assert (
        signature(mapie_cal.fit).parameters["shuffle"].default
        is True
    )
    assert (
        signature(mapie_cal.fit).parameters["stratify"].default
        is None
    )


def test_false_str_estimator() -> None:
    with pytest.raises(
        ValueError,
        match=r".*Please provide a valid string*",
    ):
        mapie_cal = MapieCalibrator(
            calibration_method="not_estimator"
        )
        mapie_cal.fit(X, y)


def test_estimator_none() -> None:
    mapie_cal = MapieCalibrator()
    mapie_cal.fit(X, y)
    assert mapie_cal.calibration_method == "sigmoid"


def test_other_methods() -> None:
    with pytest.raises(
        ValueError,
        match=r".*No other methods have been*",
    ):
        mapie_cal = MapieCalibrator(method="no_method")
        mapie_cal.fit(X, y)


def test_not_seen_calibrator() -> None:
    with pytest.warns(
        UserWarning,
        match=r".*WARNING: This calibration was not previously seen*"
    ):
        mapie_cal = MapieCalibrator()
        mapie_cal.fit(X, y)
        mapie_cal.calibrators.clear()
        mapie_cal.predict_proba(X)


@pytest.mark.parametrize("calibrator", CALIBRATORS)
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_shape_of_output(
    calibrator: Union[str, RegressorMixin],
    estimator: ClassifierMixin
) -> None:
    mapie_cal = MapieCalibrator(
        estimator=estimator,
        calibration_method=calibrator,
    )
    mapie_cal.fit(X, y)
    calib_ = mapie_cal.predict_proba(X)
    assert calib_.shape == (len(y), mapie_cal.n_classes_)


def test_number_of_classes_equal_calibrators():
    mapie_cal = MapieCalibrator()
    mapie_cal.fit(
        X=X_train,
        y=y_train,
        X_calib=X_calib,
        y_calib=y_calib
    )
    y_pred_calib_set = mapie_cal.estimator.predict(X=X_calib)
    assert len(mapie_cal.calibrators) == len(np.unique(y_pred_calib_set))


def test_same_predict():
    mapie_cal = MapieCalibrator()
    mapie_cal.fit(
        X=X_train,
        y=y_train,
        X_calib=X_calib,
        y_calib=y_calib
    )
    y_pred_calib_set = mapie_cal.estimator.predict(X=X_test)
    y_pred_calibrated_test_set = mapie_cal.predict(X=X_test)
    np.testing.assert_allclose(y_pred_calib_set, y_pred_calibrated_test_set)


def test_correct_results():
    mapie_cal = MapieCalibrator()
    mapie_cal.fit(
        X=X_train, 
        y=y_train,
        X_calib=X_calib,
        y_calib=y_calib
    )
    pred_ = mapie_cal.predict_proba(X_test)
    np.testing.assert_allclose(results["normal"], pred_)



@pytest.mark.parametrize("calibrator", CALIBRATORS)
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_results_with_constant_sample_weights(
    calibrator: Union[str, RegressorMixin],
    estimator: ClassifierMixin
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie_clf0 = MapieCalibrator(estimator=estimator, calibration_method=calibrator)
    mapie_clf1 = MapieCalibrator(estimator=estimator, calibration_method=calibrator)
    mapie_clf2 = MapieCalibrator(estimator=estimator, calibration_method=calibrator)
    mapie_clf0.fit(X, y, sample_weight=None)
    mapie_clf1.fit(X, y, sample_weight=np.ones(shape=n_samples))
    mapie_clf2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 5)
    y_pred0 = mapie_clf0.predict_proba(X)
    y_pred1 = mapie_clf1.predict_proba(X)
    y_pred2 = mapie_clf2.predict_proba(X)

    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred0, y_pred2)
