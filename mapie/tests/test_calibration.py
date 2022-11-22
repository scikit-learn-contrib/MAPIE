import numpy as np
from typing import Union
from sklearn.calibration import _SigmoidCalibration
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mapie.calibration import MapieCalibrator
from inspect import signature
import pytest

CALIBRATORS = [
    "sigmoid", "isotonic", _SigmoidCalibration(), LinearRegression()
]

random_state = 20

X, y = make_classification(
    n_samples=100000,
    n_classes=3,
    n_informative=4,
    random_state=random_state
)
y += 1

X_train, X_, y_train, y_ = train_test_split(
    X, y, test_size=0.33, random_state=random_state)


X_calib, X_test, y_calib, y_test = train_test_split(
    X_, y_, test_size=0.33, random_state=random_state)


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
def test_shape_of_output(
    calibrator: Union[str, RegressorMixin]
    ) -> None:
    mapie_cal = MapieCalibrator(calibration_method=calibrator)
    mapie_cal.fit(X, y)
    calib_ = mapie_cal.predict_proba(X)
    assert len(calib_) == len(y)
    assert np.size(calib_.shape) == 1
