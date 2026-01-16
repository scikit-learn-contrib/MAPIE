from inspect import signature
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytest
import sklearn
from numpy.random import RandomState
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.calibration import _SigmoidCalibration
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from mapie._venn_abers import VennAbers, VennAbersMultiClass, predict_proba_prefitted_va
from mapie.calibration import TopLabelCalibrator, VennAbersCalibrator
from mapie.metrics.calibration import expected_calibration_error, top_label_ece, _get_binning_groups

random_state = 20
random_state_va = 42

CALIBRATORS = ["sigmoid", "isotonic", _SigmoidCalibration(), LinearRegression()]

ESTIMATORS = [
    LogisticRegression(),
    RandomForestClassifier(random_state=random_state),
]

results = {
    "split": {
        "y_score": [
            [np.nan, 0.33333333, np.nan],
            [0.66666667, np.nan, np.nan],
            [np.nan, 0.33333333, np.nan],
            [np.nan, 0.33333333, np.nan],
            [np.nan, 0.33333333, np.nan],
            [np.nan, np.nan, 0.35635314],
            [np.nan, np.nan, 0.18501723],
        ],
        "top_label_ece": 0.3881,
    },
    "prefit": {
        "y_score": [
            [np.nan, np.nan, 0.85714286],
            [0.83333333, np.nan, np.nan],
            [np.nan, 0.83333333, np.nan],
            [np.nan, np.nan, 0.85714286],
            [np.nan, np.nan, 0.85714286],
            [np.nan, np.nan, 0.85714286],
            [0.83333333, np.nan, np.nan],
        ],
        "top_label_ece": 0.31349206349206343,
    },
}

results_binary = {
    "split": {
        "y_score": [
            [0.74020596, np.nan],
            [0.4247601, np.nan],
            [np.nan, 0.66666667],
            [0.72980855, np.nan],
            [np.nan, 0.66666667],
            [0.81058943, np.nan],
            [0.7551083, np.nan],
            [0.59798388, np.nan],
            [np.nan, 0.66666667],
            [np.nan, 0.66666667],
        ],
        "top_label_ece": 0.315922,
        "ece": 0.554227,
    },
    "prefit": {
        "y_score": [
            [0.85714286, np.nan],
            [np.nan, 0.85714286],
            [np.nan, 0.85714286],
            [0.85714286, np.nan],
            [np.nan, 0.85714286],
            [0.85714286, np.nan],
            [0.85714286, np.nan],
            [0.85714286, np.nan],
            [np.nan, 0.85714286],
            [np.nan, 0.85714286],
        ],
        "top_label_ece": 0.1428571428571429,
        "ece": 0.3571428571428571,
    },
}


X, y = make_classification(
    n_samples=20, n_classes=3, n_informative=4, random_state=random_state
)

X_, X_test, y_, y_test = train_test_split(
    X, y, test_size=0.33, random_state=random_state
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_, y_, test_size=0.33, random_state=random_state
)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    TopLabelCalibrator()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_cal = TopLabelCalibrator()
    assert mapie_cal.calibrator is None
    assert mapie_cal.cv == "split"


def test_default_fit_params() -> None:
    """Test default sample weights and other parameters."""
    mapie_cal = TopLabelCalibrator()
    assert signature(mapie_cal.fit).parameters["sample_weight"].default is None
    assert signature(mapie_cal.fit).parameters["calib_size"].default == 0.33
    assert signature(mapie_cal.fit).parameters["random_state"].default is None
    assert signature(mapie_cal.fit).parameters["shuffle"].default is True
    assert signature(mapie_cal.fit).parameters["stratify"].default is None


def test_false_str_estimator() -> None:
    """Test invalid string input for calibrator."""
    with pytest.raises(
        ValueError,
        match=r".*Please provide a string in*",
    ):
        mapie_cal = TopLabelCalibrator(calibrator="not_estimator")
        mapie_cal.fit(X, y)


def test_estimator_none() -> None:
    """Test that no input for calibrator will return a sigmoid"""
    mapie_cal = TopLabelCalibrator()
    mapie_cal.fit(X, y)
    assert isinstance(
        mapie_cal.calibrators[list(mapie_cal.calibrators.keys())[0]],
        _SigmoidCalibration,
    )


def test_check_type_of_target() -> None:
    """Test the type of target."""
    X = [0.5, 0.2, 0.4, 0.8, 3.8]
    y = [0.4, 0.2, 3.6, 3, 0.2]
    mapie_cal = TopLabelCalibrator()
    with pytest.raises(
        ValueError, match=r".*Make sure to have one of the allowed targets:*"
    ):
        mapie_cal.fit(X, y)


def test_prefit_cv_argument() -> None:
    """Test that prefit method works"""
    est = RandomForestClassifier().fit(X, y)
    mapie_cal = TopLabelCalibrator(estimator=est, cv="prefit")
    mapie_cal.fit(X, y)


def test_split_cv_argument() -> None:
    """Test that split method works"""
    mapie_cal = TopLabelCalibrator(cv="split")
    mapie_cal.fit(X, y)


@pytest.mark.parametrize("cv", ["noprefit", "nosplit"])
def test_invalid_cv_argument(cv: str) -> None:
    """Test that other cv method does not work"""
    with pytest.raises(
        ValueError,
        match=r".*Invalid cv argument*",
    ):
        mapie_cal = TopLabelCalibrator(cv=cv)
        mapie_cal.fit(X, y)


def test_prefit_split_same_results() -> None:
    """Test that prefit and split method return the same result"""
    est = RandomForestClassifier(random_state=random_state).fit(X_train, y_train)
    mapie_cal_prefit = TopLabelCalibrator(estimator=est, cv="prefit")
    mapie_cal_prefit.fit(X_calib, y_calib)

    mapie_cal_split = TopLabelCalibrator(
        estimator=RandomForestClassifier(random_state=random_state)
    )
    mapie_cal_split.fit(X_, y_, random_state=random_state)
    y_prefit = mapie_cal_prefit.predict_proba(X_test)
    y_split = mapie_cal_split.predict_proba(X_test)
    np.testing.assert_allclose(y_split, y_prefit)


def test_not_seen_calibrator() -> None:
    """
    Test that there is a warning if no calibration occurs
    due to no calibrator for this class.
    """
    with pytest.warns(UserWarning, match=r".*WARNING: This predicted label*"):
        mapie_cal = TopLabelCalibrator()
        mapie_cal.fit(X, y)
        mapie_cal.calibrators.clear()
        mapie_cal.predict_proba(X)


@pytest.mark.parametrize("calibrator", CALIBRATORS)
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.filterwarnings("ignore:.*predicted label.*not been seen.*:UserWarning")
def test_shape_of_output(
    calibrator: Union[str, RegressorMixin], estimator: ClassifierMixin
) -> None:
    """Test that the size of the outputs are coherent."""
    mapie_cal = TopLabelCalibrator(
        estimator=estimator,
        calibrator=calibrator,
    )
    mapie_cal.fit(X, y)
    calib_ = mapie_cal.predict_proba(X)
    assert calib_.shape == (len(y), mapie_cal.n_classes_)


def test_number_of_classes_equal_calibrators() -> None:
    """
    Test that the number of calibrators is the same as the number
    of classes in the calibration step.
    """
    mapie_cal = TopLabelCalibrator()
    mapie_cal.fit(X=X_, y=y_, random_state=random_state)
    y_pred_calib_set = mapie_cal.single_estimator_.predict(X=X_calib)
    assert len(mapie_cal.calibrators) == len(np.unique(y_pred_calib_set))


def test_same_predict() -> None:
    """Test that the same prediction is made regardless of the calibration."""
    mapie_cal = TopLabelCalibrator()
    mapie_cal.fit(X=X_, y=y_, random_state=random_state)
    y_pred_calib_set = mapie_cal.single_estimator_.predict(X=X_test)
    y_pred_calib_set_through_predict = mapie_cal.predict(X=X_test)
    y_pred_calibrated_test_set = np.nanargmax(mapie_cal.predict_proba(X=X_test), axis=1)
    np.testing.assert_allclose(y_pred_calib_set, y_pred_calibrated_test_set)
    np.testing.assert_allclose(y_pred_calib_set, y_pred_calib_set_through_predict)


@pytest.mark.parametrize("cv", TopLabelCalibrator.valid_cv)
def test_correct_results(cv: str) -> None:
    """
    Test that the y_score and top label score from the test dataset result
    in the correct scores (in a multi-class setting).
    """
    mapie_cal = TopLabelCalibrator(cv=cv)
    mapie_cal.fit(X=X_, y=y_, random_state=random_state)
    pred_ = mapie_cal.predict_proba(X_test)
    top_label_ece_ = top_label_ece(y_test, pred_)
    np.testing.assert_array_almost_equal(
        np.array(results[cv]["y_score"]), np.array(pred_), decimal=2
    )
    np.testing.assert_allclose(  # type:ignore
        results[cv]["top_label_ece"], top_label_ece_, rtol=1e-2
    )


@pytest.mark.parametrize("cv", TopLabelCalibrator.valid_cv)
def test_correct_results_binary(cv: str) -> None:
    """
    Test that the y_score and top label score from the test dataset result
    in the correct scores (in a binary setting).
    """
    X_binary, y_binary = make_classification(
        n_samples=10,
        n_classes=2,
        n_informative=4,
        random_state=random_state,
    )
    mapie_cal = TopLabelCalibrator(cv=cv)
    mapie_cal.fit(X=X_binary, y=y_binary, random_state=random_state)
    pred_ = mapie_cal.predict_proba(X_binary)
    top_label_ece_ = top_label_ece(y_binary, pred_)
    ece = expected_calibration_error(y_binary, pred_)
    np.testing.assert_array_almost_equal(
        np.array(results_binary[cv]["y_score"]), np.array(pred_), decimal=2
    )
    np.testing.assert_allclose(  # type:ignore
        results_binary[cv]["top_label_ece"], top_label_ece_, rtol=1e-2
    )
    np.testing.assert_allclose(  # type:ignore
        results_binary[cv]["ece"], ece, rtol=1e-2
    )


def test_different_binary_y_combinations() -> None:
    """
    Test that despite the different maximum in y value, the
    scores are always the same.
    """
    X_comb, y_comb = make_classification(
        n_samples=20, n_classes=3, n_informative=4, random_state=random_state
    )
    mapie_cal = TopLabelCalibrator()
    mapie_cal.fit(X_comb, y_comb, random_state=random_state)
    y_score = mapie_cal.predict_proba(X_comb)

    y_comb1 = np.where(y_comb == 2, 3, y_comb)
    mapie_cal1 = TopLabelCalibrator()
    mapie_cal1.fit(X_comb, y_comb1, random_state=random_state)
    y_score1 = mapie_cal1.predict_proba(X_comb)

    y_comb2 = np.where(y_comb == 2, 40, y_comb)
    mapie_cal2 = TopLabelCalibrator()
    mapie_cal2.fit(X_comb, y_comb2, random_state=random_state)
    y_score2 = mapie_cal2.predict_proba(X_comb)
    np.testing.assert_array_almost_equal(y_score, y_score1)
    np.testing.assert_array_almost_equal(y_score, y_score2)
    assert top_label_ece(y_comb, y_score, classes=mapie_cal.classes_) == top_label_ece(
        y_comb1, y_score1, classes=mapie_cal1.classes_
    )
    assert top_label_ece(y_comb, y_score, classes=mapie_cal.classes_) == top_label_ece(
        y_comb2, y_score2, classes=mapie_cal2.classes_
    )


@pytest.mark.parametrize("calibrator", [LinearRegression(), "isotonic"])
def test_results_with_constant_sample_weights(
    calibrator: Union[str, RegressorMixin],
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.

    Note that the calibration implementations from sklearn `calibration.py`
    file would not pass these tests.
    """
    n_samples = len(X)
    estimator = RandomForestClassifier(random_state=random_state)
    mapie_clf0 = TopLabelCalibrator(estimator=estimator, calibrator=calibrator)
    mapie_clf1 = TopLabelCalibrator(estimator=estimator, calibrator=calibrator)
    mapie_clf2 = TopLabelCalibrator(estimator=estimator, calibrator=calibrator)
    mapie_clf0.fit(X, y, sample_weight=None, random_state=random_state)
    mapie_clf1.fit(
        X, y, sample_weight=np.ones(shape=n_samples), random_state=random_state
    )
    mapie_clf2.fit(
        X, y, sample_weight=np.ones(shape=n_samples) * 5, random_state=random_state
    )
    y_pred0 = mapie_clf0.predict_proba(X)
    y_pred1 = mapie_clf1.predict_proba(X)
    y_pred2 = mapie_clf2.predict_proba(X)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred0, y_pred2)


def test_pipeline_compatibility() -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5],
        }
    )
    y = pd.Series([0, 1, 2, 0, 1, 0])
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
    pipe = make_pipeline(preprocessor, LogisticRegression())
    pipe.fit(X, y)
    mapie = TopLabelCalibrator(estimator=pipe)
    mapie.fit(X, y)
    mapie.predict(X)


def test_fit_parameters_passing() -> None:
    """
    Test passing fit parameters, here early stopping at iteration 3.
    Checks that underlying GradientBoosting estimators have used 3 iterations
    only during boosting, instead of default value for n_estimators (=100).
    """
    gb = GradientBoostingClassifier(random_state=random_state)

    mapie = TopLabelCalibrator(estimator=gb)

    def early_stopping_monitor(i, est, locals):
        """Returns True on the 3rd iteration."""
        if i == 2:
            return True
        else:
            return False

    mapie.fit(X, y, monitor=early_stopping_monitor)

    assert mapie.single_estimator_.estimators_.shape[0] == 3


random_state_bins = 1234567890
prng = RandomState(random_state_bins)
y_score = prng.random(51)

results_binning = {
    "quantile": [
        0.03075388,
        0.17261836,
        0.33281326,
        0.43939618,
        0.54867626,
        0.64881987,
        0.73440899,
        0.77793816,
        0.89000413,
        0.99610621,
    ],
    "uniform": [
        0,
        0.11111111,
        0.22222222,
        0.33333333,
        0.44444444,
        0.55555556,
        0.66666667,
        0.77777778,
        0.88888889,
        1,
    ],
    "array split": [
        0.62689056,
        0.74743526,
        0.87642114,
        0.88321124,
        0.8916548,
        0.94083846,
        0.94999075,
        0.98759822,
        0.99610621,
        np.inf,
    ],
}


@pytest.mark.parametrize("strategy", ["quantile", "uniform", "array split"])
def test_binning_group_strategies(strategy: str) -> None:
    """Test that different strategies have the correct outputs."""
    bins_ = _get_binning_groups(y_score, num_bins=10, strategy=strategy)
    np.testing.assert_allclose(results_binning[strategy], bins_, rtol=1e-05)


# ============================================================================
# VennAbersCalibrator Tests (merged from test_venn_abers_calibration.py)
# ============================================================================


VA_ESTIMATORS = [
    LogisticRegression(random_state=random_state_va),
    RandomForestClassifier(n_estimators=10, random_state=random_state_va),
    GaussianNB(),
]


# Binary classification dataset
X_binary, y_binary = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    n_informative=10,
    random_state=random_state_va,
)
X_binary_train, X_binary_test, y_binary_train, y_binary_test = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=random_state_va
)
X_binary_proper, X_binary_cal, y_binary_proper, y_binary_cal = train_test_split(
    X_binary_train, y_binary_train, test_size=0.3, random_state=random_state_va
)

# Multi-class classification dataset
X_multi, y_multi = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=3,
    n_informative=10,
    random_state=random_state_va,
)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=random_state_va
)
X_multi_proper, X_multi_cal, y_multi_proper, y_multi_cal = train_test_split(
    X_multi_train, y_multi_train, test_size=0.3, random_state=random_state_va
)


@pytest.mark.parametrize("cv", ["prefit", None])
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_valid_cv_argument(cv: Optional[str]) -> None:
    """Test valid cv methods."""
    if cv == "prefit":
        est = GaussianNB().fit(X_binary_train, y_binary_train)
        va_cal = VennAbersCalibrator(estimator=est, cv=cv)
        va_cal.fit(X_binary_cal, y_binary_cal)
    else:
        va_cal = VennAbersCalibrator(estimator=GaussianNB(), cv=cv, inductive=True)
        va_cal.fit(X_binary_train, y_binary_train)


@pytest.mark.parametrize("cv", ["split", "invalid"])
def test_va_invalid_cv_argument(cv: str) -> None:
    """Test that invalid cv methods raise ValueError."""
    with pytest.raises(ValueError, match=r".*Invalid cv argument*"):
        va_cal = VennAbersCalibrator(estimator=GaussianNB(), cv=cv)
        va_cal.fit(X_binary_train, y_binary_train)


def test_va_prefit_unfitted_estimator_raises_error() -> None:
    """Test that VennAbersCalibrator in 'prefit' mode raises if estimator not fitted."""
    va_cal = VennAbersCalibrator(estimator=GaussianNB(), cv="prefit")
    with pytest.raises((ValueError, AttributeError)):
        va_cal.fit(X_binary_cal, y_binary_cal)


def test_va_prefit_requires_estimator() -> None:
    """Test that prefit mode requires an estimator."""
    va_cal = VennAbersCalibrator(cv="prefit")
    with pytest.raises(ValueError, match=r".*an estimator must be provided*"):
        va_cal.fit(X_binary_train, y_binary_train)


def test_va_prefit_missing_last_step_raises_not_fitted_error() -> None:
    """Test that a pipeline lacking a fitted final step raises NotFittedError."""

    class MissingEstimatorPipeline(Pipeline):
        def __getitem__(self, ind):
            if isinstance(ind, int) and ind == -1:
                return None
            return super().__getitem__(ind)

    faulty_pipeline = MissingEstimatorPipeline(
        [
            ("transform", SimpleImputer(strategy="mean")),
            ("clf", LogisticRegression(random_state=random_state_va)),
        ]
    )

    va_cal = VennAbersCalibrator(estimator=faulty_pipeline, cv="prefit")
    with pytest.raises(
        NotFittedError, match=r"For cv='prefit', the estimator must be already fitted"
    ):
        va_cal.fit(X_binary_cal, y_binary_cal)


@pytest.mark.parametrize(
    "mode,mode_params,X_train,y_train,X_test,n_classes",
    [
        (
            "inductive",
            {"inductive": True},
            X_binary_train,
            y_binary_train,
            X_binary_test,
            2,
        ),
        (
            "inductive",
            {"inductive": True},
            X_multi_train,
            y_multi_train,
            X_multi_test,
            3,
        ),
        (
            "cross_val",
            {"inductive": False, "n_splits": 3},
            X_binary_train,
            y_binary_train,
            X_binary_test,
            2,
        ),
        (
            "cross_val",
            {"inductive": False, "n_splits": 3},
            X_multi_train,
            y_multi_train,
            X_multi_test,
            3,
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_mode_functionality(
    mode, mode_params, X_train, y_train, X_test, n_classes
) -> None:
    """Test all modes (inductive/cross-validation) for binary and multiclass."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), random_state=random_state_va, **mode_params
    )
    va_cal.fit(X_train, y_train)
    probs = va_cal.predict_proba(X_test)
    preds = va_cal.predict(X_test)

    assert probs.shape == (len(X_test), n_classes)
    assert preds.shape == (len(X_test),)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))
    assert hasattr(va_cal, "classes_")
    assert hasattr(va_cal, "n_classes_")
    assert va_cal.n_classes_ == n_classes


@pytest.mark.parametrize(
    "X_proper,y_proper,X_cal,y_cal,X_test,n_classes",
    [
        (
            X_binary_proper,
            y_binary_proper,
            X_binary_cal,
            y_binary_cal,
            X_binary_test,
            2,
        ),
        (X_multi_proper, y_multi_proper, X_multi_cal, y_multi_cal, X_multi_test, 3),
    ],
)
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_prefit_mode(X_proper, y_proper, X_cal, y_cal, X_test, n_classes) -> None:
    """Test prefit mode for binary and multiclass."""
    clf = GaussianNB().fit(X_proper, y_proper)
    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_cal, y_cal)
    probs = va_cal.predict_proba(X_test)

    assert probs.shape == (len(X_test), n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert hasattr(va_cal, "single_estimator_")


def test_va_cross_validation_requires_n_splits() -> None:
    """Test that CVAP requires n_splits parameter."""
    va_cal = VennAbersCalibrator(estimator=GaussianNB(), inductive=False, n_splits=None)
    with pytest.raises(
        ValueError, match=r".*For Cross Venn-ABERS please provide n_splits*"
    ):
        va_cal.fit(X_binary_train, y_binary_train)


@pytest.mark.parametrize("estimator", VA_ESTIMATORS)
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_different_estimators(estimator) -> None:
    """Test VennAbersCalibrator with different base estimators."""
    va_cal = VennAbersCalibrator(
        estimator=estimator, inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_va_estimator_none_raises_error() -> None:
    """Test that None estimator raises ValueError."""
    va_cal = VennAbersCalibrator(estimator=None)
    with pytest.raises(ValueError, match=r".*an estimator must be provided*"):
        va_cal.fit(X_binary_train, y_binary_train)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_sample_weights_constant() -> None:
    """Test that constant sample weights give same results as None."""
    sklearn.set_config(enable_metadata_routing=True)
    n_samples = len(X_binary_train)
    weighted_estimator = GaussianNB().set_fit_request(sample_weight=True)

    va_cal_none = VennAbersCalibrator(
        estimator=weighted_estimator, inductive=True, random_state=random_state_va
    )
    va_cal_none.fit(X_binary_train, y_binary_train, sample_weight=None)

    va_cal_ones = VennAbersCalibrator(
        estimator=weighted_estimator, inductive=True, random_state=random_state_va
    )
    va_cal_ones.fit(X_binary_train, y_binary_train, sample_weight=np.ones(n_samples))

    probs_none = va_cal_none.predict_proba(X_binary_test)
    probs_ones = va_cal_ones.predict_proba(X_binary_test)
    np.testing.assert_allclose(probs_none, probs_ones, rtol=1e-2, atol=1e-2)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_sample_weights_variable() -> None:
    """Test that variable sample weights affect the results."""
    sklearn.set_config(enable_metadata_routing=True)
    n_samples = len(X_binary_train)

    va_cal_uniform = VennAbersCalibrator(
        estimator=RandomForestClassifier(n_estimators=10, random_state=random_state_va),
        inductive=True,
        random_state=random_state_va,
    )
    va_cal_uniform.fit(X_binary_train, y_binary_train, sample_weight=None)

    sample_weights = np.random.RandomState(random_state_va).uniform(
        0.1, 2.0, size=n_samples
    )
    estimator_weighted = RandomForestClassifier(
        n_estimators=10, random_state=random_state_va
    ).set_fit_request(sample_weight=True)

    va_cal_weighted = VennAbersCalibrator(
        estimator=estimator_weighted, inductive=True, random_state=random_state_va
    )
    va_cal_weighted.fit(X_binary_train, y_binary_train, sample_weight=sample_weights)

    probs_uniform = va_cal_uniform.predict_proba(X_binary_test)
    probs_weighted = va_cal_weighted.predict_proba(X_binary_test)
    assert not np.allclose(probs_uniform, probs_weighted)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_venn_abers_cv_with_sample_weight() -> None:
    """Test VennAbersCV with sample weights in cross-validation mode."""
    sklearn.set_config(enable_metadata_routing=True)
    sample_weight = np.ones(len(y_binary_train))
    sample_weight[: len(y_binary_train) // 2] = 2.0

    weighted_estimator = GaussianNB().set_fit_request(sample_weight=True)
    va_cal = VennAbersCalibrator(
        estimator=weighted_estimator,
        inductive=False,
        n_splits=3,
        random_state=random_state_va,
    )
    va_cal.fit(X_binary_train, y_binary_train, sample_weight=sample_weight)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_random_state_reproducibility() -> None:
    """Test that random_state ensures reproducible results."""
    va_cal1 = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal1.fit(X_binary_train, y_binary_train)
    probs1 = va_cal1.predict_proba(X_binary_test)

    va_cal2 = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal2.fit(X_binary_train, y_binary_train)
    probs2 = va_cal2.predict_proba(X_binary_test)

    np.testing.assert_array_equal(probs1, probs2)


@pytest.mark.parametrize(
    "param_name,override_value",
    [
        ("random_state", 123),
        ("shuffle", False),
        ("stratify", y_binary_train),
        ("calib_size", 0.4),
    ],
)
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_fit_parameters_override(param_name, override_value) -> None:
    """Test that fit() parameters override constructor parameters."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    kwargs = {param_name: override_value}
    va_cal.fit(X_binary_train, y_binary_train, **kwargs)
    probs = va_cal.predict_proba(X_binary_test)
    assert probs.shape == (len(X_binary_test), 2)


@pytest.mark.parametrize("cal_size", [0.2, 0.3, 0.5])
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_different_calibration_sizes(cal_size: float) -> None:
    """Test that different calibration sizes work correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train, calib_size=cal_size)
    probs = va_cal.predict_proba(X_binary_test)
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("n_splits", [2, 3, 5])
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_different_n_splits(n_splits: int) -> None:
    """Test that different n_splits values work correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=n_splits,
        random_state=random_state_va,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_va_n_splits_too_small_raises_error() -> None:
    """Test that n_splits < 2 raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=1,
        random_state=random_state_va,
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_binary_train, y_binary_train)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_pipeline_compatibility() -> None:
    """Test that VennAbersCalibrator works with sklearn pipelines."""
    X_df = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"] * 10,
            "x_num": [0, 1, 1, 4, np.nan, 5] * 10,
        }
    )
    y_series = pd.Series([0, 1, 0, 1, 0, 1] * 10)

    numeric_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    categorical_preprocessor = Pipeline(
        steps=[("encoding", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_preprocessor, ["x_cat"]),
            ("num", numeric_preprocessor, ["x_num"]),
        ]
    )
    pipe = make_pipeline(preprocessor, LogisticRegression(random_state=random_state_va))
    pipe.fit(X_df, y_series)

    va_cal = VennAbersCalibrator(
        estimator=pipe, inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_df, y_series)
    probs = va_cal.predict_proba(X_df)

    assert probs.shape == (len(y_series), 2)


@pytest.mark.parametrize(
    "X_type,y_type",
    [
        (pd.DataFrame, pd.Series),
        (np.ndarray, np.ndarray),
        (pd.DataFrame, np.ndarray),
    ],
)
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_different_input_types(X_type, y_type) -> None:
    """Test with different input data types."""
    X_train = X_type(X_binary_train) if X_type == pd.DataFrame else X_binary_train
    y_train = y_type(y_binary_train) if y_type == pd.Series else y_binary_train
    X_test = X_type(X_binary_test) if X_type == pd.DataFrame else X_binary_test

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_train, y_train)
    probs = va_cal.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)


@pytest.mark.parametrize(
    "X,y,error_match",
    [
        (np.array([]).reshape(0, 20), np.array([]), ".*"),
        (np.zeros((10, 20)), np.zeros(10), ".*"),
        (X_binary_train[:50], y_binary_train[:40], ".*"),
    ],
)
def test_va_invalid_inputs_raise_error(X, y, error_match) -> None:
    """Test that invalid inputs raise appropriate errors."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    with pytest.raises(ValueError):
        va_cal.fit(X, y)


def test_va_predict_before_fit_raises_error() -> None:
    """Test that calling predict before fit raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    with pytest.raises(Exception):
        va_cal.predict(X_binary_test)


@pytest.mark.parametrize("calib_size", [1.5, -0.1])
def test_va_invalid_cal_size_raises_error(calib_size) -> None:
    """Test that invalid cal_size values raise an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_binary_train, y_binary_train, calib_size=calib_size)


@pytest.mark.parametrize("precision", [None, 2, 4])
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_precision_parameter(precision: Optional[int]) -> None:
    """Test that precision parameter works correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        random_state=random_state_va,
        precision=precision,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_precision_parameter_multiclass() -> None:
    """Test that precision parameter works correctly for multiclass."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        precision=4,
        random_state=random_state_va,
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)
    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_integration_with_cross_validation() -> None:
    """Test integration with sklearn's cross-validation utilities."""
    from sklearn.model_selection import cross_val_score

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    scores = cross_val_score(
        va_cal, X_binary_train, y_binary_train, cv=3, scoring="accuracy"
    )
    assert len(scores) == 3
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_va_clone_estimator() -> None:
    """Test that VennAbersCalibrator can be cloned."""
    from sklearn.base import clone

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train)
    va_cal_clone = clone(va_cal)

    is_fitted = True
    try:
        check_is_fitted(va_cal_clone.estimator)
    except NotFittedError:
        is_fitted = False

    assert va_cal_clone.inductive == va_cal.inductive
    assert is_fitted is False


def test_va_check_is_fitted_after_fit() -> None:
    """Test that check_is_fitted passes after fitting."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train)
    check_is_fitted(va_cal)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_predict_proba_prefitted_va_one_vs_all() -> None:
    """Test predict_proba_prefitted_va with one_vs_all strategy."""
    X, y = make_classification(
        n_samples=500, n_classes=3, n_informative=10, random_state=random_state_va
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state_va
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state_va
    )

    clf = GaussianNB().fit(X_train, y_train)
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)

    p_calibrated, p0p1 = predict_proba_prefitted_va(
        p_cal, y_cal, p_test, precision=None, va_tpe="one_vs_all"
    )

    assert p_calibrated.shape == p_test.shape
    assert np.allclose(p_calibrated.sum(axis=1), 1.0)
    assert len(p0p1) == 3


def test_va_predict_proba_prefitted_va_invalid_type() -> None:
    """Test that invalid va_tpe raises ValueError."""
    X, y = make_classification(n_samples=100, n_classes=2, random_state=random_state_va)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state_va
    )

    clf = GaussianNB().fit(X_train, y_train)
    p_cal = clf.predict_proba(X_train)
    p_test = clf.predict_proba(X_test)

    with pytest.raises(ValueError, match="Invalid va_tpe"):
        predict_proba_prefitted_va(p_cal, y_train, p_test, va_tpe="invalid_type")


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_venn_abers_basic() -> None:
    """Test basic VennAbers functionality for binary classification."""
    X, y = make_classification(n_samples=500, n_classes=2, random_state=random_state_va)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state_va
    )
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state_va
    )

    clf = GaussianNB().fit(X_train_proper, y_train_proper)
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)

    va = VennAbers()
    va.fit(p_cal, y_cal)
    p_prime, p0_p1 = va.predict_proba(p_test)

    assert p_prime.shape == (len(X_test), 2)
    assert p0_p1.shape == (len(X_test), 2)
    assert np.allclose(p_prime.sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_venn_abers_cv_p0_p1_output() -> None:
    """Test VennAbersCV predict_proba with p0_p1_output=True."""
    from mapie._venn_abers import VennAbersCV

    va_cv = VennAbersCV(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cv.fit(X_binary_train, y_binary_train)
    p_prime, p0_p1 = va_cv.predict_proba(X_binary_test, p0_p1_output=True)

    assert p_prime.shape == (len(X_binary_test), 2)
    assert p0_p1.shape == (len(X_binary_test), 2)
    assert np.allclose(p_prime.sum(axis=1), 1.0)


def test_va_multiclass_cross_validation_requires_n_splits() -> None:
    """Test that VennAbersMultiClass in CVAP mode requires n_splits parameter."""
    va_multi = VennAbersMultiClass(
        estimator=GaussianNB(), inductive=False, n_splits=None
    )
    with pytest.raises(
        Exception, match=r".*For Cross Venn ABERS please provide n_splits.*"
    ):
        va_multi.fit(X_multi_train, y_multi_train)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_multiclass_p0_p1_output() -> None:
    """Test VennAbersMultiClass with p0_p1_output=True."""
    n_samples, n_features, n_classes = 100, 4, 3
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_test = np.random.randn(30, n_features)

    va_multi = VennAbersMultiClass(
        estimator=GaussianNB(),
        inductive=True,
        cal_size=0.3,
        random_state=random_state_va,
    )
    va_multi.fit(X_train, y_train)
    p_prime, p0_p1_list = va_multi.predict_proba(X_test, loss="log", p0_p1_output=True)

    assert p_prime.shape == (len(X_test), n_classes)
    assert np.allclose(p_prime.sum(axis=1), 1.0)
    assert len(p0_p1_list) == n_classes * (n_classes - 1) // 2


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_prefit_predict_proba_without_single_estimator() -> None:
    """Test that predict_proba raises RuntimeError when single_estimator_ is None in prefit mode."""
    clf = GaussianNB().fit(X_binary_proper, y_binary_proper)
    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)
    va_cal.single_estimator_ = None

    with pytest.raises(
        RuntimeError, match=r"single_estimator_ should not be None in prefit mode"
    ):
        va_cal.predict_proba(X_binary_test)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_prefit_predict_proba_without_n_classes() -> None:
    """Test that predict_proba raises RuntimeError when n_classes_ is None after fitting."""
    clf = GaussianNB().fit(X_binary_proper, y_binary_proper)
    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)
    va_cal.n_classes_ = None

    with pytest.raises(
        RuntimeError, match=r"n_classes_ should not be None after fitting"
    ):
        va_cal.predict_proba(X_binary_test)


def test_va_predict_without_classes() -> None:
    """Test that predict raises RuntimeError when classes_ is None after fitting."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train)
    va_cal.classes_ = None

    with pytest.raises(
        RuntimeError, match=r"classes_ should not be None after fitting"
    ):
        va_cal.predict(X_binary_test)


def test_va_prefit_classes_none_after_fitting() -> None:
    """Test that fit raises RuntimeError when classes_ is None after fitting estimator."""
    clf = GaussianNB().fit(X_binary_train, y_binary_train)
    clf.classes_ = None
    va_cal = VennAbersCalibrator(
        estimator=clf, cv="prefit", random_state=random_state_va
    )

    with pytest.raises(
        RuntimeError, match=r"classes_ should not be None after fitting estimator"
    ):
        va_cal.fit(X_binary_test, y_binary_test)


@pytest.mark.parametrize("cv_ensemble", [True, False])
@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_cv_ensemble_cross_binary(cv_ensemble) -> None:
    """Test cv_ensemble parameter with cross-validation mode."""
    va_cal = VennAbersCalibrator(
        estimator=LogisticRegression(random_state=random_state_va),
        inductive=False,
        n_splits=3,
        cv_ensemble=cv_ensemble,
        random_state=random_state_va,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    proba = va_cal.predict_proba(X_binary_test)

    assert proba.shape == (len(X_binary_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_venn_abers_cv_brier_loss() -> None:
    """Test VennAbersCV with Brier loss."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=3,
        random_state=random_state_va,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs_brier = va_cal.predict_proba(X_binary_test, loss="brier")

    assert probs_brier.shape == (len(X_binary_test), 2)
    assert np.allclose(probs_brier.sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_comprehensive_workflow() -> None:
    """Comprehensive test covering multiple aspects of VennAbersCalibrator."""
    modes: list[tuple[str, dict[str, Any]]] = [
        ("inductive", {"inductive": True}),
        ("cross_val", {"inductive": False, "n_splits": 3}),
    ]

    for mode_name, mode_params in modes:
        va_cal_binary = VennAbersCalibrator(
            estimator=RandomForestClassifier(
                n_estimators=10, random_state=random_state_va
            ),
            random_state=random_state_va,
            **mode_params,
        )
        va_cal_binary.fit(X_binary_train, y_binary_train)
        probs_binary = va_cal_binary.predict_proba(X_binary_test)
        preds_binary = va_cal_binary.predict(X_binary_test)

        assert probs_binary.shape == (len(X_binary_test), 2)
        assert preds_binary.shape == (len(X_binary_test),)
        assert np.allclose(probs_binary.sum(axis=1), 1.0)

        va_cal_multi = VennAbersCalibrator(
            estimator=RandomForestClassifier(
                n_estimators=10, random_state=random_state_va
            ),
            random_state=random_state_va,
            **mode_params,
        )
        va_cal_multi.fit(X_multi_train, y_multi_train)
        probs_multi = va_cal_multi.predict_proba(X_multi_test)
        preds_multi = va_cal_multi.predict(X_multi_test)

        assert probs_multi.shape == (len(X_multi_test), 3)
        assert preds_multi.shape == (len(X_multi_test),)
        assert np.allclose(probs_multi.sum(axis=1), 1.0)

    clf_binary = RandomForestClassifier(n_estimators=10, random_state=random_state_va)
    clf_binary.fit(X_binary_proper, y_binary_proper)
    va_cal_prefit = VennAbersCalibrator(estimator=clf_binary, cv="prefit")
    va_cal_prefit.fit(X_binary_cal, y_binary_cal)
    probs_prefit = va_cal_prefit.predict_proba(X_binary_test)

    assert probs_prefit.shape == (len(X_binary_test), 2)
    assert np.allclose(probs_prefit.sum(axis=1), 1.0)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_prefit_binary_va_calibrator_none_raises() -> None:
    clf = GaussianNB().fit(X_binary_proper, y_binary_proper)
    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)
    va_cal.va_calibrator_ = None
    with pytest.raises(
        RuntimeError,
        match="va_calibrator_ should not be None for binary classification",
    ):
        va_cal.predict_proba(X_binary_test)


def test_va_inductive_va_calibrator_none_raises() -> None:
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train)
    va_cal.va_calibrator_ = None
    with pytest.raises(
        RuntimeError,
        match="va_calibrator_ should not be None in inductive/cross-validation mode",
    ):
        va_cal.predict_proba(X_binary_test)


def test_va_inductive_va_calibrator_wrong_type_raises() -> None:
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train)
    va_cal.va_calibrator_ = VennAbers()
    with pytest.raises(
        RuntimeError,
        match="va_calibrator_ should be VennAbersMultiClass instance",
    ):
        va_cal.predict_proba(X_binary_test)


@pytest.mark.filterwarnings("ignore:: RuntimeWarning")
def test_va_inductive_loss_branch_and_else_branch() -> None:
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state_va
    )
    va_cal.fit(X_binary_train, y_binary_train)
    assert va_cal.va_calibrator_ is not None
    assert "loss" in signature(va_cal.va_calibrator_.predict_proba).parameters
    _ = va_cal.predict_proba(X_binary_test, loss="brier")
    original = va_cal.va_calibrator_.predict_proba

    def predict_proba_no_loss(X_processed, p0_p1_output=False):
        return original(X_processed, p0_p1_output=p0_p1_output)

    va_cal.va_calibrator_.predict_proba = predict_proba_no_loss  # type: ignore[method-assign]
    assert "loss" not in signature(va_cal.va_calibrator_.predict_proba).parameters
    _ = va_cal.predict_proba(X_binary_test, loss="log")
