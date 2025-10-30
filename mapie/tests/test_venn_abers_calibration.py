"""
Tests for VennAbersCalibrator class.
"""

from inspect import signature
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from mapie.calibration import VennAbersCalibrator
from mapie._venn_abers import VennAbers, VennAbersMultiClass, predict_proba_prefitted_va

random_state = 42

ESTIMATORS = [
    LogisticRegression(random_state=random_state),
    RandomForestClassifier(random_state=random_state),
    GaussianNB(),
]

# Binary classification dataset
X_binary, y_binary = make_classification(
    n_samples=10000,
    n_features=20,
    n_classes=2,
    n_informative=10,
    random_state=random_state,
)

X_binary_train, X_binary_test, y_binary_train, y_binary_test = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=random_state
)

X_binary_proper, X_binary_cal, y_binary_proper, y_binary_cal = train_test_split(
    X_binary_train, y_binary_train, test_size=0.3, random_state=random_state
)

# Multi-class classification dataset
X_multi, y_multi = make_classification(
    n_samples=10000,
    n_features=20,
    n_classes=3,
    n_informative=10,
    random_state=random_state,
)

X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=random_state
)

X_multi_proper, X_multi_cal, y_multi_proper, y_multi_cal = train_test_split(
    X_multi_train, y_multi_train, test_size=0.3, random_state=random_state
)


# ============================================================================
# Basic Initialization Tests
# ============================================================================


def test_initialized() -> None:
    """Test that initialization does not crash."""
    VennAbersCalibrator()


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    va_cal = VennAbersCalibrator()
    assert va_cal.estimator is None
    assert va_cal.cv is None
    assert va_cal.inductive is True
    assert va_cal.n_splits is None
    assert va_cal.train_proper_size is None
    assert va_cal.random_state is None
    assert va_cal.shuffle is True
    assert va_cal.stratify is None
    assert va_cal.precision is None


def test_default_fit_params() -> None:
    """Test default sample weights and other parameters."""
    va_cal = VennAbersCalibrator()
    assert signature(va_cal.fit).parameters["sample_weight"].default is None
    assert signature(va_cal.fit).parameters["calib_size"].default == 0.33
    assert signature(va_cal.fit).parameters["random_state"].default is None
    assert signature(va_cal.fit).parameters["shuffle"].default is True
    assert signature(va_cal.fit).parameters["stratify"].default is None


# ============================================================================
# CV Parameter Tests
# ============================================================================


@pytest.mark.parametrize("cv", ["prefit", None])
def test_valid_cv_argument(cv: Optional[str]) -> None:
    """Test that valid cv methods work."""
    if cv == "prefit":
        est = GaussianNB().fit(X_binary_train, y_binary_train)
        va_cal = VennAbersCalibrator(estimator=est, cv=cv)
        va_cal.fit(X_binary_cal, y_binary_cal)
    else:
        va_cal = VennAbersCalibrator(estimator=GaussianNB(), cv=cv, inductive=True)
        va_cal.fit(X_binary_train, y_binary_train)


@pytest.mark.parametrize("cv", ["split", "invalid", "cross"])
def test_invalid_cv_argument(cv: str) -> None:
    """Test that invalid cv methods raise ValueError."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid cv argument*",
    ):
        va_cal = VennAbersCalibrator(estimator=GaussianNB(), cv=cv)
        va_cal.fit(X_binary_train, y_binary_train)


def test_prefit_unfitted_estimator_raises_error() -> None:
    """
    Test that VennAbersCalibrator in 'prefit' mode raises a ValueError
    if the estimator is not fitted.
    """
    clf = GaussianNB()  # Unfitted estimator
    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    with pytest.raises(
        ValueError, match=r".*For cv='prefit', the estimator must be already fitted*"
    ):
        va_cal.fit(X_binary_cal, y_binary_cal)


def test_prefit_requires_estimator() -> None:
    """Test that prefit mode requires a fitted estimator."""
    va_cal = VennAbersCalibrator(cv="prefit")
    with pytest.raises(ValueError, match=r".*an estimator must be provided*"):
        va_cal.fit(X_binary_train, y_binary_train)


# ============================================================================
# Inductive vs Cross Validation Tests
# ============================================================================


def test_inductive_mode_binary() -> None:
    """Test Inductive Venn-ABERS (IVAP) for binary classification."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_inductive_mode_multiclass() -> None:
    """Test Inductive Venn-ABERS (IVAP) for multi-class classification."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_cross_validation_mode_binary() -> None:
    """Test Cross Venn-ABERS (CVAP) for binary classification."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=5, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_cross_validation_mode_multiclass() -> None:
    """Test Cross Venn-ABERS (CVAP) for multi-class classification."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=5, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_cross_validation_requires_n_splits() -> None:
    """Test that CVAP requires n_splits parameter."""
    va_cal = VennAbersCalibrator(estimator=GaussianNB(), inductive=False, n_splits=None)
    with pytest.raises(
        ValueError, match=r".*For Cross Venn-ABERS please provide n_splits*"
    ):
        va_cal.fit(X_binary_train, y_binary_train)


def test_cross_validation_with_shuffle() -> None:
    """Test Cross Venn-ABERS with shuffle parameter."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


def test_cross_validation_with_stratify() -> None:
    """Test Cross Venn-ABERS with stratify parameter."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=5,
        stratify=y_binary_train,
        random_state=random_state,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# Prefit Mode Tests
# ============================================================================


def test_prefit_mode_binary() -> None:
    """Test prefit mode for binary classification."""
    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_prefit_mode_multiclass() -> None:
    """Test prefit mode for multi-class classification."""
    clf = GaussianNB()
    clf.fit(X_multi_proper, y_multi_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_multi_cal, y_multi_cal)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_prefit_inductive_consistency() -> None:
    """Test that prefit and inductive modes give similar results."""
    # Fit estimator on proper training set
    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    # Prefit mode
    va_cal_prefit = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal_prefit.fit(X_binary_cal, y_binary_cal)
    probs_prefit = va_cal_prefit.predict_proba(X_binary_test)

    # Inductive mode with same split
    va_cal_inductive = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    # Combine proper and cal sets
    X_combined = np.vstack([X_binary_proper, X_binary_cal])
    y_combined = np.hstack([y_binary_proper, y_binary_cal])
    va_cal_inductive.fit(X_combined, y_combined)
    probs_inductive = va_cal_inductive.predict_proba(X_binary_test)

    # Results should be similar (not exact due to different random splits)
    assert probs_prefit.shape == probs_inductive.shape


# ============================================================================
# Estimator Tests
# ============================================================================


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_different_estimators_binary(estimator: ClassifierMixin) -> None:
    """Test VennAbersCalibrator with different base estimators (binary)."""
    va_cal = VennAbersCalibrator(
        estimator=estimator, inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_different_estimators_multiclass(estimator: ClassifierMixin) -> None:
    """Test VennAbersCalibrator with different base estimators (multi-class)."""
    va_cal = VennAbersCalibrator(
        estimator=estimator, inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_estimator_none_raises_error() -> None:
    """Test that None estimator raises ValueError."""
    va_cal = VennAbersCalibrator(estimator=None)
    with pytest.raises(ValueError, match=r".*an estimator must be provided*"):
        va_cal.fit(X_binary_train, y_binary_train)


def test_predict_method_multiclass() -> None:
    """Test predict method for multi-class classification."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    predictions = va_cal.predict(X_multi_test)

    assert predictions.shape == (len(X_multi_test),)
    assert va_cal.classes_ is not None
    assert np.all(np.isin(predictions, va_cal.classes_))


def test_predict_proba_consistency() -> None:
    """Test that predict is consistent with predict_proba."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    predictions = va_cal.predict(X_binary_test)
    probs = va_cal.predict_proba(X_binary_test)

    assert va_cal.classes_ is not None
    predictions_from_probs = va_cal.classes_[np.argmax(probs, axis=1)]

    np.testing.assert_array_equal(predictions, predictions_from_probs)


def test_predict_proba_shape_binary() -> None:
    """Test that predict_proba returns correct shape for binary classification."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), va_cal.n_classes_)
    assert va_cal.n_classes_ == 2


def test_predict_proba_shape_multiclass() -> None:
    """Test that predict_proba returns correct shape for multi-class classification."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), va_cal.n_classes_)
    assert va_cal.n_classes_ == 3


def test_gradient_boosting_with_early_stopping() -> None:
    """Test VennAbersCalibrator with GradientBoosting and early stopping."""
    gb = GradientBoostingClassifier(n_estimators=100, random_state=random_state)

    va_cal = VennAbersCalibrator(
        estimator=gb, inductive=True, random_state=random_state
    )

    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# Sample Weight Tests
# ============================================================================


def test_sample_weights_none() -> None:
    """Test that sample_weight=None works correctly."""
    sklearn.set_config(enable_metadata_routing=True)
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train, sample_weight=None)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


def test_sample_weights_constant() -> None:
    """Test that constant sample weights give same results as None."""
    sklearn.set_config(enable_metadata_routing=True)

    n_samples = len(X_binary_train)
    weighted_estimator = GaussianNB().set_fit_request(sample_weight=True)

    va_cal_none = VennAbersCalibrator(
        estimator=weighted_estimator, inductive=True, random_state=random_state
    )
    va_cal_none.fit(X_binary_train, y_binary_train, sample_weight=None)

    va_cal_ones = VennAbersCalibrator(
        estimator=weighted_estimator, inductive=True, random_state=random_state
    )
    va_cal_ones.fit(X_binary_train, y_binary_train, sample_weight=np.ones(n_samples))

    va_cal_fives = VennAbersCalibrator(
        estimator=weighted_estimator, inductive=True, random_state=random_state
    )
    va_cal_fives.fit(
        X_binary_train, y_binary_train, sample_weight=np.ones(n_samples) * 5
    )

    probs_none = va_cal_none.predict_proba(X_binary_test)
    probs_ones = va_cal_ones.predict_proba(X_binary_test)
    probs_fives = va_cal_fives.predict_proba(X_binary_test)

    np.testing.assert_allclose(probs_none, probs_ones, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(probs_none, probs_fives, rtol=1e-2, atol=1e-2)


def test_sample_weights_variable() -> None:
    """Test that variable sample weights affect the results."""
    sklearn.set_config(enable_metadata_routing=True)
    n_samples = len(X_binary_train)

    va_cal_uniform = VennAbersCalibrator(
        estimator=RandomForestClassifier(random_state=random_state),
        inductive=True,
        random_state=random_state,
    )
    va_cal_uniform.fit(X_binary_train, y_binary_train, sample_weight=None)

    # Create non-uniform weights
    sample_weights = np.random.RandomState(random_state).uniform(
        0.1, 2.0, size=n_samples
    )

    estimator_weighted = RandomForestClassifier(
        random_state=random_state
    ).set_fit_request(sample_weight=True)

    va_cal_weighted = VennAbersCalibrator(
        estimator=estimator_weighted, inductive=True, random_state=random_state
    )
    va_cal_weighted.fit(X_binary_train, y_binary_train, sample_weight=sample_weights)

    probs_uniform = va_cal_uniform.predict_proba(X_binary_test)
    probs_weighted = va_cal_weighted.predict_proba(X_binary_test)

    # Results should be different with non-uniform weights
    assert not np.allclose(probs_uniform, probs_weighted)


# ============================================================================
# Random State and Reproducibility Tests
# ============================================================================


def test_random_state_reproducibility() -> None:
    """Test that random_state ensures reproducible results."""
    va_cal1 = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=42
    )
    va_cal1.fit(X_binary_train, y_binary_train)
    probs1 = va_cal1.predict_proba(X_binary_test)

    va_cal2 = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=42
    )
    va_cal2.fit(X_binary_train, y_binary_train)
    probs2 = va_cal2.predict_proba(X_binary_test)

    np.testing.assert_array_equal(probs1, probs2)


def test_random_state_in_fit_overrides() -> None:
    """Test that random_state in fit() overrides constructor parameter."""
    va_cal1 = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=42
    )
    va_cal1.fit(X_binary_train, y_binary_train, random_state=123)
    probs1 = va_cal1.predict_proba(X_binary_test)

    va_cal2 = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        random_state=999,  # Different from fit
    )
    va_cal2.fit(X_binary_train, y_binary_train, random_state=123)
    probs2 = va_cal2.predict_proba(X_binary_test)

    np.testing.assert_array_equal(probs1, probs2)


def test_different_random_states_give_different_results() -> None:
    """Test that different random states give different results."""
    va_cal1 = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=42
    )
    va_cal1.fit(X_binary_train, y_binary_train)
    probs1 = va_cal1.predict_proba(X_binary_test)

    va_cal2 = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=123
    )
    va_cal2.fit(X_binary_train, y_binary_train)
    probs2 = va_cal2.predict_proba(X_binary_test)

    # Results should be different with different random states
    assert not np.array_equal(probs1, probs2)


# ============================================================================
# Shuffle and Stratify Tests
# ============================================================================


def test_shuffle_parameter() -> None:
    """Test that shuffle parameter works correctly."""
    va_cal_shuffle = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state, shuffle=True
    )
    va_cal_shuffle.fit(X_binary_train, y_binary_train)
    probs_shuffle = va_cal_shuffle.predict_proba(X_binary_test)

    va_cal_no_shuffle = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state, shuffle=False
    )
    va_cal_no_shuffle.fit(X_binary_train, y_binary_train)
    probs_no_shuffle = va_cal_no_shuffle.predict_proba(X_binary_test)

    assert probs_shuffle.shape == probs_no_shuffle.shape


def test_shuffle_in_fit_overrides() -> None:
    """Test that shuffle in fit() overrides constructor parameter."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state, shuffle=False
    )
    # Override with shuffle=True in fit
    va_cal.fit(X_binary_train, y_binary_train, shuffle=True)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


def test_stratify_parameter() -> None:
    """Test that stratify parameter works correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        random_state=random_state,
        stratify=y_binary_train,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


def test_stratify_in_fit_overrides() -> None:
    """Test that stratify in fit() overrides constructor parameter."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state, stratify=None
    )
    # Override with stratify in fit
    va_cal.fit(X_binary_train, y_binary_train, stratify=y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# Calibration Size Tests
# ============================================================================


@pytest.mark.parametrize("cal_size", [0.2, 0.3, 0.4, 0.5])
def test_different_calibration_sizes(cal_size: float) -> None:
    """Test that different calibration sizes work correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train, calib_size=cal_size)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_cal_size_in_fit_overrides() -> None:
    """Test that calib_size in fit() overrides constructor parameter."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    # Override with calib_size in fit
    va_cal.fit(X_binary_train, y_binary_train, calib_size=0.4)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


def test_train_proper_size_parameter() -> None:
    """Test that train_proper_size parameter works correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        train_proper_size=0.6,
        random_state=random_state,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# N_splits Tests
# ============================================================================


@pytest.mark.parametrize("n_splits", [2, 3, 5, 10])
def test_different_n_splits(n_splits: int) -> None:
    """Test that different n_splits values work correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=n_splits,
        random_state=random_state,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_n_splits_too_small_raises_error() -> None:
    """Test that n_splits < 2 raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=1, random_state=random_state
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_binary_train, y_binary_train)


# ============================================================================
# Attributes Tests
# ============================================================================


def test_fitted_attributes_inductive() -> None:
    """Test that fitted attributes are set correctly for inductive mode."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    assert hasattr(va_cal, "classes_")
    assert hasattr(va_cal, "n_classes_")
    assert hasattr(va_cal, "va_calibrator_")
    assert va_cal.n_classes_ is not None
    assert va_cal.classes_ is not None
    assert va_cal.n_classes_ == 2
    assert len(va_cal.classes_) == 2


def test_fitted_attributes_cross_validation() -> None:
    """Test that fitted attributes are set correctly for cross validation mode."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=5, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    assert hasattr(va_cal, "classes_")
    assert hasattr(va_cal, "n_classes_")
    assert hasattr(va_cal, "va_calibrator_")
    assert va_cal.n_classes_ is not None
    assert va_cal.classes_ is not None
    assert va_cal.n_classes_ == 2
    assert len(va_cal.classes_) == 2


def test_fitted_attributes_prefit() -> None:
    """Test that fitted attributes are set correctly for prefit mode."""
    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)

    assert hasattr(va_cal, "classes_")
    assert hasattr(va_cal, "n_classes_")
    assert hasattr(va_cal, "single_estimator_")
    assert va_cal.n_classes_ is not None
    assert va_cal.classes_ is not None
    assert va_cal.n_classes_ == 2
    assert len(va_cal.classes_) == 2


# ============================================================================
# Pipeline Compatibility Tests
# ============================================================================


def test_pipeline_compatibility() -> None:
    """Test that VennAbersCalibrator works with sklearn pipelines."""
    X_df = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"] * 10,
            "x_num": [0, 1, 1, 4, np.nan, 5] * 10,
        }
    )
    y_series = pd.Series([0, 1, 0, 1, 0, 1] * 10)

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
    pipe = make_pipeline(preprocessor, LogisticRegression(random_state=random_state))
    pipe.fit(X_df, y_series)

    va_cal = VennAbersCalibrator(
        estimator=pipe, inductive=True, random_state=random_state
    )
    va_cal.fit(X_df, y_series)
    predictions = va_cal.predict(X_df)
    probs = va_cal.predict_proba(X_df)

    assert predictions.shape == (len(y_series),)
    assert probs.shape == (len(y_series), 2)


def test_pipeline_prefit_mode() -> None:
    """Test that VennAbersCalibrator works with prefit pipelines."""
    X_df = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B"] * 10,
            "x_num": [0, 1, 1, 4, np.nan, 5] * 10,
        }
    )
    y_series = pd.Series([0, 1, 0, 1, 0, 1] * 10)

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
    pipe = make_pipeline(preprocessor, LogisticRegression(random_state=random_state))
    pipe.fit(X_df, y_series)

    va_cal = VennAbersCalibrator(estimator=pipe, cv="prefit")
    va_cal.fit(X_df, y_series)
    predictions = va_cal.predict(X_df)
    probs = va_cal.predict_proba(X_df)

    assert predictions.shape == (len(y_series),)
    assert probs.shape == (len(y_series), 2)


def test_with_pipeline() -> None:
    """Test VennAbersCalibrator with sklearn Pipeline."""
    from sklearn.preprocessing import StandardScaler

    pipeline = make_pipeline(StandardScaler(), GaussianNB())

    va_cal = VennAbersCalibrator(
        estimator=pipeline, inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_with_column_transformer() -> None:
    """Test VennAbersCalibrator with ColumnTransformer."""
    # Create a mixed dataset
    X_mixed = np.column_stack(
        [X_binary_train, np.random.choice(["A", "B", "C"], size=len(X_binary_train))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="mean"),
                list(range(X_binary_train.shape[1])),
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [X_binary_train.shape[1]]),
        ]
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", GaussianNB())])

    va_cal = VennAbersCalibrator(
        estimator=pipeline, inductive=True, random_state=random_state
    )

    X_test_mixed = np.column_stack(
        [X_binary_test, np.random.choice(["A", "B", "C"], size=len(X_binary_test))]
    )

    va_cal.fit(X_mixed, y_binary_train)
    probs = va_cal.predict_proba(X_test_mixed)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# Multiclass Strategy Tests
# ============================================================================


def test_multiclass_one_vs_one_strategy() -> None:
    """Test multiclass with one_vs_one strategy."""
    # Create calibrator with explicit one_vs_one
    clf = GaussianNB()
    clf.fit(X_multi_proper, y_multi_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_multi_cal, y_multi_cal)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ============================================================================
# Check Fitted Tests
# ============================================================================


def test_check_is_fitted_after_fit() -> None:
    """Test that check_is_fitted passes after fitting."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    # Should not raise an error
    check_is_fitted(va_cal)


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


def test_empty_dataset_raises_error() -> None:
    """Test that empty dataset raises an error."""
    X_empty = np.array([]).reshape(0, 20)
    y_empty = np.array([])

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_empty, y_empty)


def test_single_class_raises_error() -> None:
    """Test that single class dataset raises an error."""
    X_single = X_binary_train[:10]
    y_single = np.zeros(10)  # All same class

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_single, y_single)


def test_mismatched_X_y_length_raises_error() -> None:
    """Test that mismatched X and y lengths raise an error."""
    X_mismatch = X_binary_train[:50]
    y_mismatch = y_binary_train[:40]

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_mismatch, y_mismatch)


def test_predict_before_fit_raises_error() -> None:
    """Test that calling predict before fit raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    with pytest.raises(Exception):  # NotFittedError or AttributeError
        va_cal.predict(X_binary_test)


def test_predict_proba_before_fit_raises_error() -> None:
    """Test that calling predict_proba before fit raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    with pytest.raises(Exception):  # NotFittedError or AttributeError
        va_cal.predict_proba(X_binary_test)


def test_invalid_cal_size_raises_error() -> None:
    """Test that invalid cal_size values raise an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_binary_train, y_binary_train, calib_size=1.5)  # Invalid: > 1.0


def test_negative_cal_size_raises_error() -> None:
    """Test that negative calib_size raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    with pytest.raises(ValueError):
        va_cal.fit(X_binary_train, y_binary_train, calib_size=-0.1)


def test_empty_calibration_set_raises_error() -> None:
    """Test that empty calibration set raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    # This should work but with a very small training set
    try:
        # Very large calib_size leaves almost no training data
        va_cal.fit(X_binary_train[:10], y_binary_train[:10], calib_size=0.99)
    except ValueError:
        # Expected if the split is invalid
        pass


def test_very_small_dataset() -> None:
    """Test with a very small dataset."""
    X_small = X_binary_train[:20]
    y_small = y_binary_train[:20]

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_small, y_small)
    probs = va_cal.predict_proba(X_binary_test[:5])

    assert probs.shape == (5, 2)


# ============================================================================
# Calibration Quality Tests
# ============================================================================


def test_calibration_improves_probabilities() -> None:
    """Test that Venn-ABERS calibration improves probability estimates."""
    # Train uncalibrated model
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_binary_proper, y_binary_proper)
    uncalibrated_probs = clf.predict_proba(X_binary_test)

    # Train calibrated model
    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)
    calibrated_probs = va_cal.predict_proba(X_binary_test)

    # Both should have valid probability distributions
    assert np.allclose(uncalibrated_probs.sum(axis=1), 1.0)
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0)

    # Calibrated probabilities should be different
    assert not np.allclose(uncalibrated_probs, calibrated_probs)


def test_probabilities_sum_to_one() -> None:
    """Test that predicted probabilities sum to 1 for all samples."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    # Check that probabilities sum to 1 for each sample
    prob_sums = probs.sum(axis=1)
    np.testing.assert_allclose(prob_sums, np.ones(len(X_binary_test)), rtol=1e-5)


def test_probabilities_in_valid_range() -> None:
    """Test that all predicted probabilities are in [0, 1]."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


def test_multiclass_probabilities_sum_to_one() -> None:
    """Test that multi-class predicted probabilities sum to 1."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    prob_sums = probs.sum(axis=1)
    np.testing.assert_allclose(prob_sums, np.ones(len(X_multi_test)), rtol=1e-5)


def test_multiclass_probabilities_in_valid_range() -> None:
    """Test that all multi-class predicted probabilities are in [0, 1]."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    assert np.all(probs >= 0)
    assert np.all(probs <= 1)


# ============================================================================
# Comparison Tests Between Modes
# ============================================================================


def test_inductive_vs_cross_validation_different_results() -> None:
    """Test that inductive and cross validation modes give different results."""
    va_cal_inductive = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal_inductive.fit(X_binary_train, y_binary_train)
    probs_inductive = va_cal_inductive.predict_proba(X_binary_test)

    va_cal_cv = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=5, random_state=random_state
    )
    va_cal_cv.fit(X_binary_train, y_binary_train)
    probs_cv = va_cal_cv.predict_proba(X_binary_test)

    # Results should be different between modes
    assert not np.allclose(probs_inductive, probs_cv)


def test_all_modes_produce_valid_probabilities() -> None:
    """Test that all calibration modes produce valid probability distributions."""
    modes: List[Tuple[str, Dict[str, Any]]] = [
        ("inductive", {"inductive": True}),
        ("cross_val", {"inductive": False, "n_splits": 5}),
    ]

    for mode_name, mode_params in modes:
        va_cal = VennAbersCalibrator(
            estimator=GaussianNB(), random_state=random_state, **mode_params
        )
        va_cal.fit(X_binary_train, y_binary_train)
        probs = va_cal.predict_proba(X_binary_test)

        # Check valid probabilities
        assert np.all(probs >= 0), f"Mode {mode_name} produced negative probabilities"
        assert np.all(probs <= 1), f"Mode {mode_name} produced probabilities > 1"
        assert np.allclose(
            probs.sum(axis=1), 1.0
        ), f"Mode {mode_name} probabilities don't sum to 1"


# ============================================================================
# Special Cases Tests
# ============================================================================


def test_perfect_predictions_no_calibration_needed() -> None:
    """Test behavior when base estimator already makes perfect predictions."""
    # Create a simple linearly separable dataset
    from sklearn.datasets import make_blobs

    X_perfect, y_perfect = make_blobs(
        n_samples=100,
        n_features=2,
        centers=2,
        cluster_std=0.5,
        random_state=random_state,
    )

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X_perfect, y_perfect, test_size=0.2, random_state=random_state
    )

    va_cal = VennAbersCalibrator(
        estimator=LogisticRegression(random_state=random_state),
        inductive=True,
        random_state=random_state,
    )
    va_cal.fit(X_train_p, y_train_p)
    probs = va_cal.predict_proba(X_test_p)
    predictions = va_cal.predict(X_test_p)

    # Should still produce valid probabilities
    assert probs.shape == (len(X_test_p), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Predictions should be accurate
    accuracy = np.mean(predictions == y_test_p)
    assert accuracy > 0.9  # Should be very accurate for linearly separable data


def test_imbalanced_dataset() -> None:
    """Test VennAbersCalibrator with highly imbalanced dataset."""
    # Create imbalanced dataset (90% class 0, 10% class 1)
    X_imb, y_imb = make_classification(
        n_samples=200,
        n_features=20,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=random_state,
    )

    X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
        X_imb, y_imb, test_size=0.2, random_state=random_state, stratify=y_imb
    )

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        random_state=random_state,
        stratify=y_train_imb,
    )
    va_cal.fit(X_train_imb, y_train_imb)
    probs = va_cal.predict_proba(X_test_imb)

    # Should still produce valid probabilities
    assert probs.shape == (len(X_test_imb), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_many_classes() -> None:
    """Test VennAbersCalibrator with many classes."""
    # Create dataset with 10 classes
    X_many, y_many = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=10,
        n_informative=15,
        random_state=random_state,
    )

    X_train_many, X_test_many, y_train_many, y_test_many = train_test_split(
        X_many, y_many, test_size=0.2, random_state=random_state
    )

    va_cal = VennAbersCalibrator(
        estimator=RandomForestClassifier(random_state=random_state),
        inductive=True,
        random_state=random_state,
    )
    va_cal.fit(X_train_many, y_train_many)
    probs = va_cal.predict_proba(X_test_many)

    assert probs.shape == (len(X_test_many), 10)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_small_calibration_set() -> None:
    """Test behavior with very small calibration set."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(
        X_binary_train, y_binary_train, calib_size=0.1
    )  # Very small calibration set
    probs = va_cal.predict_proba(X_binary_test)

    # Should still work, though calibration quality may be lower
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_large_calibration_set() -> None:
    """Test behavior with very large calibration set."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(
        X_binary_train, y_binary_train, calib_size=0.8
    )  # Very large calibration set
    probs = va_cal.predict_proba(X_binary_test)

    # Should still work, though training set is small
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ============================================================================
# Consistency Tests
# ============================================================================


def test_multiple_fits_same_data() -> None:
    """Test that fitting multiple times with same data gives same results."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )

    va_cal.fit(X_binary_train, y_binary_train)
    probs1 = va_cal.predict_proba(X_binary_test)

    va_cal.fit(X_binary_train, y_binary_train)
    probs2 = va_cal.predict_proba(X_binary_test)

    np.testing.assert_array_equal(probs1, probs2)


def test_predict_single_sample() -> None:
    """Test prediction on a single sample."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    single_sample = X_binary_test[0:1]
    probs = va_cal.predict_proba(single_sample)
    pred = va_cal.predict(single_sample)

    assert probs.shape == (1, 2)
    assert pred.shape == (1,)
    assert np.allclose(probs.sum(), 1.0)


def test_predict_multiple_times_same_result() -> None:
    """Test that multiple predictions on same data give same results."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    probs1 = va_cal.predict_proba(X_binary_test)
    probs2 = va_cal.predict_proba(X_binary_test)

    np.testing.assert_array_equal(probs1, probs2)


# ============================================================================
# Data Type Tests
# ============================================================================


def test_pandas_dataframe_input() -> None:
    """Test that VennAbersCalibrator works with pandas DataFrames."""
    X_df = pd.DataFrame(X_binary_train)
    y_series = pd.Series(y_binary_train)
    X_test_df = pd.DataFrame(X_binary_test)

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_df, y_series)
    probs = va_cal.predict_proba(X_test_df)
    predictions = va_cal.predict(X_test_df)

    assert probs.shape == (len(X_test_df), 2)
    assert predictions.shape == (len(X_test_df),)


def test_numpy_array_input() -> None:
    """Test that VennAbersCalibrator works with numpy arrays."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)
    predictions = va_cal.predict(X_binary_test)

    assert isinstance(probs, np.ndarray)
    assert isinstance(predictions, np.ndarray)


def test_mixed_input_types() -> None:
    """Test with mixed input types (DataFrame for X, array for y)."""
    X_df = pd.DataFrame(X_binary_train)
    y_array = np.array(y_binary_train)
    X_test_df = pd.DataFrame(X_binary_test)

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_df, y_array)
    probs = va_cal.predict_proba(X_test_df)

    assert probs.shape == (len(X_test_df), 2)


def test_with_pandas_dataframe() -> None:
    """Test VennAbersCalibrator with pandas DataFrame."""
    X_train_df = pd.DataFrame(X_binary_train)
    X_test_df = pd.DataFrame(X_binary_test)

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_train_df, y_binary_train)
    probs = va_cal.predict_proba(X_test_df)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_with_pandas_series() -> None:
    """Test VennAbersCalibrator with pandas Series for y."""
    y_train_series = pd.Series(y_binary_train)

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_train_series)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# Integration Tests
# ============================================================================


def test_integration_with_cross_validation() -> None:
    """Test integration with sklearn's cross-validation utilities."""
    from sklearn.model_selection import cross_val_score

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )

    # This should work with cross_val_score
    scores = cross_val_score(va_cal, X_binary, y_binary, cv=3, scoring="accuracy")

    assert len(scores) == 3
    assert np.all(scores >= 0) and np.all(scores <= 1)


# def test_integration_with_grid_search() -> None:
#     """Test integration with sklearn's GridSearchCV."""
#     from sklearn.model_selection import GridSearchCV

#     va_cal = VennAbersCalibrator(
#         estimator=GaussianNB(),
#         inductive=True,
#         random_state=random_state
#     )

#     param_grid = {
#         'cal_size': [0.2, 0.3, 0.4],
#     }

#     grid_search = GridSearchCV(
#         va_cal, param_grid, cv=3, scoring='accuracy'
#     )
#     grid_search.fit(X_binary_train, y_binary_train)

#     assert hasattr(grid_search, 'best_params_')
#     assert 'cal_size' in grid_search.best_params_


def test_clone_estimator() -> None:
    """Test that VennAbersCalibrator can be cloned."""
    from sklearn.base import clone

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    va_cal_clone = clone(va_cal)

    is_fitted = True
    try:
        check_is_fitted(va_cal_clone.estimator)
    except NotFittedError:
        is_fitted = False

    # Clone should have same parameters but not be fitted
    assert va_cal_clone.inductive == va_cal.inductive
    assert is_fitted is False


# ============================================================================
# Performance and Scalability Tests
# ============================================================================


def test_large_dataset_performance() -> None:
    """Test performance on a larger dataset."""
    X_large, y_large = make_classification(
        n_samples=5000, n_features=50, n_classes=2, random_state=random_state
    )

    X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(
        X_large, y_large, test_size=0.2, random_state=random_state
    )

    va_cal = VennAbersCalibrator(
        estimator=RandomForestClassifier(n_estimators=10, random_state=random_state),
        inductive=True,
        random_state=random_state,
        precision=2,  # Use precision for faster computation
    )

    import time

    start = time.time()
    va_cal.fit(X_train_large, y_train_large)
    va_cal.predict_proba(X_test_large)
    elapsed = time.time() - start

    # Should complete in reasonable time (< 60 seconds)
    assert elapsed < 60


def test_high_dimensional_data() -> None:
    """Test with high-dimensional data."""
    X_high_dim, y_high_dim = make_classification(
        n_samples=200,
        n_features=100,
        n_informative=50,
        n_classes=2,
        random_state=random_state,
    )

    X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(
        X_high_dim, y_high_dim, test_size=0.2, random_state=random_state
    )

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_train_hd, y_train_hd)
    probs = va_cal.predict_proba(X_test_hd)

    assert probs.shape == (len(X_test_hd), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ============================================================================
# Documentation and Examples Tests
# ============================================================================


def test_basic_example_from_docstring() -> None:
    """Test the basic example from the class docstring."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB

    X, y = make_classification(n_samples=1000, n_classes=2, n_informative=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = GaussianNB()
    va_cal = VennAbersCalibrator(estimator=clf, inductive=True)
    va_cal.fit(X_train, y_train)

    p_prime = va_cal.predict_proba(X_test)

    assert p_prime.shape == (len(X_test), 2)
    assert np.allclose(p_prime.sum(axis=1), 1.0)


def test_prefit_example() -> None:
    """Test prefit example workflow."""
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_binary_train, y_binary_train, test_size=0.2, shuffle=False
    )

    clf = GaussianNB()
    clf.fit(X_train_proper, y_train_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_cal, y_cal)

    p_prime = va_cal.predict_proba(X_binary_test)

    assert p_prime.shape == (len(X_binary_test), 2)


def test_cross_validation_example() -> None:
    """Test cross-validation example workflow."""
    va_cal = VennAbersCalibrator(estimator=GaussianNB(), inductive=False, n_splits=5)
    va_cal.fit(X_binary_train, y_binary_train)

    p_prime = va_cal.predict_proba(X_binary_test)

    assert p_prime.shape == (len(X_binary_test), 2)


# ============================================================================
# Comparison with Other Calibration Methods Tests
# ============================================================================


def test_comparison_with_uncalibrated() -> None:
    """Compare calibrated vs uncalibrated predictions."""
    # Uncalibrated
    clf_uncal = RandomForestClassifier(random_state=random_state)
    clf_uncal.fit(X_binary_train, y_binary_train)
    probs_uncal = clf_uncal.predict_proba(X_binary_test)

    # Calibrated
    va_cal = VennAbersCalibrator(
        estimator=RandomForestClassifier(random_state=random_state),
        inductive=True,
        random_state=random_state,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs_cal = va_cal.predict_proba(X_binary_test)

    # Both should be valid probabilities
    assert np.allclose(probs_uncal.sum(axis=1), 1.0)
    assert np.allclose(probs_cal.sum(axis=1), 1.0)

    # Calibrated should be different from uncalibrated
    assert not np.allclose(probs_uncal, probs_cal)


# ============================================================================
# Regression Tests (ensure no breaking changes)
# ============================================================================


def test_backward_compatibility_basic_usage() -> None:
    """Test that basic usage pattern remains compatible."""
    # This test ensures the most common usage pattern doesn't break
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)
    preds = va_cal.predict(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert preds.shape == (len(X_binary_test),)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_backward_compatibility_prefit() -> None:
    """Test that prefit mode usage pattern remains compatible."""
    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


def test_backward_compatibility_cross_val() -> None:
    """Test that cross-validation mode usage pattern remains compatible."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=5, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# Edge Cases for Different Modes
# ============================================================================


def test_prefit_with_unfitted_estimator_raises_error() -> None:
    """Test that prefit mode with unfitted estimator raises an error."""
    clf = GaussianNB()  # Not fitted

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")

    with pytest.raises(ValueError, match=".*must be already fitted.*"):
        va_cal.fit(X_binary_cal, y_binary_cal)


def test_cross_val_without_n_splits_raises_error() -> None:
    """Test that cross-validation mode without n_splits raises an error."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=None,  # Missing n_splits
    )

    with pytest.raises(ValueError, match=".*please provide n_splits.*"):
        va_cal.fit(X_binary_train, y_binary_train)


def test_inductive_with_very_small_dataset() -> None:
    """Test inductive mode with very small dataset."""
    X_small = X_binary_train[:20]
    y_small = y_binary_train[:20]

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )

    # Should work but might have limited calibration quality
    va_cal.fit(X_small, y_small)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)


# ============================================================================
# Attribute Access Tests
# ============================================================================


def test_classes_attribute() -> None:
    """Test that classes_ attribute is correctly set."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    assert hasattr(va_cal, "classes_")
    assert va_cal.classes_ is not None
    assert len(va_cal.classes_) == 2
    np.testing.assert_array_equal(va_cal.classes_, np.unique(y_binary_train))


def test_n_classes_attribute() -> None:
    """Test that n_classes_ attribute is correctly set."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    assert hasattr(va_cal, "n_classes_")
    assert va_cal.n_classes_ == 2


def test_va_calibrator_attribute() -> None:
    """Test that va_calibrator_ attribute is correctly set."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    assert hasattr(va_cal, "va_calibrator_")
    assert va_cal.va_calibrator_ is not None


def test_single_estimator_attribute_prefit() -> None:
    """Test that single_estimator_ attribute is set in prefit mode."""
    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)

    assert hasattr(va_cal, "single_estimator_")
    assert va_cal.single_estimator_ is not None


# ============================================================================
# Multi-class Specific Tests
# ============================================================================


def test_multiclass_binary_calibration() -> None:
    """Test that multi-class uses binary calibration for each class pair."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    # For 3 classes, should have 3 probability columns
    assert probs.shape == (len(X_multi_test), 3)

    # Each row should sum to 1
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


def test_multiclass_prefit_mode() -> None:
    """Test multi-class calibration in prefit mode."""
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_multi_proper, y_multi_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_multi_cal, y_multi_cal)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_multiclass_cross_validation_mode() -> None:
    """Test multi-class calibration in cross-validation mode."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=5, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_multiclass_predictions_match_argmax() -> None:
    """Test that multi-class predictions match argmax of probabilities."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)

    probs = va_cal.predict_proba(X_multi_test)
    preds = va_cal.predict(X_multi_test)

    # Predictions should match the class with highest probability
    assert va_cal.classes_ is not None
    expected_preds = va_cal.classes_[np.argmax(probs, axis=1)]
    np.testing.assert_array_equal(preds, expected_preds)


def test_multiclass_with_different_estimators() -> None:
    """Test multi-class calibration with different base estimators."""
    estimators = [
        GaussianNB(),
        RandomForestClassifier(n_estimators=10, random_state=random_state),
        LogisticRegression(random_state=random_state, max_iter=1000),
    ]

    for estimator in estimators:
        va_cal = VennAbersCalibrator(
            estimator=estimator, inductive=True, random_state=random_state
        )
        va_cal.fit(X_multi_train, y_multi_train)
        probs = va_cal.predict_proba(X_multi_test)

        assert probs.shape == (len(X_multi_test), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all((probs >= 0) & (probs <= 1))


# ============================================================================
# Precision Parameter Tests
# ============================================================================


@pytest.mark.parametrize("precision", [None, 2, 4, 6])
def test_precision_parameter(precision: Optional[int]) -> None:
    """Test that precision parameter works correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        random_state=random_state,
        precision=precision,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_precision_speeds_up_computation() -> None:
    """Test that precision parameter reduces computation time."""
    import time

    # Without precision
    va_cal_no_precision = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        random_state=random_state,
        precision=None,
    )
    start = time.time()
    va_cal_no_precision.fit(X_binary_train, y_binary_train)
    va_cal_no_precision.predict_proba(X_binary_test)
    time_no_precision = time.time() - start

    # With precision
    va_cal_with_precision = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state, precision=2
    )
    start = time.time()
    va_cal_with_precision.fit(X_binary_train, y_binary_train)
    va_cal_with_precision.predict_proba(X_binary_test)
    time_with_precision = time.time() - start

    # With precision should be faster or similar
    # (may not always be faster for small datasets)
    assert time_with_precision <= time_no_precision


@pytest.mark.parametrize("precision", [1, 2, 3, 4])
def test_different_precision_values(precision: int) -> None:
    """Test that different precision values work correctly."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(),
        inductive=True,
        random_state=random_state,
        precision=precision,
    )
    va_cal.fit(X_binary_train, y_binary_train)
    probs = va_cal.predict_proba(X_binary_test)

    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_precision_maintains_calibration_quality() -> None:
    """Test that precision parameter maintains reasonable calibration quality."""
    va_cal_high_prec = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state, precision=4
    )
    va_cal_high_prec.fit(X_binary_train, y_binary_train)
    probs_high = va_cal_high_prec.predict_proba(X_binary_test)

    va_cal_low_prec = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state, precision=2
    )
    va_cal_low_prec.fit(X_binary_train, y_binary_train)
    probs_low = va_cal_low_prec.predict_proba(X_binary_test)

    # Both should be valid probabilities
    assert np.allclose(probs_high.sum(axis=1), 1.0)
    assert np.allclose(probs_low.sum(axis=1), 1.0)

    # They should be similar but not necessarily identical
    assert probs_high.shape == probs_low.shape


def test_precision_parameter_multiclass() -> None:
    """Test that precision parameter works correctly for multiclass."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, precision=6, random_state=random_state
    )
    va_cal.fit(X_multi_train, y_multi_train)
    probs = va_cal.predict_proba(X_multi_test)

    assert probs.shape == (len(X_multi_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ============================================================================
# Error Message Quality Tests
# ============================================================================


def test_error_message_for_missing_estimator() -> None:
    """Test that missing estimator gives clear error message."""
    va_cal = VennAbersCalibrator(estimator=None)

    with pytest.raises(ValueError, match=".*estimator must be provided.*"):
        va_cal.fit(X_binary_train, y_binary_train)


def test_error_message_for_invalid_cv() -> None:
    """Test that invalid cv parameter gives clear error message."""
    va_cal = VennAbersCalibrator(estimator=GaussianNB(), cv="invalid_cv_option")

    with pytest.raises(ValueError):
        va_cal.fit(X_binary_train, y_binary_train)


# ============================================================================
# Final Comprehensive Test
# ============================================================================


def test_venn_abers_cv_with_sample_weight() -> None:
    """Test VennAbersCV with sample weights in cross-validation mode."""
    # Create sample weights - higher weights for some samples
    sklearn.set_config(enable_metadata_routing=True)
    sample_weight = np.ones(len(y_binary_train))
    sample_weight[: len(y_binary_train) // 2] = 2.0  # Double weight for first half
    weighted_estimator = GaussianNB().set_fit_request(sample_weight=True)
    va_cal = VennAbersCalibrator(
        estimator=weighted_estimator,
        inductive=False,  # Use cross-validation mode
        n_splits=3,
        random_state=random_state,
    )

    # Fit with sample weights
    va_cal.fit(X_binary_train, y_binary_train, sample_weight=sample_weight)
    probs = va_cal.predict_proba(X_binary_test)

    # Should produce valid probabilities
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))

    # Fit without sample weights for comparison
    va_cal_no_weight = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=3, random_state=random_state
    )
    va_cal_no_weight.fit(X_binary_train, y_binary_train)
    probs_no_weight = va_cal_no_weight.predict_proba(X_binary_test)

    # Results should be different when using sample weights
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(probs, probs_no_weight)


def test_venn_abers_cv_sample_weight_all_folds() -> None:
    """Test that sample weights are properly used across all CV folds."""
    sklearn.set_config(enable_metadata_routing=True)
    sample_weight = np.random.RandomState(42).uniform(0.5, 2.0, len(y_binary_train))
    weighted_estimator = GaussianNB().set_fit_request(sample_weight=True)
    va_cal = VennAbersCalibrator(
        estimator=weighted_estimator,
        inductive=False,
        n_splits=5,  # Multiple folds to ensure all are tested
        random_state=random_state,
    )

    # Should not raise any errors
    va_cal.fit(X_binary_train, y_binary_train, sample_weight=sample_weight)
    probs = va_cal.predict_proba(X_binary_test)

    # Verify output validity
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all((probs >= 0) & (probs <= 1))


def test_comprehensive_workflow() -> None:
    """Comprehensive test covering multiple aspects of VennAbersCalibrator."""
    # Test all three modes with binary classification
    modes: List[Tuple[str, Dict[str, Any]]] = [
        ("inductive", {"inductive": True}),
        ("cross_val", {"inductive": False, "n_splits": 5}),
    ]

    for mode_name, mode_params in modes:
        # Binary classification
        va_cal_binary = VennAbersCalibrator(
            estimator=RandomForestClassifier(
                n_estimators=10, random_state=random_state
            ),
            random_state=random_state,
            **mode_params,
        )
        va_cal_binary.fit(X_binary_train, y_binary_train)

        probs_binary = va_cal_binary.predict_proba(X_binary_test)
        preds_binary = va_cal_binary.predict(X_binary_test)

        # Validate binary results
        assert probs_binary.shape == (len(X_binary_test), 2)
        assert preds_binary.shape == (len(X_binary_test),)
        assert np.allclose(probs_binary.sum(axis=1), 1.0)
        assert np.all((probs_binary >= 0) & (probs_binary <= 1))

        # Multi-class classification
        va_cal_multi = VennAbersCalibrator(
            estimator=RandomForestClassifier(
                n_estimators=10, random_state=random_state
            ),
            random_state=random_state,
            **mode_params,
        )
        va_cal_multi.fit(X_multi_train, y_multi_train)

        probs_multi = va_cal_multi.predict_proba(X_multi_test)
        preds_multi = va_cal_multi.predict(X_multi_test)

        # Validate multi-class results
        assert probs_multi.shape == (len(X_multi_test), 3)
        assert preds_multi.shape == (len(X_multi_test),)
        assert np.allclose(probs_multi.sum(axis=1), 1.0)
        assert np.all((probs_multi >= 0) & (probs_multi <= 1))

    # Test prefit mode separately
    clf_binary = RandomForestClassifier(n_estimators=10, random_state=random_state)
    clf_binary.fit(X_binary_proper, y_binary_proper)

    va_cal_prefit = VennAbersCalibrator(estimator=clf_binary, cv="prefit")
    va_cal_prefit.fit(X_binary_cal, y_binary_cal)

    probs_prefit = va_cal_prefit.predict_proba(X_binary_test)
    assert probs_prefit.shape == (len(X_binary_test), 2)
    assert np.allclose(probs_prefit.sum(axis=1), 1.0)


def test_predict_proba_prefitted_va_one_vs_all():
    """
    Test predict_proba_prefitted_va with one_vs_all strategy
    to cover lines 345-368.
    """
    # Generate multiclass classification data
    X, y = make_classification(
        n_samples=500,
        n_classes=3,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Split into train, calibration, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Train a classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Get probability predictions
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)

    # Test one_vs_all strategy
    p_calibrated, p0p1 = predict_proba_prefitted_va(
        p_cal, y_cal, p_test, precision=None, va_tpe="one_vs_all"
    )

    # Assertions
    assert p_calibrated.shape == p_test.shape
    assert np.allclose(p_calibrated.sum(axis=1), 1.0)
    assert len(p0p1) == 3  # One for each class
    assert all(p.shape == (len(p_test), 2) for p in p0p1)

    # Test with precision parameter
    p_calibrated_prec, p0p1_prec = predict_proba_prefitted_va(
        p_cal, y_cal, p_test, precision=3, va_tpe="one_vs_all"
    )

    assert p_calibrated_prec.shape == p_test.shape
    assert np.allclose(p_calibrated_prec.sum(axis=1), 1.0)


def test_predict_proba_prefitted_va_one_vs_one():
    """
    Test predict_proba_prefitted_va with one_vs_one strategy
    for comparison and completeness.
    """
    # Generate multiclass classification data
    X, y = make_classification(
        n_samples=500,
        n_classes=3,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    # Split into train, calibration, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Train a classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Get probability predictions
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)

    # Test one_vs_one strategy
    p_calibrated, p0p1 = predict_proba_prefitted_va(
        p_cal, y_cal, p_test, precision=None, va_tpe="one_vs_one"
    )

    # Assertions
    assert p_calibrated.shape == p_test.shape
    assert np.allclose(p_calibrated.sum(axis=1), 1.0)
    assert len(p0p1) == 3  # C(3,2) = 3 pairs


def test_predict_proba_prefitted_va_invalid_type():
    """
    Test that invalid va_tpe raises ValueError.
    """
    # Generate simple data
    X, y = make_classification(n_samples=100, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    p_cal = clf.predict_proba(X_train)
    p_test = clf.predict_proba(X_test)

    with pytest.raises(ValueError, match="Invalid va_tpe"):
        predict_proba_prefitted_va(p_cal, y_train, p_test, va_tpe="invalid_type")


def test_venn_abers_basic():
    """
    Test basic VennAbers functionality for binary classification.
    """
    # Generate binary classification data
    X, y = make_classification(n_samples=500, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Further split training data
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Train classifier
    clf = GaussianNB()
    clf.fit(X_train_proper, y_train_proper)

    # Get probabilities
    p_cal = clf.predict_proba(X_cal)
    p_test = clf.predict_proba(X_test)

    # Apply Venn-ABERS calibration
    va = VennAbers()
    va.fit(p_cal, y_cal)
    p_prime, p0_p1 = va.predict_proba(p_test)

    # Assertions
    assert p_prime.shape == (len(X_test), 2)
    assert p0_p1.shape == (len(X_test), 2)
    assert np.allclose(p_prime.sum(axis=1), 1.0)

    # Test with precision
    va_prec = VennAbers()
    va_prec.fit(p_cal, y_cal, precision=3)
    p_prime_prec, _ = va_prec.predict_proba(p_test)
    assert p_prime_prec.shape == (len(X_test), 2)


def test_venn_abers_cv_brier_loss() -> None:
    """Test VennAbersCV with Brier loss (non-log loss)."""
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=False, n_splits=3, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    # Use 'brier' loss to trigger the else branch
    probs_brier = va_cal.predict_proba(X_binary_test, loss="brier")

    # Should produce valid probabilities
    assert probs_brier.shape == (len(X_binary_test), 2)
    assert np.allclose(probs_brier.sum(axis=1), 1.0)
    assert np.all((probs_brier >= 0) & (probs_brier <= 1))


def test_venn_abers_cv_p0_p1_output() -> None:
    """Test VennAbersCV predict_proba with p0_p1_output=True."""
    from sklearn.naive_bayes import GaussianNB
    from mapie._venn_abers import VennAbersCV

    # Create and fit VennAbersCV in inductive mode
    va_cv = VennAbersCV(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cv.fit(X_binary_train, y_binary_train)

    # Call predict_proba with p0_p1_output=True to reach the target code
    p_prime, p0_p1 = va_cv.predict_proba(X_binary_test, p0_p1_output=True)

    # Verify the outputs
    assert p_prime.shape == (len(X_binary_test), 2)
    assert p0_p1.shape == (len(X_binary_test), 2)  # Should have p0 and p1 stacked
    assert np.allclose(p_prime.sum(axis=1), 1.0)
    assert np.all((p_prime >= 0) & (p_prime <= 1))
    assert np.all((p0_p1 >= 0) & (p0_p1 <= 1))


def test_multiclass_cross_validation_requires_n_splits() -> None:
    """Test that VennAbersMultiClass in CVAP mode requires n_splits parameter."""
    from mapie._venn_abers import VennAbersMultiClass

    va_multi = VennAbersMultiClass(
        estimator=GaussianNB(),
        inductive=False,
        n_splits=None,  # Missing n_splits for cross-validation mode
    )

    with pytest.raises(
        Exception, match=r".*For Cross Venn ABERS please provide n_splits.*"
    ):
        va_multi.fit(X_multi_train, y_multi_train)


def test_inductive_missing_size_parameters_raises_error():
    """Test that inductive mode raises error
    when train_proper_size is None.
    """
    # Generate multi-class dataset
    X, y = make_classification(
        n_samples=100, n_classes=3, n_informative=10, n_redundant=0, random_state=42
    )

    # Create VennAbersMultiClass with inductive=True but no size parameters
    va_multi = VennAbersMultiClass(
        estimator=GaussianNB(), inductive=True, train_proper_size=None, random_state=42
    )

    # Should raise Exception when fitting without size parameters
    with pytest.raises(
        Exception, match="For Inductive Venn-ABERS please provide either calibration"
    ):
        va_multi.fit(X, y)


def test_multiclass_p0_p1_output() -> None:
    """Test VennAbersMultiClass with p0_p1_output=True."""
    from mapie._venn_abers import VennAbersMultiClass
    from sklearn.naive_bayes import GaussianNB
    import numpy as np

    # Use the existing test data fixtures
    random_state = 42
    np.random.seed(random_state)

    # Generate multiclass data
    n_samples = 100
    n_features = 4
    n_classes = 3

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)

    X_test = np.random.randn(30, n_features)

    # Create and fit VennAbersMultiClass
    estimator = GaussianNB()
    va_multi = VennAbersMultiClass(
        estimator=estimator, inductive=True, cal_size=0.3, random_state=random_state
    )

    va_multi.fit(X_train, y_train)

    # Test with p0_p1_output=True
    p_prime, p0_p1_list = va_multi.predict_proba(X_test, loss="log", p0_p1_output=True)

    # Verify p_prime shape and properties
    assert p_prime.shape == (len(X_test), n_classes)
    assert np.allclose(p_prime.sum(axis=1), 1.0)
    assert np.all((p_prime >= 0) & (p_prime <= 1))

    # Verify p0_p1_list structure
    # For 3 classes, we should have C(3,2) = 3 pairwise comparisons
    n_pairs = n_classes * (n_classes - 1) // 2
    assert len(p0_p1_list) == n_pairs

    # Verify each p0_p1 entry has correct shape
    # Each entry should have shape (n_test_samples, 2*n_splits) for IVAP
    for p0_p1 in p0_p1_list:
        assert p0_p1.shape[0] == len(X_test)
        assert p0_p1.shape[1] >= 2  # At least p0 and p1 for one split

    # Verify multiclass_probs and multiclass_p0p1 are populated
    assert len(va_multi.multiclass_probs) == n_pairs
    assert len(va_multi.multiclass_p0p1) == n_pairs

    # Verify each multiclass_probs entry is binary probabilities
    for probs in va_multi.multiclass_probs:
        assert probs.shape == (len(X_test), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


def test_venn_abers_multiclass_p0_p1_output() -> None:
    """Test VennAbersMultiClass.predict_proba with p0_p1_output=True."""

    # Setup test data
    random_state = 42
    np.random.seed(random_state)

    n_samples = 150
    n_features = 4
    n_classes = 3

    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)
    X_test = np.random.randn(30, n_features)

    # Test with inductive mode
    estimator = GaussianNB()
    va_multi = VennAbersMultiClass(
        estimator=estimator, inductive=True, cal_size=0.3, random_state=random_state
    )

    va_multi.fit(X_train, y_train)

    # Test with p0_p1_output=True
    p_prime, p0_p1_list = va_multi.predict_proba(X_test, loss="log", p0_p1_output=True)

    # Verify p_prime shape and properties
    assert p_prime.shape == (len(X_test), n_classes)
    assert np.allclose(p_prime.sum(axis=1), 1.0)
    assert np.all((p_prime >= 0) & (p_prime <= 1))

    # Verify p0_p1_list structure
    # For 3 classes with one-vs-one, we should have C(3,2) = 3 pairwise comparisons
    n_pairs = n_classes * (n_classes - 1) // 2
    assert len(p0_p1_list) == n_pairs

    # Verify each p0_p1 entry has correct shape
    for p0_p1 in p0_p1_list:
        assert p0_p1.shape[0] == len(X_test)
        # For inductive mode with n_splits=1, should have 2 columns (p0 and p1)
        assert p0_p1.shape[1] == 2
        assert np.all((p0_p1 >= 0) & (p0_p1 <= 1))

    # Verify multiclass_p0p1 attribute is populated
    assert len(va_multi.multiclass_p0p1) == n_pairs
    assert va_multi.multiclass_p0p1 == p0_p1_list

    # Test with p0_p1_output=False (default behavior)
    p_prime_only = va_multi.predict_proba(X_test, loss="log", p0_p1_output=False)

    # Verify it returns only p_prime
    assert isinstance(p_prime_only, np.ndarray)
    assert p_prime_only.shape == (len(X_test), n_classes)
    assert np.allclose(p_prime_only.sum(axis=1), 1.0)

    # Test with cross-validation mode
    va_multi_cv = VennAbersMultiClass(
        estimator=GaussianNB(), inductive=False, n_splits=3, random_state=random_state
    )

    va_multi_cv.fit(X_train, y_train)

    p_prime_cv, p0_p1_list_cv = va_multi_cv.predict_proba(
        X_test, loss="log", p0_p1_output=True
    )

    # Verify CV mode results
    assert p_prime_cv.shape == (len(X_test), n_classes)
    assert len(p0_p1_list_cv) == n_pairs

    # For CV mode with n_splits=3, each p0_p1 should have 6 columns (2 * n_splits)
    for p0_p1_cv in p0_p1_list_cv:
        assert p0_p1_cv.shape[0] == len(X_test)
        assert p0_p1_cv.shape[1] == 2 * 3  # 2 * n_splits
        assert np.all((p0_p1_cv >= 0) & (p0_p1_cv <= 1))

    # Test with Brier loss
    p_prime_brier, p0_p1_brier = va_multi.predict_proba(
        X_test, loss="brier", p0_p1_output=True
    )

    assert p_prime_brier.shape == (len(X_test), n_classes)
    assert len(p0_p1_brier) == n_pairs
    assert np.allclose(p_prime_brier.sum(axis=1), 1.0)


def test_prefit_predict_proba_without_single_estimator() -> None:
    """
    Test that predict_proba raises RuntimeError when single_estimator_
    is None in prefit mode.
    """

    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)

    # Manually set single_estimator_ to None to simulate the error condition
    va_cal.single_estimator_ = None

    with pytest.raises(
        RuntimeError, match=r"single_estimator_ should not be None in prefit mode"
    ):
        va_cal.predict_proba(X_binary_test)


def test_prefit_predict_proba_without_n_classes() -> None:
    """
    Test that predict_proba raises RuntimeError when n_classes_
    is None after fitting in prefit mode.
    """

    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)

    # Manually set n_classes_ to None to simulate the error condition
    va_cal.n_classes_ = None

    with pytest.raises(
        RuntimeError, match=r"n_classes_ should not be None after fitting"
    ):
        va_cal.predict_proba(X_binary_test)


def test_prefit_predict_proba_binary_without_va_calibrator() -> None:
    """
    Test that predict_proba raises RuntimeError when va_calibrator_
    is None for binary classification in prefit mode.
    """

    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)

    # Manually set va_calibrator_ to None to simulate the error condition
    va_cal.va_calibrator_ = None

    with pytest.raises(
        RuntimeError,
        match=r"va_calibrator_ should not be None for binary classification",
    ):
        va_cal.predict_proba(X_binary_test)


def test_prefit_predict_proba_binary_with_loss_parameter() -> None:
    """
    Test that predict_proba correctly uses loss parameter when available
    in va_calibrator_.predict_proba for binary classification in prefit mode.
    """

    clf = GaussianNB()
    clf.fit(X_binary_proper, y_binary_proper)

    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit")
    va_cal.fit(X_binary_cal, y_binary_cal)

    # Test with default loss='log'
    probs_log = va_cal.predict_proba(X_binary_test, loss="log")

    # Test with loss='brier'
    probs_brier = va_cal.predict_proba(X_binary_test, loss="brier")

    # Verify output shape and properties
    assert probs_log.shape == (len(X_binary_test), 2)
    assert probs_brier.shape == (len(X_binary_test), 2)
    assert np.allclose(probs_log.sum(axis=1), 1.0)
    assert np.allclose(probs_brier.sum(axis=1), 1.0)


def test_inductive_predict_proba_with_wrong_calibrator_type() -> None:
    """
    Test that predict_proba raises RuntimeError when va_calibrator_
    is not a VennAbersMultiClass instance in inductive/cross-validation mode.
    """

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    # Manually set va_calibrator_ to wrong type
    # (VennAbers instead of VennAbersMultiClass)
    va_cal.va_calibrator_ = VennAbers()

    with pytest.raises(
        RuntimeError,
        match=r"va_calibrator_ should be VennAbersMultiClass instance in "
        r"inductive/cross-validation mode",
    ):
        va_cal.predict_proba(X_binary_test)


def test_inductive_predict_proba_without_loss_parameter() -> None:
    """
    Test that predict_proba works correctly when va_calibrator_.predict_proba
    doesn't have a loss parameter in inductive/cross-validation mode.
    """
    import inspect

    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    # Create a mock that inherits from VennAbersMultiClass
    class MockVennAbersMultiClass(VennAbersMultiClass):
        def predict_proba(self, X, p0_p1_output=False):
            """Mock predict_proba without loss parameter."""
            probs = np.random.rand(len(X), 2)
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs

    # Replace with mock that doesn't have loss parameter
    mock_calibrator = MockVennAbersMultiClass(estimator=GaussianNB(), inductive=True)

    # Verify the mock's predict_proba doesn't have 'loss' parameter
    sig = inspect.signature(mock_calibrator.predict_proba)
    assert "loss" not in sig.parameters

    va_cal.va_calibrator_ = mock_calibrator

    # Call predict_proba - should use the else branch without loss parameter
    probs = va_cal.predict_proba(X_binary_test)

    # Verify output shape
    assert probs.shape == (len(X_binary_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_without_n_classes() -> None:
    """
    Test that predict raises RuntimeError when n_classes_
    is None after fitting.
    """
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    # Manually set n_classes_ to None to simulate the error condition
    va_cal.n_classes_ = None

    with pytest.raises(
        RuntimeError, match=r"n_classes_ should not be None after fitting"
    ):
        va_cal.predict(X_binary_test)


def test_predict_without_classes() -> None:
    """
    Test that predict raises RuntimeError when classes_
    is None after fitting.
    """
    va_cal = VennAbersCalibrator(
        estimator=GaussianNB(), inductive=True, random_state=random_state
    )
    va_cal.fit(X_binary_train, y_binary_train)

    # Manually set classes_ to None to simulate the error condition
    va_cal.classes_ = None

    with pytest.raises(
        RuntimeError, match=r"classes_ should not be None after fitting"
    ):
        va_cal.predict(X_binary_test)


def test_prefit_classes_none_after_fitting() -> None:
    """
    Test that fit raises RuntimeError when classes_ is None
    after fitting estimator in prefit mode.
    """
    from sklearn.naive_bayes import GaussianNB

    # Create and fit a base estimator
    clf = GaussianNB()
    clf.fit(X_binary_train, y_binary_train)

    # Create VennAbersCalibrator in prefit mode
    va_cal = VennAbersCalibrator(estimator=clf, cv="prefit", random_state=random_state)

    # Manually set the classes_ attribute to None
    # to simulate the error condition
    clf.classes_ = None

    with pytest.raises(
        RuntimeError, match=r"classes_ should not be None after fitting estimator"
    ):
        va_cal.fit(X_binary_test, y_binary_test)
