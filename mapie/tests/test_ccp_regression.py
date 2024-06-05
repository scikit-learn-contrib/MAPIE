from __future__ import annotations

import warnings
from inspect import signature
from typing import Any, Tuple, cast

import numpy as np
import pytest
from sklearn.base import RegressorMixin, clone
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (KFold, LeaveOneOut, LeavePOut,
                                     PredefinedSplit, RepeatedKFold,
                                     ShuffleSplit, TimeSeriesSplit,
                                     train_test_split)
from sklearn.pipeline import make_pipeline

from mapie._typing import NDArray
from mapie.conformity_scores import (AbsoluteConformityScore, ConformityScore,
                                     GammaConformityScore,
                                     ResidualNormalisedScore)
from mapie.metrics import regression_coverage_score
from mapie.regression import MapieCCPRegressor
from mapie.regression.utils import (PhiFunction, CustomPhiFunction,
                                    GaussianPhiFunction, PolynomialPhiFunction)

random_state = 1
np.random.seed(random_state)

X_toy = np.linspace(0, 10, num=100).reshape(-1, 1)
y_toy = 2*X_toy[:, 0] + (max(X_toy)/10)*np.random.rand(len(X_toy))
z_toy = np.linspace(0, 10, num=len(X_toy)).reshape(-1, 1)

X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
z = X[:, -2:]


CV = ["prefit", "split"]

PHI = [
    CustomPhiFunction([lambda X: np.ones((len(X), 1))]),
    PolynomialPhiFunction(),
    GaussianPhiFunction(5),
]
WIDTHS = {
    "split": 3.87,
    "prefit": 4.81,
}

COVERAGES = {
    "split": 0.952,
    "prefit": 0.980,
}


# ======== MapieCCPRegressor =========
def test_initialized() -> None:
    """Test that initialization does not crash."""
    MapieCCPRegressor(alpha=0.1)


def test_fit() -> None:
    """Test that fit raises no errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    mapie_reg.fit_estimator(X_toy, y_toy)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_calibrate(z: Any) -> None:
    """Test that fit-calibrate raises no errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    mapie_reg.fit_estimator(X_toy, y_toy)
    mapie_reg.fit_calibrator(X_toy, y_toy, z=z)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_calibrate_combined(z: Any) -> None:
    """Test that fit_calibrate raises no errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    mapie_reg.fit(X_toy, y_toy, z=z)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_calibrate_predict(z: Any) -> None:
    """Test that fit-calibrate-predict raises no errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    mapie_reg.fit_estimator(X_toy, y_toy)
    mapie_reg.fit_calibrator(X_toy, y_toy, z=z)
    mapie_reg.predict(X_toy, z=z)


@pytest.mark.parametrize("z", [None, z_toy])
def test_fit_calibrate_combined_predict(z: Any) -> None:
    """Test that fit_calibrate-predict raises no errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    mapie_reg.fit(X_toy, y_toy, z=z)
    mapie_reg.predict(X_toy, z=z)


def test_no_fit_calibrate() -> None:
    """Test that calibrate before fit raises errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    with pytest.raises(NotFittedError):
        mapie_reg.fit_calibrator(X_toy, y_toy)


def test_calib_not_complete_phi() -> None:
    """Test that a not complete phi definition raises a warning"""
    with pytest.warns(UserWarning, match="WARNING: At least one row of the"):
        mapie_reg = MapieCCPRegressor(
            alpha=0.1,
            phi=CustomPhiFunction([lambda X: (X < 5).astype(int)],
                                  marginal_guarantee=False)
        )
        mapie_reg.fit(X_toy, y_toy)


def test_predict_not_complete_phi() -> None:
    """Test that a not complete phi definition raises a warning"""
    with pytest.warns(UserWarning, match="WARNING: At least one row of the"):
        mapie_reg = MapieCCPRegressor(
            alpha=0.1,
            phi=CustomPhiFunction([lambda X: (X < 5).astype(int)],
                                  marginal_guarantee=False)
        )
        mapie_reg.fit(X_toy[X_toy[:, 0] < 5], y_toy[X_toy[:, 0] < 5])
        mapie_reg.predict(X_toy)


@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_no_fit_prefit_calibrate(estimator: Any) -> None:
    """Test that calibrate without fit, if prefit, raises no errors."""
    estimator.fit(X_toy, y_toy)
    mapie_reg = MapieCCPRegressor(estimator, cv="prefit", alpha=0.1)
    mapie_reg.fit_calibrator(X_toy, y_toy)


def test_no_fit_predict() -> None:
    """Test that predict before fit raises errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    with pytest.raises(NotFittedError):
        mapie_reg.predict(X_toy)


def test_no_calibrate_predict() -> None:
    """Test that predict before fit raises errors."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    mapie_reg.fit_estimator(X_toy, y_toy)
    with pytest.raises(NotFittedError):
        mapie_reg.predict(X_toy)


def test_default_sample_weight() -> None:
    """Test default sample weights."""
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    assert (
        signature(mapie_reg.fit_estimator).parameters["sample_weight"].default
        is None
    )


@pytest.mark.parametrize("estimator", [0, "a", KFold(), ["a", "b"]])
def test_invalid_estimator(
    estimator: Any
) -> None:
    """Test that invalid estimators raise errors."""
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie = MapieCCPRegressor(estimator=estimator, alpha=0.1)
        mapie.fit_estimator(X, y)


@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_invalid_prefit_estimator_calibrate(
    estimator: RegressorMixin,
) -> None:
    """Test that non-fitted estimator with prefit cv raise errors when
    calibrate is called"""
    with pytest.raises(NotFittedError):
        mapie = MapieCCPRegressor(estimator=estimator, cv="prefit", alpha=0.1)
        mapie.fit_calibrator(X, y)


@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_invalid_prefit_estimator_fit(
    estimator: RegressorMixin,
) -> None:
    """Test that non-fitted estimator with prefit cv raise errors when fit
    is called."""
    with pytest.raises(NotFittedError):
        mapie = MapieCCPRegressor(estimator=estimator, cv="prefit", alpha=0.1)
        mapie.fit_estimator(X, y)


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = MapieCCPRegressor(random_state=random_state, alpha=0.1)
    mapie_reg.fit(X, y)
    assert isinstance(mapie_reg.estimator, RegressorMixin)
    assert isinstance(mapie_reg.phi, GaussianPhiFunction)
    assert isinstance(mapie_reg.cv, ShuffleSplit)
    assert mapie_reg.alpha == 0.1
    assert isinstance(mapie_reg.conformity_score_, ConformityScore)
    assert isinstance(mapie_reg.random_state, int)


@pytest.mark.parametrize(
    "alpha", ["a", 0, 2, 1.5, -0.3]
)
def test_invalid_alpha(alpha: Any) -> None:
    with pytest.raises(ValueError):
        mapie = MapieCCPRegressor(alpha=alpha)
        mapie.fit(X, y)


@pytest.mark.parametrize(
    "phi", [1, "some_string"]
)
def test_invalid_phi(phi: Any) -> None:
    with pytest.raises(ValueError):
        mapie = MapieCCPRegressor(phi=phi)
        mapie.fit(X, y)


def test_valid_estimator() -> None:
    """Test that valid estimators are not corrupted"""
    mapie_reg = MapieCCPRegressor(
        estimator=DummyRegressor(),
        random_state=random_state,
        alpha=0.1,
    )
    mapie_reg.fit_estimator(X_toy, y_toy)
    assert isinstance(mapie_reg.estimator, DummyRegressor)


@pytest.mark.parametrize(
    "cv", [None, ShuffleSplit(n_splits=1),
           PredefinedSplit(
                test_fold=[1]*(len(X_toy)//2) + [-1]*(len(X_toy)-len(X_toy)//2)
           ), "prefit", "split"]
)
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_valid_cv(cv: Any, estimator: RegressorMixin) -> None:
    """Test that valid cv raise no errors."""
    estimator.fit(X_toy, y_toy)
    mapie_reg = MapieCCPRegressor(estimator=estimator, cv=cv, alpha=0.1,
                                  random_state=random_state)
    mapie_reg.fit(X_toy, y_toy)
    mapie_reg.predict(X_toy)


@pytest.mark.parametrize(
    "cv", ["dummy", 0, 1, 1.5] + [  # Cross val splitters
            3, -1, KFold(n_splits=5), LeaveOneOut(),
            RepeatedKFold(n_splits=5, n_repeats=2), ShuffleSplit(n_splits=5),
            TimeSeriesSplit(), LeavePOut(p=2),
            PredefinedSplit(test_fold=[0]*(len(X_toy)//4) + [1]*(len(X_toy)//4)
                            + [-1]*(len(X_toy)-len(X_toy)//2)),
           ]
)
def test_invalid_cv(cv: Any) -> None:
    """Test that invalid agg_functions raise errors."""
    with pytest.raises(ValueError, match="Invalid cv argument."):
        mapie = MapieCCPRegressor(cv=cv, alpha=0.1, random_state=random_state)
        mapie.fit_estimator(X, y)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("alpha", [0.2])
@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_fit_calibrate_combined_equivalence(
    alpha: Any, dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any, phi: PhiFunction, estimator: RegressorMixin
) -> None:
    """Test predict output shape."""
    (X, y, z) = dataset

    cloned_phi = clone(phi)
    cloned_phi.fit(X)
    estimator_1 = clone(estimator)
    estimator_2 = clone(estimator)
    if cv == "prefit":
        estimator_1.fit(X, y)
        estimator_2.fit(X, y)

    mapie_1 = MapieCCPRegressor(estimator=estimator_1, phi=cloned_phi, cv=cv,
                                alpha=alpha, random_state=random_state)
    mapie_2 = MapieCCPRegressor(estimator=estimator_2, phi=cloned_phi, cv=cv,
                                alpha=alpha, random_state=random_state)
    mapie_1.fit(X, y, z=z)
    mapie_2.fit_estimator(X, y)
    mapie_2.fit_calibrator(X, y, z=z)
    y_pred_1, y_pis_1 = mapie_1.predict(X, z)
    y_pred_2, y_pis_2 = mapie_2.predict(X, z)
    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


def test_recalibrate_warning() -> None:
    """
    Test that a warning is triggered when we calibrate a second time with
    a different alpha value
    """
    mapie_reg = MapieCCPRegressor(alpha=0.1)
    mapie_reg.fit(X_toy, y_toy)
    with pytest.warns(UserWarning, match=r"WARNING: The old value of alpha"):
        mapie_reg.fit_calibrator(X_toy, y_toy, alpha=0.2)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_recalibrate(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any, phi: PhiFunction, estimator: RegressorMixin
) -> None:
    """
    Test that the PI are different for different value of alpha,
    but they are equal if we calibrate again with the correct alpha
    """
    (X, y, z) = dataset
    if cv == "prefit":
        estimator.fit(X, y)

    cloned_phi = clone(phi)
    cloned_phi.fit(X)

    mapie_1 = MapieCCPRegressor(estimator=estimator, phi=cloned_phi, cv=cv,
                                alpha=0.2, random_state=random_state)
    mapie_2 = MapieCCPRegressor(estimator=estimator, phi=cloned_phi, cv=cv,
                                alpha=0.1, random_state=random_state)
    mapie_1.fit(X, y, z=z)
    mapie_2.fit(X, y, z=z)

    y_pred_1, y_pis_1 = mapie_1.predict(X, z)
    y_pred_2, y_pis_2 = mapie_2.predict(X, z)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y_pis_1, y_pis_2)

    mapie_2.fit_calibrator(X, y, z=z, alpha=0.2)
    y_pred_2, y_pis_2 = mapie_2.predict(X, z)
    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_predict_output_shape_alpha(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any, phi: PhiFunction, estimator: RegressorMixin
) -> None:
    """Test predict output shape."""
    (X, y, z) = dataset
    if cv == "prefit":
        estimator.fit(X, y)

    mapie_reg = MapieCCPRegressor(estimator=estimator, phi=clone(phi), cv=cv,
                                  alpha=0.1, random_state=random_state)
    mapie_reg.fit(X, y, z=z)
    y_pred, y_pis = mapie_reg.predict(X, z)
    assert y_pred.shape == (X.shape[0],)
    assert y_pis.shape == (X.shape[0], 2, 1)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_predict_output_shape_no_alpha(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any, phi: PhiFunction, estimator: RegressorMixin
) -> None:
    """Test predict output shape."""
    (X, y, z) = dataset
    if cv == "prefit":
        estimator.fit(X, y)

    mapie_reg = MapieCCPRegressor(estimator=estimator, phi=clone(phi), cv=cv,
                                  alpha=None, random_state=random_state)
    mapie_reg.fit(X, y, z=z)
    y_pred = mapie_reg.predict(X, z)
    assert np.array(y_pred).shape == (X.shape[0],)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("estimator_1, estimator_2", zip(*[[
        LinearRegression(),
        make_pipeline(LinearRegression()),
    ]]*2)
)
def test_same_results_prefit_split(
    dataset: Tuple[NDArray, NDArray, NDArray], phi: PhiFunction,
    estimator_1: RegressorMixin, estimator_2: RegressorMixin
) -> None:
    """
    Test checking that if split and prefit method have exactly
    the same data split, then we have exactly the same results.
    """
    (X, y, z) = dataset
    cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    train_index, val_index = list(cv.split(X))[0]
    X_train, X_calib = X[train_index], X[val_index]
    y_train, y_calib = y[train_index], y[val_index]
    z_calib = z[val_index]

    cloned_phi = clone(phi)
    cloned_phi.fit(X)

    mapie_reg = MapieCCPRegressor(estimator=estimator_1, phi=cloned_phi, cv=cv,
                                  alpha=0.1, random_state=random_state)
    mapie_reg.fit(X, y, z=z)
    y_pred_1, y_pis_1 = mapie_reg.predict(X, z)

    estimator_2.fit(X_train, y_train)
    mapie_reg = MapieCCPRegressor(
        estimator=estimator_2, phi=cloned_phi, cv="prefit", alpha=0.1,
        random_state=random_state
    )
    mapie_reg.fit_calibrator(X_calib, y_calib, z=z_calib)
    y_pred_2, y_pis_2 = mapie_reg.predict(X, z)

    np.testing.assert_allclose(y_pred_1, y_pred_2)
    np.testing.assert_allclose(y_pis_1[:, 0, 0], y_pis_2[:, 0, 0])
    np.testing.assert_allclose(y_pis_1[:, 1, 0], y_pis_2[:, 1, 0])


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_results_for_ordered_alpha(
    dataset: Tuple[NDArray, NDArray, NDArray], cv: Any,
    phi: PhiFunction, estimator: RegressorMixin
) -> None:
    """
    Test that prediction intervals lower (upper) bounds give
    consistent results for ordered alphas.
    """
    (X, y, z) = dataset
    if cv == "prefit":
        estimator.fit(X, y)
    cloned_phi = clone(phi)
    cloned_phi.fit(X)

    mapie_reg_1 = MapieCCPRegressor(estimator=estimator, phi=cloned_phi, cv=cv,
                                    alpha=0.05, random_state=random_state)
    mapie_reg_1.fit(X, y, z=z)
    _, y_pis_1 = mapie_reg_1.predict(X, z)

    mapie_reg_2 = MapieCCPRegressor(estimator=estimator, phi=cloned_phi, cv=cv,
                                    alpha=0.1, random_state=random_state)
    mapie_reg_2.fit(X, y, z=z)
    _, y_pis_2 = mapie_reg_1.predict(X, z)

    assert (y_pis_1[:, 0, 0] <= y_pis_2[:, 0, 0]).all()
    assert (y_pis_1[:, 1, 0] >= y_pis_2[:, 1, 0]).all()


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("estimator1, estimator2, estimator3", zip([
    LinearRegression(),
    make_pipeline(LinearRegression()),
], [
    LinearRegression(),
    make_pipeline(LinearRegression()),
], [
    LinearRegression(),
    make_pipeline(LinearRegression()),
]))
def test_results_with_constant_sample_weights(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any,
    estimator1: RegressorMixin,
    estimator2: RegressorMixin,
    estimator3: RegressorMixin,
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    (X, y, z) = dataset
    if cv == "prefit":
        estimator1.fit(X, y)
        estimator2.fit(X, y)
        estimator3.fit(X, y)

    cloned_phi = clone(PHI[0])
    cloned_phi.fit(X)

    n_samples = len(X)
    mapie0 = MapieCCPRegressor(estimator=estimator1, phi=cloned_phi,
                               cv=cv, alpha=0.1, random_state=random_state)
    mapie1 = MapieCCPRegressor(estimator=estimator2, phi=cloned_phi,
                               cv=cv, alpha=0.1, random_state=random_state)
    mapie2 = MapieCCPRegressor(estimator=estimator3, phi=cloned_phi,
                               cv=cv, alpha=0.1, random_state=random_state)

    mapie0.fit(X, y, z=z, sample_weight=None)
    mapie1.fit(X, y, z=z, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X, y, z=z, sample_weight=np.ones(shape=n_samples) * 3)

    y_pred0, y_pis0 = mapie0.predict(X, z=z)
    y_pred1, y_pis1 = mapie1.predict(X, z=z)
    y_pred2, y_pis2 = mapie2.predict(X, z=z)
    np.testing.assert_allclose(y_pred0, y_pred1, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(y_pred0, y_pred2, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(y_pis0, y_pis1, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(y_pis0, y_pis2, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dataset", [(X, y, z), (X_toy, y_toy, z_toy)])
@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("alpha", [0.2, 0.1, 0.05])
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_prediction_between_low_up(
    dataset: Tuple[NDArray, NDArray, NDArray],
    cv: Any,
    phi: PhiFunction,
    alpha: float,
    estimator: RegressorMixin
) -> None:
    """Test that prediction lies between low and up prediction intervals."""
    (X, y, z) = dataset

    if cv == "prefit":
        estimator.fit(X, y)

    mapie = MapieCCPRegressor(estimator=estimator, phi=clone(phi), cv=cv,
                              alpha=alpha, random_state=random_state)
    mapie.fit(X, y, z=z)

    with warnings.catch_warnings(record=True) as record:
        y_pred, y_pis = mapie.predict(X, z=z)

    # Check if the warning was issued
    warning_issued = any("The predictions are ill-sorted." in str(w.message)
                         for w in record)

    # Perform assertions based on whether the warning was issued
    if not warning_issued:
        assert (y_pred >= y_pis[:, 0, 0]).all()
        assert (y_pred <= y_pis[:, 1, 0]).all()


@pytest.mark.parametrize("phi", PHI[:2])
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("alpha", [0.2, 0.1])
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_linear_data_confidence_interval(
    cv: Any,
    phi: PhiFunction,
    alpha: float,
    estimator: RegressorMixin
) -> None:
    """
    Test that MapieRegressor applied on a linear regression estimator
    fitted on a linear curve results in null uncertainty.
    """
    X_toy = np.arange(0, 200, 1).reshape(-1, 1)
    y_toy = X_toy[:, 0]*2
    z_toy = np.ones((len(X_toy), 1))

    if cv == "prefit":
        estimator.fit(X_toy, y_toy)

    mapie = MapieCCPRegressor(estimator=estimator, phi=clone(phi), cv=cv,
                              alpha=alpha, random_state=random_state)
    mapie.fit(X_toy, y_toy, z=z_toy)

    y_pred, y_pis = mapie.predict(X_toy, z=z_toy)
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 1, 0],
                               rtol=0.01, atol=0.1)
    np.testing.assert_allclose(y_pred, y_pis[:, 0, 0],
                               rtol=0.01, atol=0.1)


def test_linear_regression_results() -> None:
    """
    Test that the CCP method in the case of a constant
    phi = x -> np.ones(len(x)), on a multivariate linear regression problem
    with fixed random state, is strictly equivalent to the regular CP method
    (base, jacknife and cv)
    """

    mapie = MapieCCPRegressor(
        phi=clone(PHI[0]),
        cv=ShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state),
        alpha=0.05,
        random_state=random_state
    )
    mapie.fit(X, y)
    _, y_pis = mapie.predict(X)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS["split"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["split"], rtol=1e-2)


@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
def test_results_prefit(estimator: RegressorMixin) -> None:
    """Test prefit results on a standard train/validation/test split."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=1 / 10, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=1 / 9, random_state=1
    )
    estimator.fit(X_train, y_train)
    mapie_reg = MapieCCPRegressor(
        estimator=estimator, phi=clone(PHI[0]), cv="prefit", alpha=0.05,
        random_state=random_state
    )
    mapie_reg.fit(X_val, y_val)
    _, y_pis = mapie_reg.predict(X_test)
    width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
    coverage = regression_coverage_score(
        y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]
    )
    np.testing.assert_allclose(width_mean, WIDTHS["prefit"], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES["prefit"], rtol=1e-2)


@pytest.mark.parametrize("phi", PHI)
@pytest.mark.parametrize("cv", CV)
@pytest.mark.parametrize("estimator", [
    LinearRegression(),
    make_pipeline(LinearRegression()),
])
@pytest.mark.parametrize(
    "conformity_score", [AbsoluteConformityScore(), GammaConformityScore(),
                         ResidualNormalisedScore()]
)
def test_conformity_score(
    cv: Any,
    phi: PhiFunction,
    estimator: RegressorMixin,
    conformity_score: ConformityScore
) -> None:
    """Test that any conformity score function with MAPIE raises no error."""

    if cv == "prefit":
        estimator.fit(X, y + 1e3)

    mapie_reg = MapieCCPRegressor(
        estimator=estimator,
        phi=clone(phi),
        cv=cv,
        alpha=0.1,
        conformity_score=conformity_score,
        random_state=random_state,
    )
    mapie_reg.fit(X, y + 1e3, z=z)
    mapie_reg.predict(X, z=z)


def test_fit_parameters_passing() -> None:
    """
    Test passing fit parameters, here early stopping at iteration 3.
    Checks that underlying GradientBoosting estimators have used 3 iterations
    only during boosting, instead of default value for n_estimators (=100).
    """
    gb = GradientBoostingRegressor(random_state=random_state)

    mapie_reg = MapieCCPRegressor(estimator=gb, alpha=0.1,
                                  random_state=random_state)

    def early_stopping_monitor(i, est, locals):
        """Returns True on the 3rd iteration."""
        if i == 2:
            return True
        else:
            return False

    mapie_reg.fit(X, y, monitor=early_stopping_monitor)

    assert cast(RegressorMixin, mapie_reg.estimator).estimators_.shape[0] == 3
