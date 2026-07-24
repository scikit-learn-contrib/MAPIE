from __future__ import annotations

from types import MethodType
from typing import Any, Tuple, Optional

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from typing_extensions import TypedDict

from mapie.metrics.regression import regression_coverage_score
from mapie.conformity_scores import BaseRegressionScore
from mapie.regression.quantile_regression import (
    AbsoluteQuantileRegressionScore,
    QuantileRegressionScore,
    CrossConformalizedQuantileRegressor,
    _QuantileConformalizer,
    _MapieQuantileRegressor,
)
from mapie.utils import check_is_fitted

X_toy = np.array(
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
).reshape(-1, 1)
y_toy = np.array(
    [5, 7, 9, 11, 13, 15, 5, 7, 9, 11, 13, 15, 5, 7, 9, 11, 13, 15, 5, 7, 9, 11, 13, 15]
)

random_state = 1

X_train_toy, X_calib_toy, y_train_toy, y_calib_toy = train_test_split(
    X_toy, y_toy, test_size=0.5, random_state=random_state
)

qt = QuantileRegressor(solver="highs-ds")
gb = GradientBoostingRegressor(loss="quantile", random_state=random_state)

X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X, y, test_size=0.5, random_state=random_state
)

SYMMETRY = [True, False]
ESTIMATOR = [qt, gb]

Params = TypedDict(
    "Params",
    {
        "method": str,
        "alpha": float,
    },
)

STRATEGIES = {
    "quantile_alpha2": Params(method="quantile", alpha=0.2),
    "quantile_alpha3": Params(method="quantile", alpha=0.3),
    "quantile_alpha4": Params(method="quantile", alpha=0.4),
    "quantile_alpha8": Params(method="quantile", alpha=0.8),
}

WIDTHS = {
    "quantile_alpha2": 2.7360884795455576,
    "quantile_alpha3": 2.185652142101473,
    "quantile_alpha4": 1.731718678152845,
    "quantile_alpha8": 0.66909752420949,
}

COVERAGES = {
    "quantile_alpha2": 0.834,
    "quantile_alpha3": 0.738,
    "quantile_alpha4": 0.646,
    "quantile_alpha8": 0.264,
}


class NotFitPredictEstimator:
    def __init__(self, alpha):
        self.alpha = alpha


class NoLossParameterEstimator(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict(self, *args: Any) -> None:
        """Dummy predict."""


class NoAlphaParameterEstimator(BaseEstimator):
    def __init__(self, alpha, loss):
        self.alpha = alpha
        self.loss = loss

    def fit(self, *args: Any) -> None:
        """Dummy fit."""

    def predict(self, *args: Any) -> None:
        """Dummy predict."""


class DummyQuantileConformalizer(_QuantileConformalizer):
    def _check_cv(self, cv: Any) -> str:
        return "split" if cv is None else cv


class FixedPredictor(BaseEstimator):
    def __init__(self, prediction: NDArray):
        self.prediction = np.asarray(prediction)

    def fit(self, X: Any, y: Any) -> FixedPredictor:
        return self

    def predict(self, X: Any, **kwargs: Any) -> NDArray:
        return self.prediction


class PrefitLikeCV:
    """Lightweight CV stub behaving like prefit mode for _predict_calib tests."""

    def __eq__(self, other: Any) -> bool:
        return other == "prefit"

    def get_n_splits(self, X: Any, y: Any = None, groups: Any = None) -> int:
        return 2

    def split(self, X: Any, y: Any = None, groups: Any = None):
        yield np.array([0, 1]), np.array([0, 2])
        yield np.array([2, 3]), np.array([1, 3])


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = _MapieQuantileRegressor()
    assert mapie_reg.method == "quantile"
    assert mapie_reg.cv is None
    assert mapie_reg.alpha == 0.1


@pytest.mark.parametrize(
    "cv, alpha, expected_cv, expected_alpha_np",
    [
        (None, 0.1, "split", np.array([0.05, 0.95, 0.5])),
        ("prefit", 0.2, "prefit", np.array([0.1, 0.9, 0.5])),
    ],
)
def test_initialize_fit_conformalize(
    cv: Any,
    alpha: float,
    expected_cv: str,
    expected_alpha_np: NDArray,
) -> None:
    """Test initialization with default and user-provided values."""
    mapie_reg = _MapieQuantileRegressor(cv=cv, alpha=alpha)
    mapie_reg.estimators_ = [clone(qt)]

    mapie_reg._initialize_fit_conformalize()

    assert mapie_reg.cv == expected_cv
    np.testing.assert_allclose(mapie_reg.alpha_np, expected_alpha_np)
    assert mapie_reg.estimators_ == []


def test_default_parameters_estimator() -> None:
    """Test default values of estimator."""
    mapie_reg = _MapieQuantileRegressor()
    mapie_reg._initialize_fit_conformalize()
    mapie_reg._fit_estimators(X_train, y_train)
    for estimator in mapie_reg.estimators_:
        assert isinstance(estimator, QuantileRegressor)
        assert estimator.__dict__["solver"] == "highs-ds"


def test_no_predict_fit_estimator() -> None:
    """Test that estimators with not fit or predict methods raise an error."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid estimator.*",
    ):
        mapie_reg = _MapieQuantileRegressor(estimator=NotFitPredictEstimator(alpha=0.2))
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train, y_train)


def test_no_para_loss_estimator() -> None:
    """Test to check when it does not have a valid loss_name."""
    with pytest.raises(
        ValueError,
        match=r".*The matching parameter `loss_name`*",
    ):
        mapie_reg = _MapieQuantileRegressor()
        mapie_reg.quantile_estimator_params["NoLossParameterEstimator"] = {
            "loss_name": "noloss",
            "alpha_name": "alpha",
        }
        mapie_reg.estimator = NoLossParameterEstimator(alpha=0.2)
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train, y_train)


def test_no_para_alpha_estimator() -> None:
    """Test to check when it does not have a valid alpha parameter name"""
    with pytest.raises(
        ValueError,
        match=r".*The matching parameter `alpha_name`*",
    ):
        mapie_reg = _MapieQuantileRegressor()
        mapie_reg.quantile_estimator_params["NoAlphaParameterEstimator"] = {
            "loss_name": "loss",
            "alpha_name": "noalpha",
        }
        mapie_reg.estimator = NoAlphaParameterEstimator(alpha=0.2, loss="quantile")
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train, y_train)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
def test_valid_method(strategy: str, estimator: RegressorMixin) -> None:
    """Test that valid strategies and estimators raise no error"""
    mapie_reg = _MapieQuantileRegressor(estimator=estimator, **STRATEGIES[strategy])
    mapie_reg._initialize_fit_conformalize()
    mapie_reg._fit_estimators(X_train, y_train)
    check_is_fitted(mapie_reg)
    assert mapie_reg.__dict__["method"] == "quantile"


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize(
    "dataset",
    [
        (X_train, X_calib, y_train, y_calib),
        (X_train_toy, X_calib_toy, y_train_toy, y_calib_toy),
    ],
)
@pytest.mark.parametrize("symmetry", SYMMETRY)
def test_predict_output_shape(
    strategy: str,
    estimator: RegressorMixin,
    dataset: Tuple[NDArray, NDArray, NDArray, NDArray],
    symmetry: bool,
) -> None:
    """Test predict output shape."""
    mapie_reg = _MapieQuantileRegressor(estimator=estimator, **STRATEGIES[strategy])
    (X_t, X_c, y_t, y_c) = dataset
    mapie_reg._initialize_fit_conformalize()
    mapie_reg._fit_estimators(X_t, y_t)
    mapie_reg.conformalize(X_c, y_c)
    y_pred, y_pis = mapie_reg.predict(X_t, symmetry=symmetry)
    assert y_pred.shape == (X_t.shape[0],)
    assert y_pis[:, 0, 0].shape == (X_t.shape[0],)
    assert y_pis[:, 1, 0].shape == (X_t.shape[0],)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(
    strategy: str,
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X_train)
    mapie0 = _MapieQuantileRegressor(estimator=qt, **STRATEGIES[strategy])
    mapie1 = _MapieQuantileRegressor(estimator=qt, **STRATEGIES[strategy])
    mapie2 = _MapieQuantileRegressor(estimator=qt, **STRATEGIES[strategy])
    mapie0._initialize_fit_conformalize()
    mapie0._fit_estimators(X_train, y_train)
    mapie0.conformalize(X_calib, y_calib)
    mapie1._initialize_fit_conformalize()
    mapie1._fit_estimators(X_train, y_train, sample_weight=np.ones(shape=n_samples))
    mapie1.conformalize(X_calib, y_calib)
    mapie2._initialize_fit_conformalize()
    mapie2._fit_estimators(X_train, y_train, sample_weight=np.ones(shape=n_samples) * 5)
    mapie2.conformalize(X_calib, y_calib)

    np.testing.assert_allclose(mapie0.conformity_scores, mapie1.conformity_scores)
    np.testing.assert_allclose(mapie0.conformity_scores, mapie2.conformity_scores)

    y_pred0, y_pis0 = mapie0.predict(X)
    y_pred1, y_pis1 = mapie1.predict(X)
    y_pred2, y_pis2 = mapie2.predict(X)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)


@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize("symmetry", SYMMETRY)
def test_results_for_same_alpha(estimator: RegressorMixin, symmetry: bool) -> None:
    """
    Test that predictions and intervals
    are similar with two equal values of alpha.
    """
    mapie_reg = _MapieQuantileRegressor(estimator=estimator, alpha=0.2)
    mapie_reg_clone = clone(mapie_reg)
    mapie_reg._initialize_fit_conformalize()
    mapie_reg._fit_estimators(X_train, y_train)
    mapie_reg.conformalize(X_calib, y_calib)
    mapie_reg_clone._initialize_fit_conformalize()
    mapie_reg_clone._fit_estimators(X_train, y_train)
    mapie_reg_clone.conformalize(X_calib, y_calib)
    y_pred, y_pis = mapie_reg.predict(X, symmetry=symmetry)
    y_pred_clone, y_pis_clone = mapie_reg_clone.predict(X, symmetry=symmetry)
    np.testing.assert_allclose(y_pred, y_pred_clone)
    np.testing.assert_allclose(y_pis[:, 0, 0], y_pis_clone[:, 0, 0])
    np.testing.assert_allclose(y_pis[:, 1, 0], y_pis_clone[:, 1, 0])


@pytest.mark.parametrize("alphas", ["hello", _MapieQuantileRegressor, [2], 1])
def test_wrong_alphas_types(alphas: float) -> None:
    """Checking for wrong type of alphas"""
    with pytest.raises(
        ValueError,
        match=r".*Invalid confidence_level. Allowed values are float.*",
    ):
        mapie_reg = _MapieQuantileRegressor(alpha=alphas)
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train, y_train)
        mapie_reg.conformalize(X_calib, y_calib)


@pytest.mark.parametrize("alphas", [1.0, 1.6, 1.95, 5.0, -0.1, -0.001, -10.0])
def test_wrong_alphas(alphas: float) -> None:
    """Checking for alphas values that are too big according to all value."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid confidence_level. Allowed values are between .*",
    ):
        mapie_reg = _MapieQuantileRegressor(alpha=alphas)
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train, y_train)
        mapie_reg.conformalize(X_calib, y_calib)


def test_estimators_quantile_function() -> None:
    """Checking for badly set estimator parameters."""
    with pytest.raises(
        ValueError,
        match=r".*You need to set the loss/objective*",
    ):
        mapie_reg = _MapieQuantileRegressor(estimator=GradientBoostingRegressor())
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train, y_train)
        mapie_reg.conformalize(X_calib, y_calib)


@pytest.mark.parametrize("cv", [-1, 2, KFold(), LeaveOneOut()])
def test_invalid_cv(cv: Any) -> None:
    """Test that valid cv raise errors."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid cv method.*",
    ):
        mapie = _MapieQuantileRegressor(cv=cv)
        mapie._initialize_fit_conformalize()
        mapie._fit_estimators(X_train, y_train)
        mapie.conformalize(X_calib, y_calib)


@pytest.mark.parametrize("cv", [None, "split"])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = _MapieQuantileRegressor(cv=cv)
    mapie._initialize_fit_conformalize()
    mapie._fit_estimators(X_train, y_train)
    mapie.conformalize(X_calib, y_calib)


def test_calib_dataset_is_none() -> None:
    """Test that the fit method works when X_calib or y_calib is None."""
    mapie = _MapieQuantileRegressor()
    mapie._initialize_fit_conformalize()
    X_train, y_train, X_calib, y_calib, _ = mapie._prepare_train_calib(
        X, y, calib_size=0.5
    )
    mapie._fit_estimators(X_train, y_train)
    mapie.conformalize(X_calib, y_calib)
    mapie.predict(X)


def test_calib_dataset_is_none_with_sample_weight() -> None:
    """
    Test that the fit method works with calib dataset defined is None
    with sample weights.
    """
    mapie = _MapieQuantileRegressor()
    mapie._initialize_fit_conformalize()
    X_train, y_train, X_calib, y_calib, weights = mapie._prepare_train_calib(
        X, y, sample_weight=np.ones(X.shape[0]), calib_size=0.5
    )
    mapie._fit_estimators(X_train, y_train, sample_weight=weights)
    mapie.conformalize(X_calib, y_calib)
    mapie.predict(X)


@pytest.mark.parametrize("est", [RandomForestClassifier(), LinearRegression()])
def test_estimators_not_in_list(est: RegressorMixin) -> None:
    """
    Test for estimators that are not in the list, hence not accepted
    estimators
    """
    with pytest.raises(
        ValueError,
        match=r".*The base model is not supported.*",
    ):
        mapie_reg = _MapieQuantileRegressor(estimator=est)
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train, y_train)
        mapie_reg.conformalize(X_calib, y_calib)


def test_for_small_dataset() -> None:
    """Test for when we have calibration datasets that are too small."""
    with pytest.raises(
        ValueError,
        match=r".*Number of samples of the score is too low*",
    ):
        mapie_reg = _MapieQuantileRegressor(estimator=qt, alpha=0.1)
        X_calib_toy_small = X_calib_toy[:2]
        y_calib_toy_small = y_calib_toy[:2]
        mapie_reg._initialize_fit_conformalize()
        mapie_reg._fit_estimators(X_train_toy, y_train_toy)
        mapie_reg.conformalize(X_calib_toy_small, y_calib_toy_small)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize(
    "dataset",
    [
        (X_train, X_calib, y_train, y_calib),
        (X_train_toy, X_calib_toy, y_train_toy, y_calib_toy),
    ],
)
def test_conformity_len(
    strategy: str,
    estimator: RegressorMixin,
    dataset: Tuple[NDArray, NDArray, NDArray, NDArray],
) -> None:
    """Test conformity scores output shape."""
    (X_t, X_c, y_t, y_c) = dataset
    n_samples = int(len(X_c))
    mapie_regressor = _MapieQuantileRegressor(
        estimator=estimator, **STRATEGIES[strategy]
    )
    mapie_regressor._initialize_fit_conformalize()
    mapie_regressor._fit_estimators(X_t, y_t)
    mapie_regressor.conformalize(X_c, y_c)
    assert mapie_regressor.conformity_scores[0].shape[0] == n_samples


# Working but want to add both symmetry and different estimators
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a different strategies.
    """
    mapie = _MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie._initialize_fit_conformalize()
    mapie._fit_estimators(X_train, y_train)
    mapie.conformalize(X_calib, y_calib)
    _, y_pis = mapie.predict(X)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pis)[0]
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)


def test_quantile_prefit_three_estimators() -> None:
    """
    Test that there is a list with three estimators provided for
    cv="prefit".
    """
    with pytest.raises(
        ValueError,
        match=r".*You need to have provided 3 different estimators, th*",
    ):
        gb_trained1, gb_trained2 = clone(gb), clone(gb)
        gb_trained1.fit(X_train, y_train)
        gb_trained2.fit(X_train, y_train)
        list_estimators = [gb_trained1, gb_trained2]
        mapie_reg = _MapieQuantileRegressor(estimator=list_estimators, cv="prefit")
        mapie_reg.conformalize(X_calib, y_calib)


def test_prefit_no_fit_predict() -> None:
    """
    Check that the estimators given have a prefit and fit attribute.
    """
    with pytest.raises(
        ValueError,
        match=r"Invalid estimator. Please provide a regressor with fit and*",
    ):
        gb_trained1, gb_trained2 = clone(gb), clone(gb)
        gb_trained1.fit(X_train, y_train)
        gb_trained2.fit(X_train, y_train)
        gb_trained3 = 3
        list_estimators = [gb_trained1, gb_trained2, gb_trained3]
        mapie_reg = _MapieQuantileRegressor(
            estimator=list_estimators, cv="prefit", alpha=0.3
        )
        mapie_reg.conformalize(X_calib, y_calib)


@pytest.mark.filterwarnings("ignore:Estimator does not appear fitted.*:UserWarning")
def test_non_trained_estimator() -> None:
    """
    Check that the estimators are all already trained when used in prefit.
    """
    with pytest.raises(
        ValueError,
        match=r".*instance is not fitted yet. Call 'fit' with appropriate*",
    ):
        gb_trained1, gb_trained2, gb_trained3 = clone(gb), clone(gb), clone(gb)
        gb_trained1.fit(X_train, y_train)
        gb_trained2.fit(X_train, y_train)
        list_estimators = [gb_trained1, gb_trained2, gb_trained3]
        mapie_reg = _MapieQuantileRegressor(
            estimator=list_estimators, cv="prefit", alpha=0.3
        )
        mapie_reg.conformalize(X_calib, y_calib)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.3])
def test_prefit_and_non_prefit_equal(alpha: float) -> None:
    """
    Check that when using prefit and not prefit, the same values
    are found.
    """
    list_estimators = []
    alphas_ = [alpha / 2, 1 - (alpha / 2), 0.5]
    for alpha_ in alphas_:
        est = clone(qt)
        params = {"quantile": alpha_}
        est.set_params(**params)
        est.fit(X_train, y_train)
        list_estimators.append(est)
    mapie_reg_prefit = _MapieQuantileRegressor(
        estimator=list_estimators, cv="prefit", alpha=alpha
    )
    mapie_reg_prefit.conformalize(X_calib, y_calib)
    y_pred_prefit, y_pis_prefit = mapie_reg_prefit.predict(X)

    mapie_reg = _MapieQuantileRegressor(estimator=qt, alpha=alpha)
    mapie_reg._initialize_fit_conformalize()
    mapie_reg._fit_estimators(X_train, y_train)
    mapie_reg.conformalize(X_calib, y_calib)
    y_pred, y_pis = mapie_reg.predict(X)

    np.testing.assert_allclose(y_pred_prefit, y_pred)
    np.testing.assert_allclose(y_pis_prefit, y_pis)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.3])
def test_prefit_different_type_list_tuple_array(alpha: float) -> None:
    """
    Check that the type of Iterable (list, np.array, tuple) to
    estimators gives similar results.
    """
    list_estimators = []
    alphas_ = [alpha / 2, 1 - (alpha / 2), 0.5]
    for alpha_ in alphas_:
        est = clone(qt)
        params = {"quantile": alpha_}
        est.set_params(**params)
        est.fit(X_train, y_train)
        list_estimators.append(est)

    mapie_reg_prefit_list = _MapieQuantileRegressor(
        estimator=list_estimators, cv="prefit", alpha=alpha
    )
    mapie_reg_prefit_list.conformalize(X_calib, y_calib)
    y_pred_prefit_list, y_pis_prefit_list = mapie_reg_prefit_list.predict(X)

    mapie_reg_prefit_tuple = _MapieQuantileRegressor(
        estimator=tuple(list_estimators), cv="prefit", alpha=alpha
    )
    mapie_reg_prefit_tuple.conformalize(X_calib, y_calib)
    y_pred_prefit_tuple, y_pis_prefit_tuple = mapie_reg_prefit_tuple.predict(X)

    mapie_reg_prefit_array = _MapieQuantileRegressor(
        estimator=np.array(list_estimators), cv="prefit", alpha=alpha
    )
    mapie_reg_prefit_array.conformalize(X_calib, y_calib)
    y_pred_prefit_array, y_pis_prefit_array = mapie_reg_prefit_array.predict(X)

    np.testing.assert_allclose(y_pred_prefit_list, y_pred_prefit_tuple)
    np.testing.assert_allclose(y_pis_prefit_list, y_pis_prefit_tuple)

    np.testing.assert_allclose(y_pred_prefit_list, y_pred_prefit_array)
    np.testing.assert_allclose(y_pis_prefit_list, y_pis_prefit_array)


@pytest.mark.parametrize("estimator", ESTIMATOR)
def test_pipeline_compatibility(estimator: RegressorMixin) -> None:
    """Check that MAPIE works on pipeline based on pandas dataframes"""
    X = pd.DataFrame(
        {
            "x_cat": ["A", "A", "B", "A", "A", "B", "A", "B", "B", "B"],
            "x_num": [0, 1, 1, 4, np.nan, 5, 4, 3, np.nan, 3],
            "y": [5, 7, 3, 9, 10, 8, 9, 7, 9, 8],
        }
    )
    y = pd.Series([5, 7, 3, 9, 10, 8, 9, 7, 10, 5])
    X_train_toy, X_calib_toy, y_train_toy, y_calib_toy = train_test_split(
        X, y, test_size=0.5, random_state=random_state
    )
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
    pipe = make_pipeline(preprocessor, estimator)
    mapie = _MapieQuantileRegressor(pipe, alpha=0.4)
    mapie._initialize_fit_conformalize()
    mapie._fit_estimators(X_train_toy, y_train_toy)
    mapie.conformalize(X_calib_toy, y_calib_toy)
    mapie.predict(X)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_fit_parameters_passing(strategy: str) -> None:
    """
    Test passing fit parameters, here early stopping at iteration 3.
    Checks that underlying GradientBoosting estimators have used 3 iterations
    only during boosting, instead of default value for n_estimators (=100).
    """
    mapie = _MapieQuantileRegressor(estimator=gb, **STRATEGIES[strategy])

    def early_stopping_monitor(i, est, locals):
        """Returns True on the 3rd iteration."""
        if i == 2:
            return True
        else:
            return False

    mapie._initialize_fit_conformalize()
    mapie._fit_estimators(X_train, y_train, monitor=early_stopping_monitor)

    for estimator in mapie.estimators_:
        assert estimator.estimators_.shape[0] == 3


# ------------------------------ Test new implementation


def test_quantile_regression_score_signed_conformity_scores() -> None:
    """Test signed conformity scores for quantile regression score."""
    score = QuantileRegressionScore()
    y = np.array([2.0, 5.0, 8.0])
    y_pred = np.array(
        [
            [1.0, 4.0, 7.0],
            [3.0, 6.0, 9.0],
        ]
    )

    signed_scores = score.get_signed_conformity_scores(y, y_pred)

    expected_scores = np.vstack((y_pred[0] - y, y - y_pred[1]))
    assert signed_scores.shape == (2, 3)
    np.testing.assert_allclose(signed_scores, expected_scores)


def test_quantile_regression_score_estimation_distribution() -> None:
    """Test estimation distribution reconstruction from scores."""
    score = QuantileRegressionScore()
    y_pred = np.array([1.0, 2.0, 3.0])
    conformity_scores = np.array([0.5, -0.5, 1.5])

    estimation_distribution = score.get_estimation_distribution(
        y_pred, conformity_scores
    )

    np.testing.assert_allclose(estimation_distribution, y_pred + conformity_scores)


def test_quantile_regression_score_estimation_distribution_broadcasts() -> None:
    """Test estimation distribution broadcasting on test and calibration sizes."""
    score = QuantileRegressionScore()
    y_pred = np.array([1.0, 2.0])
    conformity_scores = np.array([0.5, -0.5, 1.5])

    estimation_distribution = score.get_estimation_distribution(
        y_pred, conformity_scores
    )

    expected_distribution = np.array(
        [
            [1.5, 0.5, 2.5],
            [2.5, 1.5, 3.5],
        ]
    )
    assert estimation_distribution.shape == (2, 3)
    np.testing.assert_allclose(estimation_distribution, expected_distribution)


def test_quantile_regression_score_get_quantile_rejects_scalar_alpha() -> None:
    """Test get_quantile expects alpha to be normalized before the call."""
    score = QuantileRegressionScore()
    conformity_scores = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(TypeError, match=r".*iterable.*"):
        score.get_quantile(
            conformity_scores,
            0.2,
            axis=1,
            reversed=True,
        )


def test_absolute_quantile_regression_score_conformity_scores() -> None:
    """Test absolute conformity score as pointwise max of signed scores."""
    score = AbsoluteQuantileRegressionScore()
    y = np.array([2.0, 5.0, 8.0])
    y_pred = np.array(
        [
            [1.5, 4.5, 7.5],
            [2.5, 5.5, 8.5],
        ]
    )

    conformity_scores = score.get_conformity_scores(y, y_pred)

    expected_scores = np.maximum(y_pred[0] - y, y - y_pred[1])
    np.testing.assert_allclose(conformity_scores, expected_scores)


def test_quantile_conformalizer_check_alpha_without_central_estimator() -> None:
    """Test alpha conversion when the central estimator is absent."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer._central_estimator = None

    alpha_np = conformalizer._check_alpha(0.2)

    np.testing.assert_allclose(alpha_np, np.array([0.1, 0.9, 0.5]))


def test_quantile_conformalizer_check_alpha_with_central_estimator() -> None:
    """Test alpha conversion when a central estimator already exists."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer._central_estimator = LinearRegression()

    alpha_np = conformalizer._check_alpha(0.2)

    np.testing.assert_allclose(alpha_np, np.array([0.1, 0.9]))


def test_quantile_conformalizer_check_score() -> None:
    """Test score validation against QuantileRegressionScore subclasses."""
    conformalizer = DummyQuantileConformalizer()

    conformalizer._check_score(AbsoluteQuantileRegressionScore)

    with pytest.raises(
        ValueError,
        match=r".*Invalid score. Allowed values are subclasses of QuantileRegressionScore.*",
    ):
        conformalizer._check_score(BaseRegressionScore)


def test_quantile_conformalizer_pinball_loss() -> None:
    """Test pinball loss computation for lower and upper quantiles."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.quantiles = np.array([0.1, 0.9])
    y_true = np.array([1.0, 3.0])
    y_pred = np.array(
        [
            [0.0, 4.0],
            [2.0, 2.0],
        ]
    )

    pinball_losses = conformalizer.pinball_loss(y_true, y_pred)

    expected_losses = np.array([0.5, 0.5])
    assert pinball_losses.shape == (2,)
    np.testing.assert_allclose(pinball_losses, expected_losses)


def test_quantile_conformalizer_set_quantile_estimator_params() -> None:
    """Test that quantile parameters are set on a clone, not in place."""
    conformalizer = DummyQuantileConformalizer()
    estimator = QuantileRegressor(solver="highs-ds", quantile=0.5)

    updated_estimator = conformalizer._set_quantile_estimator_params(
        estimator,
        alpha=0.2,
        alpha_name="quantile",
    )

    assert updated_estimator is not estimator
    assert updated_estimator.get_params()["quantile"] == 0.2
    assert estimator.get_params()["quantile"] == 0.5


def test_quantile_conformalizer_check_quantile_estimator_default() -> None:
    """Test default quantile estimator creation."""
    conformalizer = DummyQuantileConformalizer()

    estimator = conformalizer._check_quantile_estimator()

    assert isinstance(estimator, QuantileRegressor)
    assert estimator.get_params()["solver"] == "highs-ds"


def test_quantile_conformalizer_check_quantile_estimator_pipeline() -> None:
    """Test quantile estimator validation through a pipeline."""
    conformalizer = DummyQuantileConformalizer()
    estimator = make_pipeline(SimpleImputer(), QuantileRegressor(solver="highs-ds"))

    checked_estimator = conformalizer._check_quantile_estimator(estimator)

    assert isinstance(checked_estimator, Pipeline)
    assert checked_estimator[-1] is estimator[-1]


def test_quantile_conformalizer_get_estimator_name() -> None:
    """Test estimator name extraction for plain estimators and pipelines."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.estimator = QuantileRegressor(solver="highs-ds")
    assert conformalizer.get_estimator_name() == "QuantileRegressor"

    conformalizer.estimator = make_pipeline(
        SimpleImputer(), QuantileRegressor(solver="highs-ds")
    )
    assert conformalizer.get_estimator_name() == "QuantileRegressor"


def test_quantile_conformalizer_set_estimator_params() -> None:
    """Test in-place estimator parameter update."""
    conformalizer = DummyQuantileConformalizer()
    estimator = QuantileRegressor(solver="highs-ds", quantile=0.5)

    updated_estimator = conformalizer._set_estimator_params(estimator, quantile=0.3)

    assert updated_estimator is estimator
    assert estimator.get_params()["quantile"] == 0.3


def test_quantile_conformalizer_initialize_fit_conformalize() -> None:
    """Test conformalizer initialization state."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.alpha = [0.2]
    conformalizer.method = "base"
    conformalizer._central_estimator = None
    level = str(conformalizer.alpha[0])

    conformalizer._initialize_fit_conformalize()

    np.testing.assert_allclose(
        conformalizer.quantiles[level], np.array([0.1, 0.9, 0.5])
    )
    assert list(conformalizer.estimators_.keys()) == [level, "central"]
    assert conformalizer.estimators_ == {
        level: {"lower": [], "upper": []},
        "central": [],
    }
    assert conformalizer.estimators_[level] == {"lower": [], "upper": []}
    assert conformalizer.n_calib_samples == []
    np.testing.assert_array_equal(conformalizer.conformity_scores[level], np.array([]))
    assert conformalizer.pinball_losses[level] == []
    assert conformalizer.key_mapping == {"lower": 0, "upper": 1, "central": 2}
    assert conformalizer._base_estimator_ == {
        level: {"lower": [], "upper": []},
        "central": [],
    }


def test_quantile_conformalizer_fit_quantiles() -> None:
    """Test quantile estimator fitting returns populated estimators dict."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.cv = None
    conformalizer.alpha = [0.2]
    conformalizer.estimator = QuantileRegressor(solver="highs-ds")
    conformalizer._central_estimator = None
    conformalizer.fit_central_estimator = True
    conformalizer.method = "base"
    conformalizer._initialize_fit_conformalize()

    level = str(conformalizer.alpha[0])

    fitted_estimators = conformalizer._fit_quantiles(X_train_toy, y_train_toy, level)

    # _fit_quantiles returns a new dict and does not mutate self.estimators_.
    assert conformalizer.estimators_[level] == {"lower": [], "upper": []}
    assert list(conformalizer.estimators_.keys()) == [level, "central"]

    assert fitted_estimators.get("lower") is not None
    assert fitted_estimators.get("upper") is not None
    assert fitted_estimators.get("central") is not None
    assert fitted_estimators["lower"][0].get_params()["quantile"] == 0.1
    assert fitted_estimators["upper"][0].get_params()["quantile"] == 0.9
    assert fitted_estimators["central"][0].get_params()["quantile"] == 0.5


def test_quantile_conformalizer_fit_cv_estimator() -> None:
    """Test CV fitting helper returns one fitted estimator per quantile."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.cv = None
    conformalizer.alpha = [0.2]
    conformalizer.estimator = QuantileRegressor(solver="highs-ds")
    conformalizer._central_estimator = None
    conformalizer.fit_central_estimator = True
    conformalizer.method = "base"
    conformalizer._initialize_fit_conformalize()
    level = str(conformalizer.alpha[0])

    fitted_estimators = conformalizer._fit_cv_estimator(
        X_train_toy, y_train_toy, train_index=np.array([0, 1, 2, 3]), level=level
    )

    assert fitted_estimators.get("lower") is not None
    assert fitted_estimators.get("upper") is not None
    assert fitted_estimators.get("central") is not None
    assert fitted_estimators["lower"][0].get_params()["quantile"] == 0.1
    assert fitted_estimators["upper"][0].get_params()["quantile"] == 0.9
    assert fitted_estimators["central"][0].get_params()["quantile"] == 0.5


def test_quantile_conformalizer_fit() -> None:
    """Test fit populates CV estimators and k_ matrix."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.cv = KFold(n_splits=3)
    conformalizer.method = "plus"
    conformalizer.alpha = [0.2]
    conformalizer.estimator = QuantileRegressor(solver="highs-ds")
    conformalizer._central_estimator = None
    conformalizer.fit_central_estimator = True
    conformalizer.n_jobs = 1
    conformalizer.verbose = 0
    conformalizer._initialize_fit_conformalize()

    level = str(conformalizer.alpha[0])

    fitted_conformalizer = conformalizer.fit(X_train_toy, y_train_toy)

    assert fitted_conformalizer is conformalizer
    assert conformalizer.is_fitted
    assert conformalizer.k_.shape == (len(y_train_toy), 3)
    assert len(conformalizer.estimators_[level]["lower"]) == 3
    assert len(conformalizer.estimators_[level]["upper"]) == 3
    assert len(conformalizer.estimators_["central"]) == 3


def test_quantile_conformalizer_fit_with_base_method() -> None:
    """Test fit with method='base' also fits base quantile estimators."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.cv = KFold(n_splits=3)
    conformalizer.method = "base"
    conformalizer.alpha = [0.2]
    conformalizer.estimator = QuantileRegressor(solver="highs-ds")
    conformalizer._central_estimator = None
    conformalizer.fit_central_estimator = True
    conformalizer.n_jobs = 1
    conformalizer.verbose = 0
    conformalizer._initialize_fit_conformalize()
    level = str(conformalizer.alpha[0])

    fitted_conformalizer = conformalizer.fit(X_train_toy, y_train_toy)

    assert fitted_conformalizer is conformalizer
    assert conformalizer.is_fitted
    assert conformalizer.k_.shape == (len(y_train_toy), 3)
    assert len(conformalizer.estimators_[level]["lower"]) == 3
    assert len(conformalizer.estimators_[level]["upper"]) == 3
    assert len(conformalizer.estimators_["central"]) == 3
    assert hasattr(conformalizer, "_base_estimator_")
    assert set(conformalizer._base_estimator_.keys()) == {level, "central"}
    assert len(conformalizer._base_estimator_[level]["lower"]) == 1
    assert len(conformalizer._base_estimator_[level]["upper"]) == 1
    assert len(conformalizer._base_estimator_["central"]) == 1


def test_quantile_conformalizer_predict_quantiles() -> None:
    """Test stacked lower, upper and central quantile predictions."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.quantiles = {"0.2": np.array([0.1, 0.9, 0.5])}
    conformalizer.estimators_ = {
        "0.2": {
            "lower": [FixedPredictor(np.array([1.0, 2.0]))],
            "upper": [FixedPredictor(np.array([3.0, 4.0]))],
            "central": [FixedPredictor(np.array([2.0, 3.0]))],
        }
    }

    predictions = conformalizer._predict_quantiles(X_toy[:2], 0, level="0.2")

    assert predictions.shape == (3, 2)
    np.testing.assert_allclose(
        predictions,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [2.0, 3.0],
            ]
        ),
    )


def test_quantile_conformalizer_predict_center() -> None:
    """Test central predictor dispatch."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.estimators_ = {
        "lower": [],
        "upper": [],
        "central": [FixedPredictor(np.array([2.0, 3.0]))],
    }

    predictions = conformalizer._predict_center(X_toy[:2], 0)

    assert predictions.shape == (2,)
    np.testing.assert_allclose(predictions, np.array([2.0, 3.0]))


def test_quantile_conformalizer_predict_returns_center_lower_upper() -> None:
    """Test _predict returns center, lower, upper predictions in that order."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.quantiles = {"0.2": np.array([0.1, 0.9, 0.5])}
    conformalizer.method = "plus"
    conformalizer.agg_function = "mean"
    conformalizer.estimators_ = {
        "lower": [FixedPredictor(np.array([1.0, 2.0]))],
        "upper": [FixedPredictor(np.array([3.0, 4.0]))],
        "central": [FixedPredictor(np.array([2.0, 3.0]))],
    }

    y_pred_central, y_pred_low, y_pred_up = conformalizer._predict(
        X_toy[:2], ensemble=True
    )

    assert y_pred_central.shape == (2,)
    assert y_pred_low.shape == (2,)
    assert y_pred_up.shape == (2,)
    np.testing.assert_allclose(y_pred_central, np.array([2.0, 3.0]))
    np.testing.assert_allclose(y_pred_low, np.array([1.0, 2.0]))
    np.testing.assert_allclose(y_pred_up, np.array([3.0, 4.0]))


def test_quantile_conformalizer_predict_multiple_predictions_mean_aggregation() -> None:
    """Test _predict with multiple estimators aggregates predictions as expected."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.quantiles = {"0.2": np.array([0.1, 0.9, 0.5])}
    conformalizer.method = "plus"
    conformalizer.agg_function = "mean"
    conformalizer.estimators_ = {
        "0.2": {
            "lower": [
                FixedPredictor(np.array([1.0, 2.0])),
                FixedPredictor(np.array([3.0, 4.0])),
            ],
            "upper": [
                FixedPredictor(np.array([5.0, 6.0])),
                FixedPredictor(np.array([7.0, 8.0])),
            ],
        },
        "central": [
            FixedPredictor(np.array([2.0, 3.0])),
            FixedPredictor(np.array([4.0, 5.0])),
        ],
    }

    y_pred_central, y_pred_low, y_pred_up = conformalizer._predict(
        X_toy[:2], ensemble=True
    )

    np.testing.assert_allclose(y_pred_central, np.array([3.0, 4.0]))
    np.testing.assert_allclose(y_pred_low, np.array([2.0, 3.0]))
    np.testing.assert_allclose(y_pred_up, np.array([6.0, 7.0]))


def test_quantile_conformalizer_predict_multiple_predictions_minmax() -> None:
    """Test _predict minmax uses min lower, max upper and mean central predictions."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.quantiles = np.array([0.1, 0.9, 0.5])
    conformalizer.method = "minmax"
    conformalizer.agg_function = "mean"
    conformalizer.estimators_ = {
        "lower": [
            FixedPredictor(np.array([1.0, 4.0])),
            FixedPredictor(np.array([3.0, 2.0])),
        ],
        "upper": [
            FixedPredictor(np.array([5.0, 6.0])),
            FixedPredictor(np.array([7.0, 8.0])),
        ],
        "central": [
            FixedPredictor(np.array([2.0, 3.0])),
            FixedPredictor(np.array([4.0, 5.0])),
        ],
    }

    y_pred_central, y_pred_low, y_pred_up = conformalizer._predict(
        X_toy[:2], ensemble=True
    )

    np.testing.assert_allclose(y_pred_central, np.array([3.0, 4.0]))
    np.testing.assert_allclose(y_pred_low, np.array([1.0, 2.0]))
    np.testing.assert_allclose(y_pred_up, np.array([7.0, 8.0]))


def test_quantile_conformalizer_pinball_weighted_mean() -> None:
    """Test weighted aggregation from pinball losses."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.pinball_losses = np.array([1.0, 3.0])
    y_preds = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ]
    )

    weighted_mean = conformalizer._pinball_weighted_mean(y_preds)

    assert weighted_mean.shape == (2, 1)
    np.testing.assert_allclose(weighted_mean, np.array([[15.0], [25.0]]))


def test_quantile_conformalizer_pinball_weighted_mean_columnwise_weights() -> None:
    """Test weighting behavior when pinball losses are 2D."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.pinball_losses = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
        ]
    )
    y_preds = np.array(
        [
            [10.0, 100.0],
            [30.0, 200.0],
        ]
    )

    weighted_mean = conformalizer._pinball_weighted_mean(y_preds)

    assert weighted_mean.shape == (2, 1)
    np.testing.assert_allclose(weighted_mean, np.array([[15.0], [150.0]]))


def test_quantile_conformalizer_fit_prefit_sets_state() -> None:
    """Test fit in prefit mode initializes k_ and fitted state."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.cv = "prefit"
    conformalizer.alpha = [0.2]
    conformalizer._central_estimator = None
    conformalizer._initialize_fit_conformalize()

    fitted_conformalizer = conformalizer.fit(X_train_toy, y_train_toy)

    assert fitted_conformalizer is conformalizer
    assert conformalizer.is_fitted
    assert conformalizer.k_.shape == (len(y_train_toy), 1)
    assert np.isnan(conformalizer.k_).all()


def test_quantile_conformalizer_fit_twice_warns_and_resets() -> None:
    """Test second fit emits warning and clears previous conformalization state."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.cv = "prefit"
    conformalizer.alpha = [0.2]
    conformalizer.method = "base"
    conformalizer._central_estimator = None
    conformalizer._initialize_fit_conformalize()
    conformalizer.fit(X_train_toy, y_train_toy)

    conformalizer.is_conformalized = True
    conformalizer.conformity_scores = [np.array([42.0])]
    conformalizer.pinball_losses = [np.array([1.0, 2.0, 3.0])]

    with pytest.warns(
        UserWarning,
        match=r".*fit method has already been called.*",
    ):
        conformalizer.fit(X_train_toy, y_train_toy)

    assert conformalizer.is_fitted
    assert not conformalizer.is_conformalized
    np.testing.assert_array_equal(conformalizer.conformity_scores, np.array([]))
    assert conformalizer.pinball_losses == []


def test_quantile_conformalizer_predict_oof_empty_validation() -> None:
    """Test _predict_oof returns an empty array for empty validation index."""
    conformalizer = DummyQuantileConformalizer()

    predictions = conformalizer._predict_oof(
        X_toy[:2],
        np.array([], dtype=int),
        index=0,
    )

    assert predictions.size == 0


def test_quantile_conformalizer_predict_oof_calls_predict_quantiles() -> None:
    """Test _predict_oof delegates to predict_quantiles with provided index."""
    conformalizer = DummyQuantileConformalizer()
    call_args = {}

    def _mock_predict_quantiles(
        self: DummyQuantileConformalizer, X: NDArray, index: int, **kwargs: Any
    ) -> NDArray:
        call_args["X"] = X
        call_args["index"] = index
        call_args["kwargs"] = kwargs
        return np.array([[1.0], [2.0], [3.0]])

    conformalizer._predict_quantiles = MethodType(
        _mock_predict_quantiles, conformalizer
    )

    predictions = conformalizer._predict_oof(X_toy[:2], np.array([1]), index=7, a=1)

    assert predictions.shape == (3, 1)
    np.testing.assert_allclose(predictions, np.array([[1.0], [2.0], [3.0]]))
    np.testing.assert_allclose(call_args["X"], X_toy[[1]])
    assert call_args["index"] == 7
    assert call_args["kwargs"] == {"a": 1}


def test_quantile_conformalizer_conformalize_requires_fit() -> None:
    """Test conformalize raises when called before fit."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.is_fitted = False

    with pytest.raises(ValueError, match=r".*Incorrect method order*"):
        conformalizer.conformalize(X_toy[:2], y_toy[:2])


def test_quantile_conformalizer_conformalize_twice_warns_and_overwrites() -> None:
    """Test second conformalize call warns and overwrites previous scores."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.is_fitted = True
    conformalizer.is_conformalized = True
    conformalizer.score = AbsoluteQuantileRegressionScore()
    conformalizer.conformity_scores = [np.array([99.0])]

    def _mock_predict_calib(
        self: DummyQuantileConformalizer,
        X: NDArray,
        y: NDArray,
        groups: Any = None,
        **kwargs: Any,
    ) -> NDArray:
        return np.array(
            [
                [1.0, 3.0, 2.0],
                [4.0, 6.0, 5.0],
            ]
        )

    conformalizer._predict_calib = MethodType(_mock_predict_calib, conformalizer)

    with pytest.warns(
        UserWarning,
        match=r".*conformalize method has already been called.*",
    ):
        conformalizer.conformalize(X_toy[:2], y_toy[:2])

    assert conformalizer.is_conformalized
    assert conformalizer.conformity_scores.shape == (2,)
    np.testing.assert_allclose(conformalizer.conformity_scores, np.array([2.0, 1.0]))


def test_quantile_conformalizer_fit_cv_estimator_indexes_sample_weight() -> None:
    """Test _fit_cv_estimator forwards indexed sample_weight to _fit_quantiles."""
    conformalizer = DummyQuantileConformalizer()
    observed: dict[str, NDArray] = {}

    def _mock_fit_quantiles(
        self: DummyQuantileConformalizer,
        X: NDArray,
        y: NDArray,
        level: str,
        sample_weight: Optional[NDArray] = None,
        **fit_params: Any,
    ) -> dict[str, list]:
        observed["X"] = np.asarray(X)
        observed["y"] = np.asarray(y)
        observed["sample_weight"] = np.asarray(sample_weight)
        observed["fit_params"] = fit_params
        return {"lower": [], "upper": [], "central": []}

    conformalizer._fit_quantiles = MethodType(_mock_fit_quantiles, conformalizer)

    train_index = np.array([0, 2, 4])
    sample_weight = np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0])

    conformalizer._fit_cv_estimator(
        X_toy[:6],
        y_toy[:6],
        train_index=train_index,
        level="0.2",
        sample_weight=sample_weight,
        dummy=123,
    )

    np.testing.assert_allclose(observed["X"], X_toy[:6][train_index])
    np.testing.assert_allclose(observed["y"], y_toy[:6][train_index])
    np.testing.assert_allclose(observed["sample_weight"], sample_weight[train_index])
    assert observed["fit_params"] == {"dummy": 123}


def test_quantile_conformalizer_conformalize_forwards_groups_and_params() -> None:
    """Test conformalize forwards groups and predict kwargs to _predict_calib."""
    conformalizer = DummyQuantileConformalizer()
    observed: dict[str, Any] = {}
    conformalizer.is_fitted = True
    conformalizer.is_conformalized = False
    conformalizer.score = AbsoluteQuantileRegressionScore()
    conformalizer.conformity_scores = []

    def _mock_predict_calib(
        self: DummyQuantileConformalizer,
        X: NDArray,
        y: NDArray,
        groups: NDArray,
        **predict_params: Any,
    ) -> NDArray:
        observed["X"] = np.asarray(X)
        observed["y"] = np.asarray(y)
        observed["groups"] = np.asarray(groups)
        observed["predict_params"] = predict_params
        return np.array(
            [
                [1.0, 3.0, 2.0],
                [4.0, 6.0, 5.0],
            ]
        )

    conformalizer._predict_calib = MethodType(_mock_predict_calib, conformalizer)

    groups = np.array([10, 11])
    conformalizer.conformalize(X_toy[:2], y_toy[:2], groups=groups, foo="bar")

    np.testing.assert_allclose(observed["X"], X_toy[:2])
    np.testing.assert_allclose(observed["y"], y_toy[:2])
    np.testing.assert_allclose(observed["groups"], groups)
    assert observed["predict_params"] == {"foo": "bar"}
    assert conformalizer.is_conformalized
    assert conformalizer.conformity_scores.shape == (2,)
    np.testing.assert_allclose(conformalizer.conformity_scores, np.array([2.0, 1.0]))


def test_quantile_conformalizer_conformalize_score_shape_matches_n_samples() -> None:
    """Test conformity score shape matches calibration sample count."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.is_fitted = True
    conformalizer.is_conformalized = False
    conformalizer.score = AbsoluteQuantileRegressionScore()
    conformalizer.conformity_scores = []

    y_local = y_toy[:4]

    def _mock_predict_calib(
        self: DummyQuantileConformalizer,
        X: NDArray,
        y: NDArray,
        groups: Any = None,
        **kwargs: Any,
    ) -> NDArray:
        return np.array(
            [
                [4.0, 6.0],
                [5.0, 7.0],
                [6.0, 8.0],
                [7.0, 9.0],
            ]
        )

    conformalizer._predict_calib = MethodType(_mock_predict_calib, conformalizer)

    conformalizer.conformalize(X_toy[:4], y_local)

    assert conformalizer.conformity_scores.shape == (4,)
    np.testing.assert_allclose(
        conformalizer.conformity_scores, np.array([-1.0, 0.0, 1.0, 2.0])
    )


def test_quantile_conformalizer_reset_clears_runtime_state() -> None:
    """Test reset clears all runtime state fields used by fit/conformalize."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.is_fitted = True
    conformalizer.is_conformalized = True
    conformalizer.estimators_ = {
        "lower": [object()],
        "upper": [object()],
        "central": [object()],
    }
    conformalizer.n_calib_samples = [2, 3]
    conformalizer.conformity_scores = [np.array([1.0])]
    conformalizer.pinball_losses = [np.array([0.1, 0.2])]
    conformalizer._predict_params = {"x": 1}

    returned = conformalizer.reset()

    assert returned is conformalizer
    assert not conformalizer.is_fitted
    assert not conformalizer.is_conformalized
    assert conformalizer.estimators_ == {"lower": [], "upper": [], "central": []}
    assert conformalizer.n_calib_samples == []
    np.testing.assert_array_equal(conformalizer.conformity_scores, np.array([]))
    assert conformalizer.pinball_losses == []
    assert conformalizer._predict_params == {}


def test_quantile_conformalizer_predict_calib_prefit_shape_and_forwarding() -> None:
    """Test _predict_calib prefit path output shape and predict params forwarding."""
    conformalizer = DummyQuantileConformalizer()
    conformalizer.is_fitted = True
    conformalizer.cv = PrefitLikeCV()
    conformalizer.quantiles = np.array([0.1, 0.9, 0.5])

    observed: dict[str, Any] = {}

    def _mock_predict_quantiles(
        self: DummyQuantileConformalizer,
        X: NDArray,
        index: int,
        **predict_params: Any,
    ) -> NDArray:
        observed["X"] = np.asarray(X)
        observed["index"] = index
        observed["predict_params"] = predict_params
        return np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )

    conformalizer._predict_quantiles = MethodType(
        _mock_predict_quantiles, conformalizer
    )

    y_pred = conformalizer._predict_calib(X_toy[:4], y_toy[:4], key="value")

    assert y_pred.shape == (3, 4)
    assert conformalizer.n_calib_samples == [2, 2]
    np.testing.assert_allclose(observed["X"], X_toy[:4])
    assert observed["index"] == 0
    assert observed["predict_params"] == {"key": "value"}


def test_cross_conformalized_quantile_regressor_predict_interval_returns_expected_shape() -> (
    None
):
    """Test predict_interval returns (point, interval) arrays with expected shapes."""

    class StubScore:
        def predict_set(
            self, X: NDArray, alpha: NDArray, **kwargs: Any
        ) -> tuple[NDArray, NDArray, NDArray]:
            n_samples = X.shape[0]
            return (
                np.full(n_samples, 2.0),
                np.full(n_samples, 1.0),
                np.full(n_samples, 3.0),
            )

    reg = CrossConformalizedQuantileRegressor(
        estimator=qt,
        cv=KFold(n_splits=2),
        conformity_score=AbsoluteQuantileRegressionScore,
    )
    reg.is_conformalized = True
    reg.alpha = np.array([0.1])
    reg.conformity_scores_ = np.array([0.2, 0.3])
    reg.method = "plus"
    reg.score = StubScore()  # type: ignore[assignment]

    y_pred, y_pis = reg.predict_interval(
        X_toy[:3],
        aggregate_point_predictions=None,
        allow_infinite_bounds=True,
    )

    assert y_pred.shape == (3,)
    assert y_pis.shape == (3, 2)
    np.testing.assert_allclose(y_pred, np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(y_pis[:, 0], np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(y_pis[:, 1], np.array([3.0, 3.0, 3.0]))


def test_cross_conformalized_quantile_regressor_predict_mean_aggregation() -> None:
    """Test predict aggregates central fold predictions with a mean."""
    reg = CrossConformalizedQuantileRegressor(
        estimator=qt,
        cv=KFold(n_splits=2),
        conformity_score=AbsoluteQuantileRegressionScore,
    )
    reg.is_fitted = True
    reg.is_conformalized = True
    reg.estimators_ = {
        "lower": [],
        "upper": [],
        "central": [
            FixedPredictor(np.array([2.0, 3.0])),
            FixedPredictor(np.array([4.0, 5.0])),
        ],
    }

    y_pred = reg.predict(X_toy[:2], aggregate_point_predictions="mean")

    assert y_pred.shape == (2,)
    np.testing.assert_allclose(y_pred, np.array([3.0, 4.0]))


def test_cross_conformalized_quantile_regressor_predict_pinball_weighted_mean() -> None:
    """Test predict aggregates central fold predictions weighted by pinball losses on central quantile."""
    reg = CrossConformalizedQuantileRegressor(
        estimator=qt,
        cv=KFold(n_splits=2),
        conformity_score=AbsoluteQuantileRegressionScore,
    )
    reg.is_fitted = True
    reg.is_conformalized = True
    reg.pinball_losses = np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 3.0]])
    reg.quantiles = np.array(
        [0.05, 0.95, 0.5]
    )  # taille 3 → branche pinball_weighted_mean active
    reg.estimators_ = {
        "lower": [],
        "upper": [],
        "central": [
            FixedPredictor(np.array([10.0, 20.0])),
            FixedPredictor(np.array([30.0, 40.0])),
        ],
    }

    y_pred = reg.predict(X_toy[:2], aggregate_point_predictions="pinball_weighted_mean")

    assert y_pred.shape == (2,)
    np.testing.assert_allclose(y_pred, np.array([25.0, 35.0]))


def test_cross_conformalized_quantile_regressor_predict_interval_mean_aggregation() -> (
    None
):
    """Test predict_interval forwards mean aggregation as ensemble=True."""
    observed: dict[str, Any] = {}

    class StubScore:
        def predict_set(
            self, X: NDArray, alpha: NDArray, **kwargs: Any
        ) -> tuple[NDArray, NDArray, NDArray]:
            observed["kwargs"] = kwargs
            n_samples = X.shape[0]
            return (
                np.full(n_samples, 5.0),
                np.full(n_samples, 4.0),
                np.full(n_samples, 6.0),
            )

    reg = CrossConformalizedQuantileRegressor(
        estimator=qt,
        cv=KFold(n_splits=2),
        conformity_score=AbsoluteQuantileRegressionScore,
    )
    reg.is_conformalized = True
    reg.alpha = np.array([0.1])
    reg.conformity_scores_ = np.array([0.2, 0.3])
    reg.method = "plus"
    reg.score = StubScore()  # type: ignore[assignment]

    y_pred, y_pis = reg.predict_interval(
        X_toy[:3],
        aggregate_point_predictions="mean",
        allow_infinite_bounds=True,
    )

    assert observed["kwargs"]["ensemble"] is True
    assert y_pred.shape == (3,)
    assert y_pis.shape == (3, 2)
    np.testing.assert_allclose(y_pred, np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(y_pis[:, 0], np.array([4.0, 4.0, 4.0]))
    np.testing.assert_allclose(y_pis[:, 1], np.array([6.0, 6.0, 6.0]))


def test_cross_conformalized_quantile_regressor_predict_interval_pinball_weighted_mean() -> (
    None
):
    """Test predict_interval accepts pinball_weighted_mean aggregation."""
    observed: dict[str, Any] = {}

    class StubScore:
        def predict_set(
            self, X: NDArray, alpha: NDArray, **kwargs: Any
        ) -> tuple[NDArray, NDArray, NDArray]:
            observed["kwargs"] = kwargs
            n_samples = X.shape[0]
            return (
                np.full(n_samples, 8.0),
                np.full(n_samples, 7.0),
                np.full(n_samples, 9.0),
            )

    reg = CrossConformalizedQuantileRegressor(
        estimator=qt,
        cv=KFold(n_splits=2),
        conformity_score=AbsoluteQuantileRegressionScore,
    )
    reg.is_conformalized = True
    reg.alpha = np.array([0.1])
    reg.conformity_scores = np.array([0.2, 0.3])
    reg.method = "plus"
    reg.score = StubScore()  # type: ignore[assignment]

    y_pred, y_pis = reg.predict_interval(
        X_toy[:3],
        aggregate_point_predictions="pinball_weighted_mean",
        allow_infinite_bounds=True,
    )

    assert observed["kwargs"]["ensemble"] is True
    assert y_pred.shape == (3,)
    assert y_pis.shape == (3, 2)
    np.testing.assert_allclose(y_pred, np.array([8.0, 8.0, 8.0]))
    np.testing.assert_allclose(y_pis[:, 0], np.array([7.0, 7.0, 7.0]))
    np.testing.assert_allclose(y_pis[:, 1], np.array([9.0, 9.0, 9.0]))


def test_cross_conformalized_quantile_regressor_predict_interval_invalid_aggregation() -> (
    None
):
    """Test predict_interval raises on unsupported aggregation value."""
    reg = CrossConformalizedQuantileRegressor(
        estimator=qt,
        cv=KFold(n_splits=2),
        conformity_score=AbsoluteQuantileRegressionScore,
    )
    reg.is_conformalized = True
    reg.alpha = np.array([0.1])
    reg.conformity_scores_ = np.array([0.2, 0.3])

    with pytest.raises(
        ValueError,
        match=r".*The value of the aggregation function is not correct.*",
    ):
        reg.predict_interval(
            X_toy[:3],
            aggregate_point_predictions="unknown",
            allow_infinite_bounds=True,
        )


def test_cross_conformalized_quantile_regressor_base_method_predict_interval() -> None:
    """Test the base method returns point predictions and intervals."""
    reg = CrossConformalizedQuantileRegressor(
        estimator=qt,
        cv=KFold(n_splits=2),
        conformity_score=AbsoluteQuantileRegressionScore,
        method="base",
    )
    reg.fit_conformalize(X_toy[:6], y_toy[:6])

    y_pred, y_pis = reg.predict_interval(
        X_toy[:2],
        aggregate_point_predictions=None,
        allow_infinite_bounds=True,
    )

    assert y_pred.shape == (2,)
    assert y_pis.shape == (2, 2)
