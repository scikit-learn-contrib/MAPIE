from __future__ import annotations

from typing import Any, Tuple

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.utils.validation import check_is_fitted
from typing_extensions import TypedDict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.base import RegressorMixin


from mapie._typing import NDArray
from mapie.aggregation_functions import aggregate_all
from mapie.metrics import regression_coverage_score
from mapie.quantile_regression import MapieQuantileRegressor


X_toy = np.array(
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
     5, 0, 1, 2, 3, 4, 5]
).reshape(-1, 1)
y_toy = np.array(
    [5, 7, 9, 11, 13, 15, 5, 7, 9, 11, 13, 15, 5, 7, 9,
     11, 13, 15, 5, 7, 9, 11, 13, 15]
)

random_state = 1

qt = QuantileRegressor(solver='highs')
gb = GradientBoostingRegressor(
            loss="quantile",
            random_state=random_state
            )

X, y = make_regression(
    n_samples=500,
    n_features=10,
    noise=1.0,
    random_state=random_state
    )
k = np.ones(shape=(5, X.shape[1]))
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
    }

WIDTHS = {
    "quantile_alpha2": 385.68,
    "quantile_alpha3": 320.95,
    "quantile_alpha4": 271.85,
}

COVERAGES = {
    "quantile_alpha2": 0.81,
    "quantile_alpha3": 0.70,
    "quantile_alpha4": 0.61,
}


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = MapieQuantileRegressor()
    assert mapie_reg.agg_function == "mean"
    assert mapie_reg.method == "quantile"
    assert mapie_reg.alpha == 0.2


def test_default_parameters_estimator() -> None:
    """Test default values of input parameters."""
    mapie_reg = MapieQuantileRegressor()
    mapie_reg.fit(X_toy, y_toy)
    for estimator in mapie_reg.list_estimators:
        assert isinstance(estimator, QuantileRegressor)
        assert estimator.__dict__["solver"] == "highs"


def test_valid_estimator() -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    mapie_reg = MapieQuantileRegressor(
        estimator=qt)
    mapie_reg.fit(X_toy, y_toy)
    for estimator in mapie_reg.list_estimators:
        assert isinstance(estimator, QuantileRegressor)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
def test_valid_method(strategy: str, estimator: RegressorMixin) -> None:
    """Test that valid methods raise no errors."""
    mapie_reg = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    mapie_reg.fit(X_toy, y_toy)
    check_is_fitted(mapie_reg, mapie_reg.fit_attributes)
    assert mapie_reg.__dict__["method"] == "quantile"


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
@pytest.mark.parametrize("symmetry", SYMMETRY)
def test_predict_output_shape(
    strategy: str,
    estimator: RegressorMixin,
    dataset: Tuple[NDArray, NDArray],
    symmetry: bool
) -> None:
    """Test predict output shape."""
    mapie_reg = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    (X, y) = dataset
    mapie_reg.fit(X, y)
    y_pred, y_pis = mapie_reg.predict(X, symmetry=symmetry)
    assert y_pred.shape == (X.shape[0],)
    assert y_pis[:, 0, 0].shape == (X.shape[0],)
    assert y_pis[:, 1, 0].shape == (X.shape[0],)


# I don't understand why this one isn't working
@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
def test_results_with_constant_sample_weights(
    strategy: str,
    estimator: RegressorMixin
) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    mapie1 = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    mapie2 = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    mapie0.fit(X, y, sample_weight=None)
    mapie1.fit(X, y, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 5)

    np.testing.assert_allclose(
        mapie0.conformity_scores_,
        mapie1.conformity_scores_
    )
    np.testing.assert_allclose(
        mapie0.conformity_scores_,
        mapie2.conformity_scores_
    )

    y_pred0, y_pis0 = mapie0.predict(X)
    y_pred1, y_pis1 = mapie1.predict(X)
    y_pred2, y_pis2 = mapie2.predict(X)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)


# # Dosen't work just yet, need to set it as a warning
# @pytest.mark.parametrize("strategy", [*STRATEGIES])
# @pytest.mark.parametrize("estimator", ESTIMATOR)
# @pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
# @pytest.mark.parametrize("symmetry", SYMMETRY)
# def test_prediction_between_low_up(
#     strategy: str,
#     estimator: RegressorMixin,
#     dataset: Tuple[NDArray, NDArray],
#     symmetry: bool
# ) -> None:
#     """
#     Test that MapieRegressor applied on a linear regression model
#     fitted on a linear curve results in null uncertainty.
#     """
#     mapie = MapieQuantileRegressor(
#         estimator=estimator,
#         **STRATEGIES[strategy]
#         )
#     (X, y) = dataset
#     mapie.fit(X, y)
#     y_pred, y_pis = mapie.predict(X, symmetry=symmetry)
#     assert (y_pis[:, 1, 0] >= y_pis[:, 0, 0]).all()
#     assert (y_pred >= y_pis[:, 0, 0]).all()
#     assert (y_pred <= y_pis[:, 1, 0]).all()


def test_invalid_aggregate_all() -> None:
    """
    Test that wrong aggregation in MAPIE raise errors.
    """
    with pytest.raises(
        ValueError,
        match=r".*Aggregation function called but not defined.*",
    ):
        aggregate_all(None, X)


@pytest.mark.parametrize("njobs", [-1, 1, 5, 10])
def test_njobs_not_implemented(njobs: int) -> None:
    """Checking that njobs are not implemented."""
    with pytest.raises(
        NotImplementedError
    ):
        mapie_reg = MapieQuantileRegressor(n_jobs=njobs)
        mapie_reg.fit(X, y)


@pytest.mark.parametrize("alphas", ["hello", MapieQuantileRegressor, [2], 1])
def test_wrong_alphas_types(alphas: float) -> None:
    """Checking for wrong type of alphas"""
    with pytest.raises(
        ValueError,
        match=r".*Invalid alpha. Allowed values are floats.*",
    ):
        mapie_reg = MapieQuantileRegressor(alpha=alphas)
        mapie_reg.fit(X, y)


@pytest.mark.parametrize("alphas", [0.5, 0.6, 0.95, 5.0, -0.1, -0.001, -10.0])
def test_wrong_alphas(alphas: float) -> None:
    """Checking for alphas values that are too big."""
    with pytest.raises(
        ValueError,
        match=r".*The alpha value has to be lower than 0.5.*",
    ):
        mapie_reg = MapieQuantileRegressor(alpha=alphas)
        mapie_reg.fit(X, y)


def test_estimators_quantile_function() -> None:
    """Checking for alphas values that are too big."""
    with pytest.raises(
        ValueError,
        match=r".*You need to set the loss/metric*",
    ):
        mapie_reg = MapieQuantileRegressor(
            estimator=GradientBoostingRegressor()
            )
        mapie_reg.fit(X, y)


@pytest.mark.parametrize("cv", [-1, 2, KFold(), LeaveOneOut()])
def test_invalid_cv(cv: Any) -> None:
    """Test that valid cv raise errors."""
    with pytest.raises(
        ValueError,
        match=r".*Invalid cv method.*",
    ):
        mapie = MapieQuantileRegressor(cv=cv)
        mapie.fit(X_toy, y_toy)


@pytest.mark.parametrize("cv", [None, "split"])
def test_valid_cv(cv: Any) -> None:
    """Test that valid cv raise no errors."""
    mapie = MapieQuantileRegressor(cv=cv)
    mapie.fit(X_toy, y_toy)


def test_estimators_not_in_list() -> None:
    """Checking for alphas values that are too big."""
    with pytest.raises(
        ValueError,
        match=r".*We cannot find your method to have a link*",
    ):
        mapie_reg = MapieQuantileRegressor(estimator=RandomForestClassifier())
        mapie_reg.fit(X, y)


def test_for_small_dataset() -> None:
    """Checking for alphas values that are too big."""
    with pytest.raises(
        ValueError,
        match=r".*The calibration set is too small.*",
    ):
        mapie_reg = MapieQuantileRegressor(
            estimator=GradientBoostingRegressor(loss="quantile"),
            alpha=0.1
        )
        mapie_reg.fit([1, 2, 3], [1, 1, 1])


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("estimator", ESTIMATOR)
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
def test_conformity_len(
    strategy: str,
    estimator: RegressorMixin,
    dataset: Tuple[NDArray, NDArray]
) -> None:
    """
    Test...
    """
    (X, y) = dataset
    n_samples = int(len(X)/2)
    mapie_regressor = MapieQuantileRegressor(
        estimator=estimator,
        **STRATEGIES[strategy]
        )
    mapie_regressor.fit(X, y)
    np.testing.assert_allclose(
        mapie_regressor.conformity_scores_[0].shape,
        n_samples
        )


# Working but want to add both symmetry and different estimators
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_linear_regression_results(strategy: str) -> None:
    """
    Test expected prediction intervals for
    a multivariate linear regression problem
    with fixed random state.
    """
    mapie = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie.fit(X, y)
    _, y_pis = mapie.predict(X)
    y_pred_low, y_pred_up = y_pis[:, 0, 0], y_pis[:, 1, 0]
    width_mean = (y_pred_up - y_pred_low).mean()
    coverage = regression_coverage_score(y, y_pred_low, y_pred_up)
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-2)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-2)
