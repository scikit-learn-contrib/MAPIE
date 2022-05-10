from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.utils.validation import check_is_fitted
from typing_extensions import TypedDict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.base import RegressorMixin, clone


from mapie._typing import ArrayLike, NDArray
from mapie.aggregation_functions import aggregate_all
from mapie.metrics import regression_coverage_score
from mapie.subsample import Subsample
from mapie.quantile_regression import MapieQuantileRegressor


X_toy = np.array(
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
     5, 0, 1, 2, 3, 4, 5]
).reshape(-1, 1)
y_toy = np.array(
    [5, 7, 9, 11, 13, 15, 5, 7, 9, 11, 13, 15, 5, 7, 9,
     11, 13, 15, 5, 7, 9, 11, 13, 15]
)

X, y = make_regression(n_samples=500, n_features=10, noise=1.0, random_state=1)
k = np.ones(shape=(5, X.shape[1]))
SYMMETRY = [True, False]
random_state = 1

Params = TypedDict(
    "Params",
    {
        "estimator": RegressorMixin,
        "method": str,
        "alpha": float,
    },
)

STRATEGIES = {
    "quantile_alpha2": Params(method="quantile", alpha=0.2),
    "quantile_alpha3": Params(method="quantile", alpha=0.3),
    "quantile_alpha4": Params(method="quantile", alpha=0.4),
    "quantile__gb_alpha2": Params(
        estimator=GradientBoostingRegressor(
            loss="quantile",
            random_state=random_state
            ),
        method="quantile",
        alpha=0.2
        ),
    "quantile_gb_alpha3": Params(
        estimator=GradientBoostingRegressor(
            loss="quantile",
            random_state=random_state
            ),
        method="quantile",
        alpha=0.3
        ),
    "quantile_gb_alpha4": Params(
        estimator=GradientBoostingRegressor(
            loss="quantile",
            random_state=random_state
            ),
        method="quantile",
        alpha=0.4
        ),
    }

WIDTHS = {
    "quantile_alpha2": 386.3,
    "quantile_alpha3": 320.95,
    "quantile_alpha4": 271.85,
    "quantile__gb_alpha2": 271.38,
    "quantile__gb_alpha3": 191.75,
    "quantile__gb_alpha4": 138.37,
}

COVERAGES = {
    "quantile_alpha2": 0.81,
    "quantile_alpha3": 0.7,
    "quantile_alpha4": 0.61,
    "quantile__gb_alpha2": 0.9,
    "quantile__gb_alpha3": 0.85,
    "quantile__gb_alpha4": 0.79,
}


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_reg = MapieQuantileRegressor()
    assert mapie_reg.agg_function == "mean"
    assert mapie_reg.method == "quantile"
    assert mapie_reg.alpha == 0.2


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_estimator(strategy: str) -> None:
    """Test that valid estimators are not corrupted, for all strategies."""
    mapie_reg = MapieQuantileRegressor(
        estimator=QuantileRegressor())
    mapie_reg.fit(X_toy, y_toy)
    for estimator in mapie_reg.list_estimators:
        assert isinstance(estimator, QuantileRegressor)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_valid_method(strategy: str) -> None:
    """Test that valid methods raise no errors."""
    mapie_reg = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie_reg.fit(X_toy, y_toy)
    check_is_fitted(mapie_reg, mapie_reg.fit_attributes)


@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
@pytest.mark.parametrize("symmetry", SYMMETRY)
def test_predict_output_shape(
    strategy: str, dataset: Tuple[NDArray, NDArray], symmetry: bool
) -> None:
    """Test predict output shape."""
    mapie_reg = MapieQuantileRegressor(**STRATEGIES[strategy])
    (X, y) = dataset
    mapie_reg.fit(X, y)
    y_pred, y_pis = mapie_reg.predict(X, symmetry=symmetry)
    assert y_pred.shape == (X.shape[0],)
    assert y_pis[:, 0, 0].shape == (X.shape[0],)
    assert y_pis[:, 1, 0].shape == (X.shape[0],)


# I don't understand why this one isn't working
@pytest.mark.parametrize("strategy", [*STRATEGIES])
def test_results_with_constant_sample_weights(strategy: str) -> None:
    """
    Test predictions when sample weights are None
    or constant with different values.
    """
    n_samples = len(X)
    mapie0 = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie1 = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie2 = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie0.fit(X, y, sample_weight=None)
    mapie1.fit(X, y, sample_weight=np.ones(shape=n_samples))
    mapie2.fit(X, y, sample_weight=np.ones(shape=n_samples) * 5)
    y_pred0, y_pis0 = mapie0.predict(X)
    y_pred1, y_pis1 = mapie1.predict(X)
    y_pred2, y_pis2 = mapie2.predict(X)
    np.testing.assert_allclose(y_pred0, y_pred1)
    np.testing.assert_allclose(y_pred1, y_pred2)
    np.testing.assert_allclose(y_pis0, y_pis1)
    np.testing.assert_allclose(y_pis1, y_pis2)


# # Dosen't work just yet, need to set it as a warning
# @pytest.mark.parametrize("strategy", [*STRATEGIES])
# @pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
# @pytest.mark.parametrize("symmetry", SYMMETRY)
# def test_prediction_between_low_up(
#     strategy: str, dataset: Tuple[NDArray, NDArray], symmetry: bool
# ) -> None:
#     """
#     Test that MapieRegressor applied on a linear regression model
#     fitted on a linear curve results in null uncertainty.
#     """
#     mapie = MapieQuantileRegressor(**STRATEGIES[strategy])
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
        mapie_reg = MapieQuantileRegressor(estimator=GradientBoostingRegressor())
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


# I don't understand why this one isn't working
@pytest.mark.parametrize("strategy", [*STRATEGIES])
@pytest.mark.parametrize("dataset", [(X, y), (X_toy, y_toy)])
@pytest.mark.parametrize("symmetry", SYMMETRY)
def test_conformity_len(
    strategy: str, dataset: Tuple[NDArray, NDArray], symmetry: bool
) -> None:
    """
    Test...
    """
    n_samples = int(len(X)/2)
    mapie_regressor = MapieQuantileRegressor(**STRATEGIES[strategy])
    mapie_regressor.fit(X, y)
    np.testing.assert_allclose(
        mapie_regressor.conformity_scores_[0].shape,
        n_samples
        )


# Not sure why it dosen't work
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
    np.testing.assert_allclose(width_mean, WIDTHS[strategy], rtol=1e-1)
    np.testing.assert_allclose(coverage, COVERAGES[strategy], rtol=1e-1)

# Extra ones?
"""
    Test that MapieRegressor applied on a linear regression model
    fitted on a linear curve results in null uncertainty.
"""

# @pytest.mark.parametrize("strategy", [*STRATEGIES])
# def test_results_for_same_alpha(strategy: str) -> None:
#     """
#     Test that predictions and intervals
#     are similar with two equal values of alpha.
#     """
#     mapie_reg = MapieRegressor(**STRATEGIES[strategy])
#     mapie_reg.fit(X, y)
#     _, y_pis = mapie_reg.predict(X, alpha=[0.1, 0.1])
#     np.testing.assert_allclose(y_pis[:, 0, 0], y_pis[:, 0, 1])
#     np.testing.assert_allclose(y_pis[:, 1, 0], y_pis[:, 1, 1])


# def test_results_prefit_ignore_method() -> None:
#     """Test that method is ignored when ``cv="prefit"``."""
#     estimator = LinearRegression().fit(X, y)
#     all_y_pis: List[NDArray] = []
#     for method in METHODS:
#         mapie_reg = MapieRegressor(
#             estimator=estimator, cv="prefit", method=method
#         )
#         mapie_reg.fit(X, y)
#         _, y_pis = mapie_reg.predict(X, alpha=0.1)
#         all_y_pis.append(y_pis)
#     for y_pis1, y_pis2 in combinations(all_y_pis, 2):
#         np.testing.assert_allclose(y_pis1, y_pis2)


# def test_results_prefit_naive() -> None:
#     """
#     Test that prefit, fit and predict on the same dataset
#     is equivalent to the "naive" method.
#     """
#     estimator = LinearRegression().fit(X, y)
#     mapie_reg = MapieRegressor(estimator=estimator, cv="prefit")
#     mapie_reg.fit(X, y)
#     _, y_pis = mapie_reg.predict(X, alpha=0.05)
#     width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
#     coverage = regression_coverage_score(y, y_pis[:, 0, 0], y_pis[:, 1, 0])
#     np.testing.assert_allclose(width_mean, WIDTHS["naive"], rtol=1e-2)
#     np.testing.assert_allclose(coverage, COVERAGES["naive"], rtol=1e-2)

# @pytest.mark.parametrize("strategy", [*STRATEGIES])
# def test_results_prefit(strategy: str) -> None:
#     """Test prefit results on a standard train/validation/test split."""
#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         X, y, test_size=1 / 10, random_state=1
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_val, y_train_val, test_size=1 / 9, random_state=1
#     )
#     mapie_reg = MapieQuantileRegressor(**STRATEGIES[strategy])
#     mapie_reg.fit(X_val, y_val)
#     _, y_pis = mapie_reg.predict(X_test)
#     width_mean = (y_pis[:, 1, 0] - y_pis[:, 0, 0]).mean()
#     coverage = regression_coverage_score(
#         y_test, y_pis[:, 0, 0], y_pis[:, 1, 0]
#     )
#     np.testing.assert_allclose(width_mean, WIDTHS["prefit"], rtol=1e-2)
#     np.testing.assert_allclose(coverage, COVERAGES["prefit"], rtol=1e-2)


# def test_not_enough_resamplings() -> None:
#     """Test that a warning is raised if at least one residual is nan."""
#     with pytest.warns(UserWarning, match=r"WARNING: at least one point of*"):
#         mapie_reg = MapieRegressor(
#             cv=Subsample(n_resamplings=1), agg_function="mean"
#         )
#         mapie_reg.fit(X, y)


# def test_no_agg_fx_specified_with_subsample() -> None:
#     """Test that a warning is raised if at least one residual is nan."""
#     with pytest.raises(
#         ValueError, match=r"You need to specify an aggregation*"
#     ):
#         mapie_reg = MapieRegressor(
#             cv=Subsample(n_resamplings=1), agg_function=None
#         )
#         mapie_reg.fit(X, y)


# def test_aggregate_with_mask_with_prefit() -> None:
#     """
#     Test ``aggregate_with_mask`` in case ``cv`` is ``"prefit"``.
#     """
#     mapie_reg = MapieRegressor(cv="prefit")
#     with pytest.raises(
#         ValueError,
#         match=r".*There should not be aggregation of predictions if cv is*",
#     ):
#         mapie_reg.aggregate_with_mask(k, k)

#     mapie_reg = MapieRegressor(agg_function="nonsense")
#     with pytest.raises(
#         ValueError,
#         match=r".*The value of self.agg_function is not correct*",
#     ):
#         mapie_reg.aggregate_with_mask(k, k)


# def test_pred_loof_isnan() -> None:
#     """Test that if validation set is empty then prediction is empty."""
#     mapie_reg = MapieRegressor()
#     y_pred: ArrayLike
#     _, y_pred, _ = mapie_reg._fit_and_predict_oof_model(
#         estimator=LinearRegression(),
#         X=X_toy,
#         y=y_toy,
#         train_index=[0, 1, 2, 3, 4],
#         val_index=[],
#     )
#     assert len(y_pred) == 0


# def test_pipeline_compatibility() -> None:
#     """Check that MAPIE works on pipeline based on pandas dataframes"""
#     X = pd.DataFrame(
#         {
#             "x_cat": ["A", "A", "B", "A", "A", "B"],
#             "x_num": [0, 1, 1, 4, np.nan, 5],
#             "y": [5, 7, 3, 9, 10, 8]
#         }
#     )
#     y = pd.Series([5, 7, 3, 9, 10, 8])
#     numeric_preprocessor = Pipeline(
#         [
#             ("imputer", SimpleImputer(strategy="mean")),
#         ]
#     )
#     categorical_preprocessor = Pipeline(
#         steps=[
#             ("encoding", OneHotEncoder(handle_unknown="ignore"))
#         ]
#     )
#     preprocessor = ColumnTransformer(
#         [
#             ("cat", categorical_preprocessor, ["x_cat"]),
#             ("num", numeric_preprocessor, ["x_num"])
#         ]
#     )
#     pipe = make_pipeline(preprocessor, LinearRegression())
#     mapie = MapieRegressor(pipe)
#     mapie.fit(X, y)
#     mapie.predict(X)

# @pytest.mark.parametrize("agg_function", ["dummy", 0, 1, 2.5, [1, 2]])
# def test_invalid_agg_function(agg_function: Any) -> None:
#     """Test that invalid agg_functions raise errors."""
#     mapie_reg = MapieQuantileRegressor(agg_function=agg_function)
#     with pytest.raises(ValueError, match=r".*Invalid aggregation function.*"):
#         mapie_reg.fit(X_toy, y_toy)

#     mapie_reg = MapieQuantileRegressor(agg_function=None)
#     with pytest.raises(ValueError, match=r".*If ensemble is True*"):
#         mapie_reg.fit(X_toy, y_toy)
#         mapie_reg.predict(X_toy, ensemble=True)


# @pytest.mark.parametrize("agg_function", [None, "mean", "median"])
# def test_valid_agg_function(agg_function: str) -> None:
#     """Test that valid agg_functions raise no errors."""
#     mapie_reg = MapieQuantileRegressor(agg_function=agg_function)
#     mapie_reg.fit(X_toy, y_toy)

# @pytest.mark.parametrize("cv", [100, 200, 300])
# def test_too_large_cv(cv: Any) -> None:
#     """Test that too large cv raise sklearn errors."""
#     mapie_reg = MapieQuantileRegressor(cv=cv)
#     with pytest.raises(
#         ValueError,
#         match=rf".*Cannot have number of splits n_splits={cv} greater.*",
#     ):
#         mapie_reg.fit(X_toy, y_toy)