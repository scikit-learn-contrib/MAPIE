from typing import Any

import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import (AbsoluteConformityScore,
                                     ConformityScore,
                                     GammaConformityScore,
                                     ResidualNormalisedScore)
from mapie.regression import MapieRegressor

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
y_pred_list = [4, 7, 10, 12, 13, 12]
conf_scores_list = [1, 0, -1, -1, 0, 3]
conf_scores_gamma_list = [1 / 4, 0, -1 / 10, -1 / 12, 0, 3 / 12]
conf_scores_residual_norm_list = [0.2, 0., 0.11111111, 0.09090909, 0., 0.2]
random_state = 42


class DummyConformityScore(ConformityScore):
    def __init__(self) -> None:
        super().__init__(sym=True, consistency_check=True)

    def get_signed_conformity_scores(
        self, X: ArrayLike, y: ArrayLike, y_pred: ArrayLike,
    ) -> NDArray:
        return np.subtract(y, y_pred)

    def get_estimation_distribution(
        self, X: ArrayLike, y_pred: ArrayLike, conformity_scores: ArrayLike
    ) -> NDArray:
        """
        A positive constant is added to the sum between predictions and
        conformity scores to make the estimated distribution inconsistent
        with the conformity score.
        """
        return np.add(y_pred, conformity_scores) + 1


@pytest.mark.parametrize("sym", [False, True])
def test_error_mother_class_initialization(sym: bool) -> None:
    with pytest.raises(TypeError):
        ConformityScore(sym)  # type: ignore


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_absolute_conformity_score_get_conformity_scores(
    y_pred: NDArray,
) -> None:
    """Test conformity score computation for AbsoluteConformityScore."""
    abs_conf_score = AbsoluteConformityScore()
    signed_conf_scores = abs_conf_score.get_signed_conformity_scores(
        X_toy, y_toy, y_pred
    )
    conf_scores = abs_conf_score.get_conformity_scores(
        X_toy, y_toy, y_pred
    )
    expected_signed_conf_scores = np.array(conf_scores_list)
    expected_conf_scores = np.abs(expected_signed_conf_scores)
    np.testing.assert_allclose(signed_conf_scores, expected_signed_conf_scores)
    np.testing.assert_allclose(conf_scores, expected_conf_scores)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
@pytest.mark.parametrize(
    "conf_scores", [np.array(conf_scores_list), conf_scores_list]
)
def test_absolute_conformity_score_get_estimation_distribution(
    y_pred: NDArray, conf_scores: NDArray
) -> None:
    """Test conformity observed value computation for AbsoluteConformityScore."""  # noqa: E501
    abs_conf_score = AbsoluteConformityScore()
    y_obs = abs_conf_score.get_estimation_distribution(
        X_toy, y_pred, conf_scores
    )
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_absolute_conformity_score_consistency(y_pred: NDArray) -> None:
    """Test methods consistency for AbsoluteConformityScore."""
    abs_conf_score = AbsoluteConformityScore()
    signed_conf_scores = abs_conf_score.get_signed_conformity_scores(
        X_toy, y_toy, y_pred
    )
    y_obs = abs_conf_score.get_estimation_distribution(
        X_toy, y_pred, signed_conf_scores
    )
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_gamma_conformity_score_get_conformity_scores(
    y_pred: NDArray,
) -> None:
    """Test conformity score computation for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    conf_scores = gamma_conf_score.get_conformity_scores(
        X_toy, y_toy, y_pred
    )
    expected_signed_conf_scores = np.array(conf_scores_gamma_list)
    np.testing.assert_allclose(conf_scores, expected_signed_conf_scores)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
@pytest.mark.parametrize(
    "conf_scores",
    [
        np.array(conf_scores_gamma_list),
        conf_scores_gamma_list,
    ],
)
def test_gamma_conformity_score_get_estimation_distribution(
    y_pred: NDArray, conf_scores: NDArray
) -> None:
    """Test conformity observed value computation for GammaConformityScore."""  # noqa: E501
    gamma_conf_score = GammaConformityScore()
    y_obs = gamma_conf_score.get_estimation_distribution(
        X_toy, y_pred, conf_scores
    )
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_gamma_conformity_score_consistency(y_pred: NDArray) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    signed_conf_scores = gamma_conf_score.get_signed_conformity_scores(
        X_toy, y_toy, y_pred
    )
    y_obs = gamma_conf_score.get_estimation_distribution(
        X_toy, y_pred, signed_conf_scores
    )
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
@pytest.mark.parametrize(
    "y_toy",
    [
        np.array([0, 7, 9, 11, 13, 15]),
        [0, 7, 9, 11, 13, 15],
        [1, -7, 9, 11, 13, 15],
    ],
)
def test_gamma_conformity_score_check_oberved_value(
    y_pred: NDArray, y_toy: NDArray
) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    with pytest.raises(ValueError):
        gamma_conf_score.get_signed_conformity_scores(
            [], y_toy, y_pred
        )


@pytest.mark.parametrize(
    "y_pred",
    [
        np.array([0, 7, 10, 12, 13, 12]),
        [0, 7, 10, 12, 13, 12],
        [1, -7, 10, 12, 13, 12],
    ],
)
@pytest.mark.parametrize(
    "X_toy",
    [
        np.array([0, -7, 10, 12, 0, 12]).reshape(-1, 1),
        np.array([0, 7, -10, 12, 1, -12]).reshape(-1, 1),
        np.array([12, -7, 0, 12, 13, 2]).reshape(-1, 1),
    ],
)
@pytest.mark.parametrize(
    "conf_scores",
    [
        np.array(conf_scores_gamma_list),
        conf_scores_gamma_list,
    ],
)
def test_gamma_conformity_score_check_predicted_value(
    y_pred: NDArray, conf_scores: NDArray, X_toy: NDArray
) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    with pytest.raises(
        ValueError,
        match=r".*At least one of the predicted target is negative.*"
    ):
        gamma_conf_score.get_signed_conformity_scores(
            X_toy, y_toy, y_pred
        )
    with pytest.raises(
        ValueError,
        match=r".*At least one of the predicted target is negative.*"
    ):
        gamma_conf_score.get_estimation_distribution(
            X_toy, y_pred, conf_scores
        )


def test_check_consistency() -> None:
    """
    Test that a dummy ConformityScore class that gives inconsistent scores
    and distributions raises an error.
    """
    dummy_conf_score = DummyConformityScore()
    conformity_scores = dummy_conf_score.get_signed_conformity_scores(
        X_toy, y_toy, y_pred_list
    )
    with pytest.raises(
        ValueError,
        match=r".*The two functions get_conformity_scores.*"
    ):
        dummy_conf_score.check_consistency(
            X_toy, y_toy, y_pred_list, conformity_scores
        )


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_residual_normalised_prefit_conformity_score_get_conformity_scores(
    y_pred: NDArray
) -> None:
    """
    Test conformity score computation for ResidualNormalisedScore
    when prefit is True.
    """
    residual_estimator = LinearRegression().fit(X_toy, y_toy)
    residual_norm_conf_score = ResidualNormalisedScore(
        residual_estimator=residual_estimator,
        prefit=True,
        random_state=random_state
    )
    conf_scores = residual_norm_conf_score.get_conformity_scores(
        X_toy, y_toy, y_pred
    )
    expected_signed_conf_scores = np.array(conf_scores_residual_norm_list)
    np.testing.assert_allclose(conf_scores, expected_signed_conf_scores)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_residual_normalised_conformity_score_get_conformity_scores(
    y_pred: NDArray
) -> None:
    """
    Test conformity score computation for ResidualNormalisedScore
    when prefit is False.
    """
    residual_norm_score = ResidualNormalisedScore(random_state=random_state)
    conf_scores = residual_norm_score.get_conformity_scores(
        X_toy, y_toy, y_pred
    )
    expected_signed_conf_scores = np.array(
        [np.nan, np.nan, 1.e+08, 1.e+08, 0.e+00, 3.e+08]
    )
    np.testing.assert_allclose(conf_scores, expected_signed_conf_scores)


def test_residual_normalised_score_prefit_with_notfitted_estim() -> None:
    """Test that a not fitted estimator and prefit=True raises an error."""
    residual_norm_conf_score = ResidualNormalisedScore(
        residual_estimator=LinearRegression(), prefit=True
    )
    with pytest.raises(ValueError):
        residual_norm_conf_score.get_conformity_scores(
            X_toy, y_toy, y_pred_list
        )


def test_residual_normalised_score_with_default_params() -> None:
    """Test that no error is raised with default parameters."""
    residual_norm_score = ResidualNormalisedScore()
    conf_scores = residual_norm_score.get_conformity_scores(
        X_toy, y_toy, y_pred_list
    )
    residual_norm_score.get_estimation_distribution(X_toy, y_toy, conf_scores)


def test_invalid_estimator() -> None:
    """Test that an estimator without predict method raises an error."""
    class DumbEstimator:
        def __init__(self):
            pass

    residual_norm_conf_score = ResidualNormalisedScore(
        residual_estimator=DumbEstimator()
    )
    with pytest.raises(ValueError):
        residual_norm_conf_score.get_conformity_scores(
            X_toy, y_toy, y_pred_list
        )


def test_cross_residual_normalised() -> None:
    """
    Test that residual normalised score with cross method raises an error.
    """
    with pytest.raises(ValueError):
        MapieRegressor(conformity_score=ResidualNormalisedScore()).fit(
            X_toy, y_toy
        )


def test_residual_normalised_score_pipe() -> None:
    """
    Test that residual normalised score function raises no error
    with a pipeline estimator.
    """
    pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("linear", LinearRegression())
        ])
    mapie_reg = MapieRegressor(
        conformity_score=ResidualNormalisedScore(
            residual_estimator=pipe, split_size=0.2
        ),
        cv="split",
        random_state=random_state
    )
    mapie_reg.fit(np.concatenate((X_toy, X_toy)),
                  np.concatenate((y_toy, y_toy)))


def test_residual_normalised_score_pipe_prefit() -> None:
    """
    Test that residual normalised score function raises no error with a
    pipeline estimator prefitted.
    """
    pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("linear", LinearRegression())
        ])
    pipe.fit(X_toy, y_toy)
    mapie_reg = MapieRegressor(
        conformity_score=ResidualNormalisedScore(
            residual_estimator=pipe, split_size=0.2, prefit=True
        ),
        cv="split",
        random_state=random_state
    )
    mapie_reg.fit(X_toy, y_toy)


def test_residual_normalised_prefit_estimator_with_neg_values() -> None:
    """
    Test that a prefit estimator for the residual estimator of the residual
    normalised score that predicts negative values raises a warning.
    """
    class NegativeRegresssor(LinearRegression):
        def predict(self, X):
            return np.full(X.shape[0], fill_value=-1.)
    estim = NegativeRegresssor().fit(X_toy, y_toy)
    residual_norm_conf_score = ResidualNormalisedScore(
        residual_estimator=estim, prefit=True
    )
    with pytest.warns(UserWarning):
        residual_norm_conf_score.get_conformity_scores(
            X_toy, y_toy, y_pred_list
        )


def test_residual_normalised_prefit_get_estimation_distribution() -> None:
    """
    Test that get_estimation_distribution with prefitted estimator in residual
    normalised score raises no error.
    """
    estim = LinearRegression().fit(X_toy, y_toy)
    residual_normalised_conf_score = ResidualNormalisedScore(
        residual_estimator=estim, prefit=True
    )
    conf_scores = residual_normalised_conf_score.get_conformity_scores(
        X_toy, y_toy, y_pred_list
    )
    residual_normalised_conf_score.get_estimation_distribution(
        X_toy, y_pred_list, conf_scores
    )


@pytest.mark.parametrize("score", [AbsoluteConformityScore(),
                                   GammaConformityScore(),
                                   ResidualNormalisedScore()])
@pytest.mark.parametrize("alpha", [[0.3], [0.5, 0.4]])
def test_intervals_shape_with_every_score(
    score: ConformityScore,
    alpha: Any
) -> None:
    mapie_reg = MapieRegressor(
        method="base", cv="split", conformity_score=score
    )
    X = np.concatenate((X_toy, X_toy))
    y = np.concatenate((y_toy, y_toy))
    mapie_reg = mapie_reg.fit(X, y)
    y_pred, intervals = mapie_reg.predict(X, alpha=alpha)
    n_samples = X.shape[0]
    assert y_pred.shape[0] == n_samples
    assert intervals.shape == (n_samples, 2, len(alpha))
