import pytest
import numpy as np

from mapie.conformity_scores import (
    AbsoluteConformityScore,
    ConformityScore,
    GammaConformityScore,
)
from mapie._typing import ArrayLike


X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])


def test_error_mother_class_initialization() -> None:
    with pytest.raises(TypeError):
        ConformityScore()


@pytest.mark.parametrize(
    "y_pred", [np.array([4, 7, 10, 12, 13, 12]), [4, 7, 10, 12, 13, 12]]
)
def test_absolute_conformity_score_get_conformity_scores(
    y_pred: ArrayLike,
) -> None:
    """Test conformity score computation for AbsoluteConformityScore."""
    abs_conf_score = AbsoluteConformityScore()
    signed_conf_scores = abs_conf_score.get_signed_conformity_scores(
        y_toy, y_pred
    )
    conf_scores = abs_conf_score.get_conformity_scores(y_toy, y_pred)
    expected_signed_conf_scores = np.array([1, 0, -1, -1, 0, 3])
    expected_conf_scores = np.abs(expected_signed_conf_scores)
    np.testing.assert_allclose(signed_conf_scores, expected_signed_conf_scores)
    np.testing.assert_allclose(conf_scores, expected_conf_scores)


@pytest.mark.parametrize(
    "y_pred", [np.array([4, 7, 10, 12, 13, 12]), [4, 7, 10, 12, 13, 12]]
)
@pytest.mark.parametrize(
    "conf_scores", [np.array([1, 0, -1, -1, 0, 3]), [1, 0, -1, -1, 0, 3]]
)
def test_absolute_conformity_score_get_observed_value(
    y_pred: ArrayLike, conf_scores: ArrayLike
) -> None:
    """Test conformity observed value computation for AbsoluteConformityScore."""  # noqa: E501
    abs_conf_score = AbsoluteConformityScore()
    y_obs = abs_conf_score.get_observed_value(y_pred, conf_scores)
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize(
    "y_pred", [np.array([4, 7, 10, 12, 13, 12]), [4, 7, 10, 12, 13, 12]]
)
def test_absolute_conformity_score_consistency(y_pred: ArrayLike) -> None:
    """Test methods consistency for AbsoluteConformityScore."""
    abs_conf_score = AbsoluteConformityScore()
    signed_conf_scores = abs_conf_score.get_signed_conformity_scores(
        y_toy, y_pred
    )
    y_obs = abs_conf_score.get_observed_value(y_pred, signed_conf_scores)
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize(
    "y_pred", [np.array([4, 7, 10, 12, 13, 12]), [4, 7, 10, 12, 13, 12]]
)
def test_gamma_conformity_score_get_conformity_scores(
    y_pred: ArrayLike,
) -> None:
    """Test conformity score computation for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    conf_scores = gamma_conf_score.get_conformity_scores(y_toy, y_pred)
    expected_signed_conf_scores = np.array(
        [1 / 4, 0, -1 / 10, -1 / 12, 0, 3 / 12]
    )
    np.testing.assert_allclose(conf_scores, expected_signed_conf_scores)


@pytest.mark.parametrize(
    "y_pred", [np.array([4, 7, 10, 12, 13, 12]), [4, 7, 10, 12, 13, 12]]
)
@pytest.mark.parametrize(
    "conf_scores",
    [
        np.array([1 / 4, 0, -1 / 10, -1 / 12, 0, 3 / 12]),
        [1 / 4, 0, -1 / 10, -1 / 12, 0, 3 / 12],
    ],
)
def test_gamma_conformity_score_get_observed_value(
    y_pred: ArrayLike, conf_scores: ArrayLike
) -> None:
    """Test conformity observed value computation for GammaConformityScore."""  # noqa: E501
    gamma_conf_score = GammaConformityScore()
    y_obs = gamma_conf_score.get_observed_value(y_pred, conf_scores)
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize(
    "y_pred", [np.array([4, 7, 10, 12, 13, 12]), [4, 7, 10, 12, 13, 12]]
)
def test_gamma_conformity_score_consistency(y_pred: ArrayLike) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    signed_conf_scores = gamma_conf_score.get_signed_conformity_scores(
        y_toy, y_pred
    )
    y_obs = gamma_conf_score.get_observed_value(y_pred, signed_conf_scores)
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize(
    "y_pred", [np.array([4, 7, 10, 12, 13, 12]), [4, 7, 10, 12, 13, 12]]
)
@pytest.mark.parametrize(
    "y_toy",
    [
        np.array([0, 7, 9, 11, 13, 15]),
        [0, 7, 9, 11, 13, 15],
        [1, -7, 9, 11, 13, 15],
    ],
)
def test_gamma_conformity_score_check_oberved_value(
    y_pred: ArrayLike, y_toy: ArrayLike
) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    with pytest.raises(ValueError):
        gamma_conf_score.get_signed_conformity_scores(y_toy, y_pred)


@pytest.mark.parametrize(
    "y_pred",
    [
        np.array([0, 7, 10, 12, 13, 12]),
        [0, 7, 10, 12, 13, 12],
        [1, -7, 10, 12, 13, 12],
    ],
)
@pytest.mark.parametrize(
    "conf_scores",
    [
        np.array([1 / 4, 0, -1 / 10, -1 / 12, 0, 3 / 12]),
        [1 / 4, 0, -1 / 10, -1 / 12, 0, 3 / 12],
    ],
)
def test_gamma_conformity_score_check_predicted_value(
    y_pred: ArrayLike, conf_scores: ArrayLike
) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    with pytest.raises(ValueError):
        gamma_conf_score.get_signed_conformity_scores(y_toy, y_pred)
    with pytest.raises(ValueError):
        gamma_conf_score.get_observed_value(y_pred, conf_scores)
