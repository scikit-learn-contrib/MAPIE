import pytest
import numpy as np

from mapie.conformity_scores import (
    AbsoluteConformityScore,
    ConformityScore,
    GammaConformityScore,
)
from mapie._typing import NDArray, ArrayLike


X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
y_pred_list = [4, 7, 10, 12, 13, 12]
conf_scores_list = [1, 0, -1, -1, 0, 3]
conf_scores_gamma_list = [1 / 4, 0, -1 / 10, -1 / 12, 0, 3 / 12]


class DummyConformityScore(ConformityScore):
    def __init__(self) -> None:
        super().__init__(sym=True, consistency_check=True)

    def get_signed_conformity_scores(
        self, y: ArrayLike, y_pred: ArrayLike,
    ) -> NDArray:
        return np.subtract(y, y_pred)

    def get_estimation_distribution(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike
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
        y_toy, y_pred
    )
    conf_scores = abs_conf_score.get_conformity_scores(y_toy, y_pred)
    expected_signed_conf_scores = np.array([1, 0, -1, -1, 0, 3])
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
    y_obs = abs_conf_score.get_estimation_distribution(y_pred, conf_scores)
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_absolute_conformity_score_consistency(y_pred: NDArray) -> None:
    """Test methods consistency for AbsoluteConformityScore."""
    abs_conf_score = AbsoluteConformityScore()
    signed_conf_scores = abs_conf_score.get_signed_conformity_scores(
        y_toy, y_pred
    )
    y_obs = abs_conf_score.get_estimation_distribution(
        y_pred, signed_conf_scores
    )
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_gamma_conformity_score_get_conformity_scores(
    y_pred: NDArray,
) -> None:
    """Test conformity score computation for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    conf_scores = gamma_conf_score.get_conformity_scores(y_toy, y_pred)
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
    y_obs = gamma_conf_score.get_estimation_distribution(y_pred, conf_scores)
    np.testing.assert_allclose(y_obs, y_toy)


@pytest.mark.parametrize("y_pred", [np.array(y_pred_list), y_pred_list])
def test_gamma_conformity_score_consistency(y_pred: NDArray) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    signed_conf_scores = gamma_conf_score.get_signed_conformity_scores(
        y_toy, y_pred
    )
    y_obs = gamma_conf_score.get_estimation_distribution(
        y_pred, signed_conf_scores
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
        np.array(conf_scores_gamma_list),
        conf_scores_gamma_list,
    ],
)
def test_gamma_conformity_score_check_predicted_value(
    y_pred: NDArray, conf_scores: NDArray
) -> None:
    """Test methods consistency for GammaConformityScore."""
    gamma_conf_score = GammaConformityScore()
    with pytest.raises(
        ValueError,
        match=r".*At least one of the predicted target is negative.*"
    ):
        gamma_conf_score.get_signed_conformity_scores(y_toy, y_pred)
    with pytest.raises(
        ValueError,
        match=r".*At least one of the predicted target is negative.*"
    ):
        gamma_conf_score.get_estimation_distribution(y_pred, conf_scores)


def test_check_consistency() -> None:
    """
    Test that a dummy ConformityScore class that gives inconsistent conformity
    scores and distributions raises an error.
    """
    dummy_conf_score = DummyConformityScore()
    conformity_scores = dummy_conf_score.get_signed_conformity_scores(
        y_toy, y_pred_list
    )
    with pytest.raises(
        ValueError,
        match=r".*The two functions get_conformity_scores.*"
    ):
        dummy_conf_score.check_consistency(
            y_toy, y_pred_list, conformity_scores
        )
