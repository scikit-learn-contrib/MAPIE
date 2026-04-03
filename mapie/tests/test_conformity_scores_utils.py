from typing import List

import numpy as np
import pytest
import torch

from mapie.conformity_scores import (
    AbsoluteConformityScore,
    BaseRegressionScore,
    GammaConformityScore,
    LACConformityScore,
    BaseClassificationScore,
    TopKConformityScore,
)
from mapie.conformity_scores.bounds.utils import RobustCovarianceHead, Trainer
from mapie.conformity_scores.sets.utils import get_true_label_position
from numpy.typing import NDArray

from mapie.conformity_scores.utils import check_and_select_conformity_score


class TestCheckAndSelectConformityScore:
    @pytest.mark.parametrize(
        "score, score_type, expected_class",
        [
            (AbsoluteConformityScore(), BaseRegressionScore, AbsoluteConformityScore),
            ("gamma", BaseRegressionScore, GammaConformityScore),
            (LACConformityScore(), BaseClassificationScore, LACConformityScore),
            ("top_k", BaseClassificationScore, TopKConformityScore),
        ],
    )
    def test_with_valid_inputs(self, score, score_type, expected_class):
        result = check_and_select_conformity_score(score, score_type)
        assert isinstance(result, expected_class)

    @pytest.mark.parametrize(
        "score_type", [BaseRegressionScore, BaseClassificationScore]
    )
    def test_with_invalid_input(self, score_type):
        with pytest.raises(ValueError):
            check_and_select_conformity_score("I'm not a valid input :(", score_type)


Y_TRUE_PROBA_PLACE = [
    [
        np.array([2, 0]),
        np.array([[0.1, 0.3, 0.6], [0.2, 0.7, 0.1]]),
        np.array([[0], [1]]),
    ],
    [
        np.array([1, 0]),
        np.array([[0.7, 0.12, 0.18], [0.5, 0.24, 0.26]]),
        np.array([[2], [0]]),
    ],
]


def test_shape_get_true_label_position() -> None:
    """
    Check the shape returned by the function
    """
    y_pred_proba = np.random.rand(5, 3)
    y = np.random.randint(0, 3, size=(5, 1))
    position = get_true_label_position(y_pred_proba, y)
    assert position.shape == y.shape


@pytest.mark.parametrize("y_true_proba_place", Y_TRUE_PROBA_PLACE)
def test_get_true_label_position(y_true_proba_place: List[NDArray]) -> None:
    """
    Check that the returned true label position the good.
    """
    y_true = y_true_proba_place[0]
    y_pred_proba = y_true_proba_place[1]
    place = y_true_proba_place[2]

    found_place = get_true_label_position(y_pred_proba, y_true)

    assert (found_place == place).all()


class TestRobustCovarianceHead:
    def test_full_cholesky_large_y_dim_prints_recommendation(self, capsys):
        RobustCovarianceHead(input_dim=4, y_dim=11, mode="full_cholesky")
        out, _ = capsys.readouterr()
        assert "low_rank" in out

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="full_cholesky.*low_rank"):
            RobustCovarianceHead(input_dim=2, y_dim=2, mode="invalid")


class TestTrainer:
    def test_fit_raises_when_y_pred_provided(self):
        trainer = Trainer(input_dim=2, output_dim=2)
        X = np.random.randn(20, 2)
        y = np.random.randn(20, 2)
        y_pred = np.random.randn(20, 2)
        with pytest.raises(Exception, match="y_pred yet"):
            trainer.fit(X, y, y_pred=y_pred)

    def test_full_cholesky_fit_predict_get_standardized_score(self):
        np.random.seed(42)
        torch.manual_seed(42)
        n, input_dim, output_dim = 50, 3, 2
        X = np.random.randn(n, input_dim).astype(np.float32)
        y = np.random.randn(n, output_dim).astype(np.float32)

        trainer = Trainer(
            input_dim=input_dim,
            output_dim=output_dim,
            mode="full_cholesky",
            hidden_dim=16,
            num_layers=1,
        )
        trainer.fit(X, y, num_epochs=2, val_size=0.0, batch_size=16)

        y_pred = trainer.predict(X)
        assert y_pred.shape == (n, output_dim)

        scores = trainer.get_standardized_score(X, y)
        assert scores.shape == (n,)
        assert np.all(scores >= 0)

    def test_get_distribution_and_covariance_matrix(self):
        np.random.seed(42)
        torch.manual_seed(42)
        n, input_dim, output_dim = 30, 2, 2
        X = np.random.randn(n, input_dim).astype(np.float32)
        y = np.random.randn(n, output_dim).astype(np.float32)

        trainer = Trainer(
            input_dim=input_dim,
            output_dim=output_dim,
            mode="full_cholesky",
            hidden_dim=8,
            num_layers=1,
        )
        trainer.fit(X, y, num_epochs=1, val_size=0.0)

        mu, Sigma = trainer.get_distribution(X)
        assert mu.shape == (n, output_dim)
        assert Sigma.shape == (n, output_dim, output_dim)

        Sigma_only = trainer.get_covariance_matrix(X)
        np.testing.assert_array_almost_equal(Sigma_only, Sigma)

    def test_low_rank_mode_fit_and_standardized_score(self):
        np.random.seed(42)
        torch.manual_seed(42)
        n, input_dim, output_dim = 40, 3, 2
        X = np.random.randn(n, input_dim).astype(np.float32)
        y = np.random.randn(n, output_dim).astype(np.float32)

        trainer = Trainer(
            input_dim=input_dim,
            output_dim=output_dim,
            mode="low_rank",
            hidden_dim=8,
            num_layers=1,
        )
        trainer.fit(X, y, num_epochs=1, val_size=0.0)

        scores = trainer.get_standardized_score(X, y)
        assert scores.shape == (n,)
