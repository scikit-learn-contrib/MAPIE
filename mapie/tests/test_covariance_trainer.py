import numpy as np
import pytest
import torch

from mapie.conformity_scores.bounds.covariance_trainer import (
    RobustCovarianceHead,
    Trainer,
)


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
