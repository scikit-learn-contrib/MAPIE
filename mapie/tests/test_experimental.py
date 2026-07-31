import numpy as np
import pytest
import torch
from numpy.typing import ArrayLike, NDArray

from mapie.experimental import (
    BaseFitRegressionScore,
    MultivariateResidualNormalisedScore,
)
from mapie.experimental.conformity_scores.bounds.utils import (
    RobustCovarianceHead,
    Trainer,
)

X_toy = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y_toy = np.array([5, 7, 9, 11, 13, 15])
y_pred_list = np.array([4, 7, 10, 12, 13, 12])


class MinimalFitRegressionScore(BaseFitRegressionScore):
    def __init__(self) -> None:
        super().__init__(sym=True, consistency_check=False)

    def get_signed_conformity_scores(
        self, y: ArrayLike, y_pred: ArrayLike, **kwargs
    ) -> NDArray:
        return np.subtract(y, y_pred)

    def get_estimation_distribution(
        self, y_pred: ArrayLike, conformity_scores: ArrayLike, **kwargs
    ) -> NDArray:
        return np.add(y_pred, conformity_scores)

    def fit(self, X: NDArray, y: NDArray, **kwargs) -> "MinimalFitRegressionScore":
        super().fit(X, y, **kwargs)
        return self


def test_base_fit_regression_score_fit_sets_is_fitted() -> None:
    score = MinimalFitRegressionScore()
    assert getattr(score, "is_fitted", None) is False
    out = score.fit(X_toy, y_toy)
    assert score.is_fitted is True
    assert out is score
    signed = score.get_signed_conformity_scores(y_toy, y_pred_list)
    np.testing.assert_allclose(signed, y_toy - y_pred_list)
    y_obs = score.get_estimation_distribution(y_pred_list, signed)
    np.testing.assert_allclose(y_obs, y_toy)


class DummyTrainer:
    """Mock estimator simulating Trainer (without get_standardized_score)."""

    def __init__(self, input_dim=1, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_fitted = False

    def fit(self, X, y, y_pred=None, **kwargs):
        self.is_fitted = True
        return self

    def predict(self, X):
        return np.zeros((len(X), self.output_dim))

    def get_distribution(self, X):
        return self.predict(X), self.get_covariance_matrix(X)

    def get_covariance_matrix(self, X):
        n_samples = X.shape[0]
        return np.array([np.eye(self.output_dim) for _ in range(n_samples)])


class DummyCenterModel:
    def __init__(self, input_dim=1, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x):
        n = len(x)
        return np.ones((n, self.output_dim))


class DummyCenterModelTorch:
    def __init__(self, input_dim=1, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, x):
        n = len(x)
        return torch.ones((n, self.output_dim))


INPUT_DIM = 3
OUTPUT_DIM = 2
N_SAMPLES = 100


@pytest.fixture
def mock_data():
    np.random.seed(42)
    X = np.random.randn(N_SAMPLES, INPUT_DIM)
    y = np.random.randn(N_SAMPLES, OUTPUT_DIM)
    y_pred = y + np.random.normal(0, 0.1, size=y.shape)
    return X, y, y_pred


def test_multivariate_initialization():
    score = MultivariateResidualNormalisedScore()
    assert score.prefit is False
    assert score.covariance_estimator_ is None

    trainer = DummyTrainer()
    score_with_trainer = MultivariateResidualNormalisedScore(
        covariance_estimator=trainer, prefit=True
    )
    assert score_with_trainer.prefit is True
    assert score_with_trainer.covariance_estimator_ == trainer


def test_multivariate_univariate_target_raises():
    X = np.random.randn(10, 2)
    y_univariate = np.random.randn(10)

    score = MultivariateResidualNormalisedScore()
    with pytest.raises(AssertionError, match="multivariate targets"):
        score.fit(X, y_univariate)


def test_multivariate_check_estimator_invalid():
    class BadEstimator:
        def fit(self, X, y):
            pass

    score = MultivariateResidualNormalisedScore(covariance_estimator=BadEstimator())
    with pytest.raises(ValueError, match="Invalid estimator"):
        score._check_estimator(score.covariance_estimator_)
    est = BadEstimator()
    est.fit(np.random.randn(5, 2), np.random.randn(5, 2))


def test_multivariate_get_signed_conformity_scores_without_y_pred(mock_data):
    X, y, _ = mock_data
    score_calculator = MultivariateResidualNormalisedScore()
    score_calculator.fit(X, y, num_epochs=1)
    scores = score_calculator.get_signed_conformity_scores(y=y, X=X)
    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1
    assert score_calculator.covariance_estimator_.is_fitted is True


def test_multivariate_get_signed_conformity_scores_with_y_pred(mock_data):
    X, y, y_pred = mock_data
    score_calculator = MultivariateResidualNormalisedScore()
    score_calculator.fit(X, y, num_epochs=1)
    scores = score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)
    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0


def test_multivariate_complex_trainer_get_signed_conformity_scores_with_y_pred(
    mock_data,
):
    X, y, y_pred = mock_data
    list_dic_params = [
        {"mode": "low_rank"},
        {"center_model": DummyCenterModel(INPUT_DIM, OUTPUT_DIM)},
        {"center_model": DummyCenterModelTorch(INPUT_DIM, OUTPUT_DIM)},
    ]
    list_fit_params = [
        {"num_epochs": 1, "val_size": 0.0},
        {"num_epochs": 1, "verbose": -1},
        {"num_epochs": 1, "verbose": 1},
        {"num_epochs": 1, "verbose": 2},
        {"num_epochs": 1, "X_val": X, "y_val": y},
    ]
    for dic_params in list_dic_params:
        score_calculator = MultivariateResidualNormalisedScore(**dic_params)
        score_calculator.fit(X, y, num_epochs=1)
        scores = score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)
        assert isinstance(scores, np.ndarray)
        assert len(scores) > 0

    for dic_params in list_fit_params:
        score_calculator = MultivariateResidualNormalisedScore()
        score_calculator.fit(X, y, **dic_params)
        scores = score_calculator.get_signed_conformity_scores(y=y, X=X)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
        assert score_calculator.covariance_estimator_.is_fitted is True


def test_multivariate_get_distribution(mock_data):
    X, y, y_pred = mock_data
    score_calculator = MultivariateResidualNormalisedScore()
    score_calculator.fit(X, y, num_epochs=1)
    scores = score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)
    with pytest.raises(NotImplementedError, match="not implemented"):
        score_calculator.get_estimation_distribution(y_pred, scores, X=X)


def test_multivariate_complex_trainer_get_signed_conformity_scores_without_y_pred(
    mock_data,
):
    X, y, _ = mock_data
    list_init_params = [
        {"mode": "low_rank"},
        {"center_model": DummyCenterModel(INPUT_DIM, OUTPUT_DIM)},
        {"center_model": DummyCenterModelTorch(INPUT_DIM, OUTPUT_DIM)},
    ]
    list_fit_params = [
        {"num_epochs": 1, "val_size": 0.0},
        {"num_epochs": 1, "verbose": -1},
        {"num_epochs": 1, "verbose": 1},
        {"num_epochs": 1, "verbose": 2},
    ]
    for dic_params in list_init_params:
        score_calculator = MultivariateResidualNormalisedScore(**dic_params)
        score_calculator.fit(X, y, num_epochs=1)
        scores = score_calculator.get_signed_conformity_scores(y=y, X=X)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
        assert score_calculator.covariance_estimator_.is_fitted is True

    for dic_params in list_fit_params:
        score_calculator = MultivariateResidualNormalisedScore()
        score_calculator.fit(X, y, **dic_params)
        scores = score_calculator.get_signed_conformity_scores(y=y, X=X)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
        assert score_calculator.covariance_estimator_.is_fitted is True


def test_multivariate_dummy_get_signed_conformity_scores_without_y_pred(mock_data):
    X, y, _ = mock_data
    trainer = DummyTrainer(input_dim=3, output_dim=2)
    score_calculator = MultivariateResidualNormalisedScore(
        covariance_estimator=trainer,
        split_size=0.2,
        random_state=42,
    )
    score_calculator.fit(X, y)
    scores = score_calculator.get_signed_conformity_scores(y=y, X=X)
    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1
    assert score_calculator.covariance_estimator_.is_fitted is True

    y_pred, sigma = trainer.get_distribution(X)
    np.testing.assert_allclose(sigma, trainer.get_covariance_matrix(X))


def test_multivariate_dummy_get_signed_conformity_scores_with_y_pred(mock_data):
    X, y, y_pred = mock_data
    trainer = DummyTrainer(input_dim=3, output_dim=2)
    score_calculator = MultivariateResidualNormalisedScore(
        covariance_estimator=trainer, prefit=False, split_size=0.3
    )
    score_calculator.fit(X, y)
    scores = score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)
    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0

    with pytest.raises(NotImplementedError, match="not implemented"):
        score_calculator.get_estimation_distribution(y_pred, scores, X=X)


def test_multivariate_non_existing_rank_method(mock_data):
    X, y, _ = mock_data
    score_calculator = MultivariateResidualNormalisedScore(mode="jqkncnksjc")
    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=None, X=X)


def test_multivariate_usage_with_y_pred_to_fit_TO_BE_MODIFIED_FOR_FUTURE_VERSION(
    mock_data,
):
    X, y, y_pred = mock_data
    score_calculator = MultivariateResidualNormalisedScore(mode="jqkncnksjc")
    with pytest.raises(Exception):
        score_calculator.fit(X, y, y_pred)


def test_multivariate_model_not_trained(mock_data):
    X, y, y_pred = mock_data
    score_calculator = MultivariateResidualNormalisedScore(prefit=True)
    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)


def test_multivariate_no_X_given(mock_data):
    X, y, y_pred = mock_data
    score_calculator = MultivariateResidualNormalisedScore()
    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred)


def test_multivariate_nan_in_y_pred(mock_data):
    X, y, y_pred = mock_data
    y_pred[0, 0] = np.nan
    score_calculator = MultivariateResidualNormalisedScore()
    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)


def test_multivariate_get_standardized_score_math():
    y = np.array([[2.0, 3.0], [0.0, 0.0]])
    y_pred = np.array([[0.0, 0.0], [0.0, 0.0]])
    Sigma_pred = np.array([np.eye(2), np.eye(2)])
    scores = MultivariateResidualNormalisedScore._get_standardized_score(
        y, y_pred, Sigma_pred
    )
    expected_score_0 = np.sqrt(13)
    expected_score_1 = 0.0
    np.testing.assert_allclose(scores, [expected_score_0, expected_score_1], rtol=1e-5)


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
