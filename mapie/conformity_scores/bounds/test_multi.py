import numpy as np
import pytest
import torch

# Importing your class based on your project structure
from mapie.conformity_scores import MultivariateResidualNormalisedScore


class DummyTrainer:
    """A mock estimator to simulate the behavior of Trainer in unit tests. This trainer does not have a get_standardized_score function."""

    def __init__(self, input_dim=1, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fitted = False

    def fit(self, X, y, y_pred=None, **kwargs):
        self.fitted = True
        return self

    def predict(self, X):
        return np.zeros((len(X), self.output_dim))

    def get_distribution(self, X):
        n_samples = X.shape[0]
        # Returns dummy predictions and an identity covariance matrix per sample
        y_pred = np.zeros((n_samples, self.output_dim))
        sigma = np.array([np.eye(self.output_dim) for _ in range(n_samples)])
        return y_pred, sigma

    def get_covariance_matrix(self, X):
        n_samples = X.shape[0]
        # Returns an identity covariance matrix per sample
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
    """Generates random mock data for testing."""
    np.random.seed(42)
    n_samples = N_SAMPLES
    input_dim = INPUT_DIM
    output_dim = OUTPUT_DIM

    X = np.random.randn(n_samples, input_dim)
    y = np.random.randn(n_samples, output_dim)
    y_pred = y + np.random.normal(0, 0.1, size=y.shape)

    return X, y, y_pred


def test_initialization():
    """Test that the class initializes correctly with and without an estimator."""
    # Without an estimator
    score = MultivariateResidualNormalisedScore()
    assert score.prefit is False
    assert score.covariance_estimator_ is None

    # With an estimator
    trainer = DummyTrainer()
    score_with_trainer = MultivariateResidualNormalisedScore(
        covariance_estimator=trainer, prefit=True
    )
    assert score_with_trainer.prefit is True
    assert score_with_trainer.covariance_estimator_ == trainer


def test_check_estimator_invalid():
    """Test that a ValueError is raised if the estimator lacks required methods."""

    class BadEstimator:
        def fit(self, X, y):
            pass

        # Missing get_distribution and get_covariance_matrix

    score = MultivariateResidualNormalisedScore(covariance_estimator=BadEstimator())
    with pytest.raises(ValueError, match="Invalid estimator"):
        score._check_estimator(score.covariance_estimator_)
    est = BadEstimator()
    est.fit(np.random.randn(5, 2), np.random.randn(5, 2))


def test_get_signed_conformity_scores_without_y_pred(mock_data):
    """Test score generation when no initial y_pred is provided."""
    X, y, _ = mock_data

    score_calculator = MultivariateResidualNormalisedScore()
    score_calculator.fit(X, y, num_epochs=1)

    scores = score_calculator.get_signed_conformity_scores(y=y, X=X)

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1  # Scores should be a 1D array (the norms)

    assert score_calculator.covariance_estimator_.fitted is True


def test_get_signed_conformity_scores_with_y_pred(mock_data):
    """Test score generation when y_pred is provided."""
    X, y, y_pred = mock_data

    score_calculator = MultivariateResidualNormalisedScore()
    score_calculator.fit(X, y, num_epochs=1)

    scores = score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)

    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0


def test_complex_trainer_get_signed_conformity_scores_with_y_pred(mock_data):
    """Test score generation when y_pred is provided."""
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
        assert scores.ndim == 1  # Scores should be a 1D array (the norms)

        assert score_calculator.covariance_estimator_.fitted is True


def test_get_distribution(mock_data):
    """Test score generation when y_pred is provided."""
    X, y, y_pred = mock_data

    score_calculator = MultivariateResidualNormalisedScore()
    score_calculator.fit(X, y, num_epochs=1)

    scores = score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)
    estimation_distribution = score_calculator.get_estimation_distribution(
        y_pred, scores, X=X
    )

    assert isinstance(estimation_distribution[0], np.ndarray)
    assert isinstance(estimation_distribution[1], np.ndarray)


def test_complex_trainer_get_signed_conformity_scores_without_y_pred(mock_data):
    """Test score generation when no initial y_pred is provided."""
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
        assert scores.ndim == 1  # Scores should be a 1D array (the norms)

        assert score_calculator.covariance_estimator_.fitted is True

    for dic_params in list_fit_params:
        score_calculator = MultivariateResidualNormalisedScore()
        score_calculator.fit(X, y, **dic_params)
        scores = score_calculator.get_signed_conformity_scores(y=y, X=X)

        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1  # Scores should be a 1D array (the norms)

        assert score_calculator.covariance_estimator_.fitted is True


def test_dummy_get_signed_conformity_scores_without_y_pred(mock_data):
    """Test score generation when no initial y_pred is provided."""
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
    assert scores.ndim == 1  # Scores should be a 1D array (the norms)

    assert score_calculator.covariance_estimator_.fitted is True


def test_dummy_get_signed_conformity_scores_with_y_pred(mock_data):
    """Test score generation when y_pred is provided."""
    X, y, y_pred = mock_data

    trainer = DummyTrainer(input_dim=3, output_dim=2)
    score_calculator = MultivariateResidualNormalisedScore(
        covariance_estimator=trainer, prefit=False, split_size=0.3
    )

    score_calculator.fit(X, y)
    scores = score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)

    assert isinstance(scores, np.ndarray)
    assert len(scores) > 0

    estimation_distribution = score_calculator.get_estimation_distribution(
        y_pred, scores, X=X
    )
    assert isinstance(estimation_distribution[0], np.ndarray)
    assert isinstance(estimation_distribution[1], np.ndarray)


def test_non_existing_rank_method(mock_data):
    """Test score generation when no initial y_pred is provided."""
    X, y, _ = mock_data

    score_calculator = MultivariateResidualNormalisedScore(mode="jqkncnksjc")
    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=None, X=X)


def test_usage_with_y_pred_to_fit_TO_BE_MODIFIED_FOR_FUTURE_VERSION(mock_data):
    """Test score generation when no initial y_pred is provided."""
    X, y, y_pred = mock_data

    score_calculator = MultivariateResidualNormalisedScore(mode="jqkncnksjc")
    with pytest.raises(Exception):
        score_calculator.fit(X, y, y_pred)


def test_model_not_trained(mock_data):
    """Test score generation when y_pred is provided."""
    X, y, y_pred = mock_data

    score_calculator = MultivariateResidualNormalisedScore(prefit=True)

    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)


def test_no_X_given(mock_data):
    """
    Test that the score generation raises an error when X is missing.
    """
    X, y, y_pred = mock_data

    score_calculator = MultivariateResidualNormalisedScore()

    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred)


def test_get_estimation_distribution_without_X(mock_data):
    X, y, y_pred = mock_data

    score_calculator = MultivariateResidualNormalisedScore()
    score_calculator.fit(X, y, num_epochs=1)
    scores = score_calculator.get_signed_conformity_scores(y=y, X=X)

    with pytest.raises(ValueError, match="here `X` is missing"):
        score_calculator.get_estimation_distribution(y_pred, scores)


def test_nan_in_y_pred(mock_data):
    """
    Test that the score generation raises an error when X is missing.
    """
    X, y, y_pred = mock_data
    y_pred[0, 0] = np.nan
    score_calculator = MultivariateResidualNormalisedScore()

    with pytest.raises(Exception):
        score_calculator.get_signed_conformity_scores(y=y, y_pred=y_pred, X=X)


def test_get_standardized_score_math():
    """Test that the Mahalanobis distance calculation is mathematically correct."""
    y = np.array([[2.0, 3.0], [0.0, 0.0]])
    y_pred = np.array([[0.0, 0.0], [0.0, 0.0]])

    # Identity covariance matrices
    Sigma_pred = np.array([np.eye(2), np.eye(2)])

    # For y[0]: residuals = [2, 3]. Identity^(-1/2) = Identity.
    # Standardized residuals = [2, 3]. Euclidean norm = sqrt(2^2 + 3^2) = sqrt(13) = 3.6055
    scores = MultivariateResidualNormalisedScore._get_standardized_score(
        y, y_pred, Sigma_pred
    )

    expected_score_0 = np.sqrt(13)
    expected_score_1 = 0.0

    np.testing.assert_allclose(scores, [expected_score_0, expected_score_1], rtol=1e-5)
