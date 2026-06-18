import pytest
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from covmetrics import ERT
from covmetrics.classifiers import BetterCatBoostClassifier, BetterLGBMClassifier, CheapBetterCatBoostClassifier
from covmetrics.losses import *

from unittest.mock import patch

@pytest.mark.parametrize("backend", ["numpy", "torch", "dataframe"])
def test_ert_initialization(backend):
    # Préparer les données selon le backend
    n_samples, n_features = 100, 5
    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
    elif backend == "torch":
        X = torch.rand(n_samples, n_features)
    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])

    # Initialisation avec un vrai modèle scikit-learn
    ert = ERT(LogisticRegression, max_iter=100)
    assert isinstance(ert.model, LogisticRegression)
    assert ert.fitted is False
    assert ert.added_losses is None


@pytest.mark.parametrize("backend", ["numpy", "torch", "dataframe"])
def test_ert_evaluate_all_start(backend):
    n_samples, n_features = 100, 5
    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
        cover = np.random.randint(0, 2, size=n_samples)
    elif backend == "torch":
        X = torch.rand(n_samples, n_features)
        cover = torch.randint(0, 2, size=(n_samples,))
    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])
        cover = pd.Series(np.random.randint(0, 2, size=n_samples))

    ert = ERT(LogisticRegression, max_iter=100)

    # Mocker les fonctions pour tester uniquement le flux
    with patch("covmetrics.ERT.check_tabular") as mock_check_tabular, \
        patch("covmetrics.ERT.check_cover") as mock_check_cover, \
        patch("covmetrics.ERT.check_consistency") as mock_check_consistency, \
        patch("covmetrics.ERT.check_alpha_tab_ok") as mock_check_alpha, \
        patch.object(ERT, "make_losses") as mock_make_losses:

        results = ert.evaluate_multiple_losses(X, cover, alpha=0.8, n_splits=2)

        # Vérifier que les check ont été appelés
        mock_check_tabular.assert_called()
        mock_check_cover.assert_called()
        mock_check_consistency.assert_called()
        mock_check_alpha.assert_called()
        mock_make_losses.assert_called()

        # Vérifier que les clés ERT sont présentes
        for loss_name in ert.tab_losses:
            key = "ERT_" + loss_name.__name__
            assert key in results


def dummy_loss(pred, y):
    return (pred - y)**2

def alt_loss(pred, y):
    return (pred - y)**2

@pytest.mark.parametrize("backend", ["numpy", "torch", "dataframe"])
@pytest.mark.parametrize("loss_func", [dummy_loss, alt_loss])
def test_evaluate_basic(backend, loss_func):
    n_samples, n_features = 100, 5
    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
        cover = np.random.randint(0, 2, size=n_samples)
    elif backend == "torch":
        X = torch.rand(n_samples, n_features)
        cover = torch.randint(0, 2, size=(n_samples,))
    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])
        cover = pd.Series(np.random.randint(0, 2, size=n_samples))

    ert = ERT(LogisticRegression, max_iter=100)
    ert.fit(X, cover)

    # Mocker les checks et prédictions
    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):

        results = ert.evaluate(X, cover, alpha=0.8, n_splits=0, loss=loss_func)
        assert isinstance(results, float)

@pytest.mark.parametrize("backend", ["numpy", "torch", "dataframe"])
@pytest.mark.parametrize("loss_func", [dummy_loss, alt_loss])
def test_evaluate_with_cv(backend, loss_func):
    n_samples, n_features = 100, 5
    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
        cover = np.random.randint(0, 2, size=n_samples)
    elif backend == "torch":
        X = torch.rand(n_samples, n_features)
        cover = torch.randint(0, 2, size=(n_samples,))
    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])
        cover = pd.Series(np.random.randint(0, 2, size=n_samples))

    ert = ERT(LogisticRegression, max_iter=100)
    ert.fit(X, cover)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):

        # Test n_splits=3
        results = ert.evaluate(X, cover, alpha=0.7, n_splits=3, loss=loss_func)
        assert isinstance(results, float)  # avec cross-validation, retourne une moyenne

def test_evaluate_exception_if_not_fitted():
    X = np.random.rand(100, 5)
    cover = np.random.randint(0, 2, size=100)
    ert = ERT(LogisticRegression, max_iter=100)
    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"):
        with pytest.raises(Exception):
            ert.evaluate(X, cover, alpha=0.8, n_splits=0)

def test_evaluate_with_torch_tensor():
    X = torch.rand(100, 4)
    cover = torch.randint(0, 2, size=(100,))
    ert = ERT(model_cls=LogisticRegression)
    ert.fit(X, cover)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):
        results = ert.evaluate(X, cover, alpha=0.5, n_splits=0, loss=dummy_loss)
        assert isinstance(results, float)

def test_evaluate_with_dataframe_and_alt_loss():
    X = pd.DataFrame(np.random.rand(100, 3))
    cover = pd.Series(np.random.randint(0, 2, size=100))
    ert = ERT(model_cls=LogisticRegression)
    ert.fit(X, cover)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):
        results = ert.evaluate(X, cover, alpha=0.9, n_splits=0, loss=alt_loss)
        assert isinstance(results, float)


@pytest.mark.parametrize("backend", ["numpy", "torch", "dataframe"])
def test_evaluate_all_basic(backend):
    n_samples, n_features = 100, 5
    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
        cover = np.random.randint(0, 2, size=n_samples)
    elif backend == "torch":
        X = torch.rand(n_samples, n_features)
        cover = torch.randint(0, 2, size=(n_samples,))
    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])
        cover = pd.Series(np.random.randint(0, 2, size=n_samples))

    ert = ERT(LogisticRegression, max_iter=100)
    ert.fit(X, cover)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "make_losses"), \
         patch.object(ERT, "init_model"):

        results = ert.evaluate_multiple_losses(X, cover, alpha=0.8, n_splits=0)
        assert isinstance(results, dict)
        for key, value in results.items():
            assert isinstance(value, float)


@pytest.mark.parametrize("backend", ["numpy", "torch", "dataframe"])
def test_evaluate_all_with_cv(backend):
    n_samples, n_features = 100, 5
    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
        cover = np.random.randint(0, 2, size=n_samples)
    elif backend == "torch":
        X = torch.rand(n_samples, n_features)
        cover = torch.randint(0, 2, size=(n_samples,))
    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"f{i}" for i in range(n_features)])
        cover = pd.Series(np.random.randint(0, 2, size=n_samples))

    ert = ERT(LogisticRegression, max_iter=100)
    ert.fit(X, cover)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "make_losses"), \
         patch.object(ERT, "init_model"):

        results = ert.evaluate_multiple_losses(X, cover, alpha=0.7, n_splits=3)
        assert isinstance(results, dict)
        for key, value in results.items():
            assert isinstance(value, float)


def test_evaluate_all_exception_if_not_fitted():
    X = np.random.rand(10, 5)
    cover = np.random.randint(0, 2, size=10)
    ert = ERT(LogisticRegression, max_iter=100)
    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "make_losses"):
        with pytest.raises(Exception):
            ert.evaluate_multiple_losses(X, cover, alpha=0.8, n_splits=0)





# def test_add_loss_basic():
#     ert = ERT(model_cls=LogisticRegression)
    
#     def dummy_loss(pred, y):
#         return (pred - y) ** 2

#     # Initially added_losses is None
#     assert ert.added_losses is None
    
#     ert.add_loss(dummy_loss)
#     assert isinstance(ert.added_losses, list)
#     assert dummy_loss in ert.added_losses

#     # Add another loss
#     def alt_loss(pred, y):
#         pred = np.asarray(pred)
#         y = np.asarray(y)
#         return np.abs(pred - y)
    
#     ert.add_loss(alt_loss)
#     assert alt_loss in ert.added_losses
#     assert len(ert.added_losses) == 2


# def test_add_loss_invalid_loss():
#     ert = ERT(model_cls=LogisticRegression)

#     # Passing a non-callable should raise an error
#     with pytest.raises(Exception):
#         ert.add_loss("not_a_function")


def test_make_losses_basic():
    ert = ERT(model_cls=LogisticRegression)

    def dummy_loss(pred, y):
        return (pred - y) ** 2

    ert.add_loss(dummy_loss)
    ert.make_losses()
    
    # Check that default losses are included
    loss_names = [loss.__name__ if hasattr(loss, "__name__") else loss.__class__.__name__ for loss in ert.tab_losses]
    assert "brier_score" in loss_names
    assert "logloss" in loss_names
    
    # Check that added loss is included
    assert "dummy_loss" in loss_names


def test_make_losses_without_added_losses():
    ert = ERT(model_cls=LogisticRegression)
    ert.make_losses()

    # added_losses is None, only default losses should be present
    loss_names = [loss.__name__ if hasattr(loss, "__name__") else loss.__class__.__name__ for loss in ert.tab_losses]
    assert "brier_score" in loss_names
    assert "logloss" in loss_names
    assert len(loss_names) == 9  # brier_score, logloss, make_L1_miscoverage and their over and over versions


def test_evaluate_all_with_explicit_losses():
    n_samples, n_features = 100, 4
    X = np.random.rand(n_samples, n_features)
    cover = np.random.randint(0, 2, size=n_samples)

    def loss1(pred, y):
        return (pred - y) ** 2

    def loss2(pred, y):
        pred = np.asarray(pred)
        y = np.asarray(y)
        return np.abs(pred - y)

    ert = ERT(model_cls=LogisticRegression)
    ert.fit(X, cover)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):

        results = ert.evaluate_multiple_losses(X, cover, alpha=0.7, all_losses=[loss1, loss2])

        assert isinstance(results, dict)
        assert "ERT_loss1" in results
        assert "ERT_loss2" in results
        for value in results.values():
            assert isinstance(value, float)

def test_evaluate_using_adaptive_coverage_policy():
    n_samples, n_features = 100, 4
    X = np.random.rand(n_samples, n_features)
    cover = np.random.randint(0, 2, size=n_samples)
    alpha = np.ones(n_samples)*0.7

    ert = ERT(model_cls=LogisticRegression)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):

        results1 = ert.evaluate(X, cover, alpha=alpha)
        results2 = ert.evaluate_multiple_losses(X, cover, alpha=alpha)

        assert isinstance(results1, float)
        assert isinstance(results2, dict)
        for value in results2.values():
            assert isinstance(value, float)


@pytest.mark.parametrize(
    "backend, model",
    [
        ("numpy", None),
        ("numpy", BetterCatBoostClassifier),
        ("numpy", BetterLGBMClassifier),
        ("numpy", CheapBetterCatBoostClassifier),
        ("torch", None),
        ("torch", BetterCatBoostClassifier),
        ("torch", BetterLGBMClassifier),
        ("torch", CheapBetterCatBoostClassifier),
        ("dataframe", None),
        ("dataframe", BetterCatBoostClassifier),
        ("dataframe", BetterLGBMClassifier),
        ("dataframe", CheapBetterCatBoostClassifier),
    ]
)
def test_default_settings(backend, model):
    n_samples, n_features = 100, 4

    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
        cover = np.random.randint(0, 2, size=n_samples)

    elif backend == "torch":
        X = torch.rand(n_samples, n_features, dtype=torch.float32)
        cover = torch.randint(0, 2, size=(n_samples,), dtype=torch.float32)

    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features))
        cover = pd.Series(np.random.randint(0, 2, size=n_samples))

    ert = ERT() if model is None else ERT(model)

    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):

        results1 = ert.evaluate(X, cover, alpha=0.7)
        results2 = ert.evaluate_multiple_losses(X, cover, alpha=0.7)

        assert isinstance(results1, float)
        assert isinstance(results2, dict)
        for value in results2.values():
            assert isinstance(value, float)


losses = [
    "brier_score",
    "logloss",
    "L1_miscoverage",
    "brier_score_under",
    "logloss_under",
    "L1_miscoverage_under",
    "brier_score_over",
    "logloss_over",
    "L1_miscoverage_over",
]

backends = ["numpy", "torch", "dataframe"]

@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("loss_name", losses)
def test_each_loss_separately(backend, loss_name):
    n_samples, n_features = 100, 4

    # Préparer les données selon le backend
    if backend == "numpy":
        X = np.random.rand(n_samples, n_features)
        cover = np.random.randint(0, 2, size=n_samples)
    elif backend == "torch":
        X = torch.rand(n_samples, n_features, dtype=torch.float32)
        cover = torch.randint(0, 2, size=(n_samples,), dtype=torch.float32)
    elif backend == "dataframe":
        X = pd.DataFrame(np.random.rand(n_samples, n_features))
        cover = pd.Series(np.random.randint(0, 2, size=n_samples))

    ert = ERT(model_cls=LogisticRegression)

    # Patch pour isoler le test
    with patch("covmetrics.ERT.check_tabular"), \
         patch("covmetrics.ERT.check_cover"), \
         patch("covmetrics.ERT.check_consistency"), \
         patch("covmetrics.ERT.check_alpha_tab_ok"), \
         patch.object(ERT, "init_model"):

        # Sélectionner la loss unique à tester
        tab_losses_to_evaluate = [globals()[loss_name]]

        results = ert.evaluate_multiple_losses(
            X, cover, alpha=0.7, all_losses=tab_losses_to_evaluate
        )

        # Vérifier le type des résultats
        assert isinstance(results, dict)
        assert all(isinstance(value, float) for value in results.values())