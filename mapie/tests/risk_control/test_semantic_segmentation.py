import numpy as np
import pytest

from mapie.risk_control.semantic_segmentation import SemanticSegmentationController


@pytest.fixture
def dummy_ssc():
    return SemanticSegmentationController(
        predict_function=lambda X: np.array([[[0.6, 0.4]]]),
        risk="precision",
        target_level=0.8,
        confidence_level=0.9,
    )


def test_transform_pred_proba_list_input(dummy_ssc):
    y_pred = [[0.2, 0.8]]
    out = dummy_ssc._transform_pred_proba(y_pred, ravel=True)
    assert isinstance(out, np.ndarray)


def test_transform_pred_proba_ravel(dummy_ssc):
    y_pred = np.array([[-5.0, 0.0, 5.0]])
    out = dummy_ssc._transform_pred_proba(y_pred, ravel=True)
    assert out.shape == (1, 3, 1)


def test_transform_pred_proba_no_ravel(dummy_ssc):
    y_pred = np.array([[-5.0, 0.0, 5.0]])
    out = dummy_ssc._transform_pred_proba(y_pred, ravel=False)
    assert out.shape == (1, 3)


def test_predict(dummy_ssc):
    dummy_ssc._is_fitted = True
    dummy_ssc.best_predict_param = np.array([0.5])
    dummy_ssc._alpha = np.array([0.1])
    X = np.zeros((1, 2, 2))
    y_pred = dummy_ssc.predict(X)
    assert y_pred.shape == (1, 1, 1, 2)
