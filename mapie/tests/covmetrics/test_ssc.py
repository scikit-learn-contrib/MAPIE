import numpy as np
import torch
import pytest
from covmetrics.group_metrics import SSC

def to_backend(array, backend, dtype="float"):
    """Helper to create numpy or torch arrays with the right type."""
    if backend == "numpy":
        if dtype == "float":
            return np.array(array, dtype=float)
        return np.array(array, dtype=int)
    elif backend == "torch":
        if dtype == "float":
            return torch.tensor(array, dtype=torch.float32)
        return torch.tensor(array, dtype=torch.int64)
    else:
        raise ValueError("Unsupported backend")


# -------------------- CORE TESTS --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_single_group_perfect_cover(backend):
    y = to_backend([0, 0, 0, 0], backend, "int")
    cover = to_backend([1, 1, 1, 1], backend, "float")
    estimator = SSC()
    val = estimator.evaluate(y, cover, alpha=0.1)
    expected = 0.1  # |1 - (1-alpha)| = alpha
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_multiple_groups_small_unique_y(backend):
    y = to_backend([0, 0, 1, 1], backend, "int")
    cover = to_backend([1, 0, 1, 0], backend, "float")
    estimator = SSC()
    val = estimator.evaluate(y, cover, alpha=0.2)
    expected = np.abs(0.5 - 0.8) / 2 + np.abs(0.5 - 0.8) /2
    assert isinstance(val, float)
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_kmeans_grouping_triggers_for_many_unique_y(backend):
    y = np.linspace(0, 9, 10)  # 10 unique values
    cover = np.array([1,0,1,0,1,0,1,0,1,0], dtype=float)
    if backend == "torch":
        y = torch.tensor(y, dtype=torch.float32)
        cover = torch.tensor(cover, dtype=torch.float32)
    estimator = SSC()
    val = estimator.evaluate(y, cover, alpha=0.1, number_max_groups=5)
    assert isinstance(val, float)  # returns a float
    # Cannot assert exact value because KMeans is approximate


# -------------------- EDGE CASES --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_empty_arrays_raise(backend):
    y = to_backend([], backend, "int")
    cover = to_backend([], backend, "float")
    estimator = SSC()
    with pytest.raises((ValueError, TypeError)):
        estimator.evaluate(y, cover, alpha=0.2)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_groups_with_noncontiguous_labels(backend):
    y = to_backend([10, 10, 42, 42], backend, "int")
    cover = to_backend([1, 1, 0, 0], backend, "float")
    estimator = SSC()
    val = estimator.evaluate(y, cover, alpha=0.2)
    expected = np.abs(1-0.8)/2+np.abs(0-0.8)/2
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_torch_float_y_small_unique(backend):
    if backend == "torch":
        y = torch.tensor([0.1, 0.1, 0.5, 0.5], dtype=torch.float32)
        cover = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    else:
        y = np.array([0.1, 0.1, 0.5, 0.5], dtype=float)
        cover = np.array([1, 0, 1, 0], dtype=float)
    estimator = SSC()
    val = estimator.evaluate(y, cover, alpha=0.3)
    assert isinstance(val, float)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_many_unique_values_trigger_kmeans(backend):
    np.random.seed(42)

    # Create 20 values around 10 (cover=1) and 20 values around -10 (cover=0)
    y_positive = 10 + np.random.randn(20) * 1e-4  # very small noise
    y_negative = -10 + np.random.randn(20) * 1e-4
    y = np.concatenate([y_positive, y_negative])

    cover = np.concatenate([np.ones(20), np.zeros(20)])

    if backend == "torch":
        y = torch.tensor(y, dtype=torch.float32)
        cover = torch.tensor(cover, dtype=torch.float32)

    estimator = SSC()
    val = estimator.evaluate(y, cover, alpha=0.1, number_max_groups=2)

    # Expected behavior: min cover of groups after KMeans should be 0
    assert isinstance(val, float)
    assert val == np.abs(1-0.1)/2 + np.abs(0-0.1)/2

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_alpha_override(backend):
    y = to_backend([0, 0, 0, 0], backend, "int")
    cover = to_backend([1, 1, 1, 1], backend, "float")
    estimator = SSC(alpha=0.1)
    val1 = estimator.evaluate(y, cover)
    val2 = estimator.evaluate(y, cover, alpha=0.5)
    assert not np.isclose(val1, val2)