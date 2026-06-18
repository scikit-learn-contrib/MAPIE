import numpy as np
import torch
import pytest

from covmetrics.group_metrics import FSC

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
    cover = to_backend([1, 1, 1, 1], backend, "float")
    groups = to_backend([0, 0, 0, 0], backend, "int")
    estimator = FSC()
    val = estimator.evaluate(groups, cover)
    expected = 1.0
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_single_group_no_cover(backend):
    cover = to_backend([0, 0, 0, 0], backend, "float")
    groups = to_backend([0, 0, 0, 0], backend, "int")
    estimator = FSC()
    val = estimator.evaluate(groups, cover)
    expected = 0.0
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_two_groups_mixed_cover(backend):
    cover = to_backend([1, 1, 1, 0, 0, 0], backend, "float")
    groups = to_backend([0, 0, 0, 1, 1, 1], backend, "int")
    estimator = FSC()
    val = estimator.evaluate(groups, cover)
    expected = 0.0
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_return_type_is_float(backend):
    cover = to_backend([1, 0, 1, 1], backend, "float")
    groups = to_backend([0, 0, 1, 1], backend, "int")
    estimator = FSC()
    val = estimator.evaluate(groups, cover)
    assert isinstance(val, float)


# -------------------- ERROR HANDLING --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_length_mismatch_raises(backend):
    cover = to_backend([1, 0, 1], backend, "float")
    groups = to_backend([0, 0], backend, "int")  # shorter
    estimator = FSC()
    with pytest.raises((ValueError, IndexError)):
        estimator.evaluate(groups, cover)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_non_integer_group_labels_raises(backend):
    cover = to_backend([1, 0, 1, 0], backend, "float")
    estimator = FSC()
    if backend == "numpy":
        groups_str = np.array(["a", "a", "b", "b"])
        groups_float = np.array([0.1, 0.1, 1.5, 1.5])
        with pytest.raises((TypeError, ValueError)):
            estimator.evaluate(groups_str, cover)
        with pytest.raises((TypeError, ValueError)):
            estimator.evaluate(groups_float, cover)
    else:
        groups_float = torch.tensor([0.1, 0.1, 1.5, 1.5], dtype=torch.float32)
        with pytest.raises((TypeError, ValueError)):
            estimator.evaluate(groups_float, cover)


# -------------------- COMPLEX GROUP STRUCTURES --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_multiple_groups_three_classes(backend):
    cover = to_backend([1, 0, 1, 0, 1, 0], backend, "float")
    groups = to_backend([0, 0, 1, 1, 2, 2], backend, "int")
    estimator = FSC()
    val = estimator.evaluate(groups, cover)
    expected = 0.5
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_groups_with_noncontiguous_labels(backend):
    cover = to_backend([1, 1, 0, 0], backend, "float")
    groups = to_backend([10, 10, 42, 42], backend, "int")
    estimator = FSC()
    val = estimator.evaluate(groups, cover)
    expected = min(1.0, 0.0)
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_many_groups_small_sizes(backend):
    cover = to_backend([1, 0, 1, 0], backend, "float")
    groups = to_backend([0, 1, 2, 3], backend, "int")
    estimator = FSC()
    val = estimator.evaluate(groups, cover)
    expected = 0.0
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_emptyness(backend):
    X = to_backend([], backend, "float")
    cover = to_backend([], backend, "int")
    estimator = FSC()
            
    # Expect either ValueError or TypeError
    with pytest.raises((ValueError, TypeError)):
        estimator.evaluate(X, cover)