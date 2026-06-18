import numpy as np
import torch
import pytest

from covmetrics.slab_metrics import WSC

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

def make_data(n=100, d=2, seed=0, backend="numpy"):
    """Create random data for WSC tests, optionally in torch."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    cover = rng.integers(0, 2, size=n)
    if backend == "torch":
        X = torch.tensor(X, dtype=torch.float32)
        cover = torch.tensor(cover, dtype=torch.float32)
    return X, cover

def check_inputs(X, cover):
    """Check type consistency and length."""
    if len(X) != len(cover):
        raise ValueError("X and cover must have same length")
    if not ((isinstance(X, np.ndarray) and isinstance(cover, np.ndarray)) or
            (isinstance(X, torch.Tensor) and isinstance(cover, torch.Tensor))):
        raise TypeError("X and cover must be same type: both np.ndarray or both torch.Tensor")

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_output_type_and_range(backend):
    X, cover = make_data(backend=backend)
    check_inputs(X, cover)
    estimator = WSC(delta=0.1)
    val = estimator.evaluate(X, cover, M=100)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_perfect_cover(backend):
    X, cover = make_data(backend=backend)
    if backend == "numpy":
        cover[:] = 1
    else:
        cover[:] = 1.0
    check_inputs(X, cover)
    estimator = WSC(delta=0.2)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 1.0)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_no_cover(backend):
    X, cover = make_data(backend=backend)
    cover[:] = 0
    check_inputs(X, cover)
    estimator = WSC(delta=0.3)
    val = estimator.evaluate(X, cover, M=200)
    assert np.isclose(val, 0.0)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_delta_threshold_effect(backend):
    X, cover = make_data(backend=backend)
    check_inputs(X, cover)
    estimator_small = WSC(delta=0.1)
    estimator_large = WSC(delta=0.5)
    val_small = estimator_small.evaluate(X, cover, M=200)
    val_large = estimator_large.evaluate(X, cover, M=200)
    assert val_large >= val_small + 1e-8

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_reproducibility_with_seed(backend):
    X, cover = make_data(backend=backend)
    check_inputs(X, cover)
    estimator1 = WSC(delta=0.1)
    estimator2 = WSC(delta=0.1)
    val1 = estimator1.evaluate(X, cover, M=200, seed=42)
    val2 = estimator2.evaluate(X, cover, M=200, seed=42)
    assert np.isclose(val1, val2)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_length_mismatch_raises(backend):
    X, cover = make_data(n=10, backend=backend)
    if backend == "numpy":
        cover = cover[:-1]
    else:
        cover = cover[:-1]
    with pytest.raises(ValueError):
        estimator = WSC(delta=0.1)
        estimator.evaluate(X, cover, M=50)
            

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_1d_easy_cases(backend):
    if backend == "numpy":
        X = np.array([[0.], [1.], [2.], [3.]])
        cover_all = np.array([1,1,1,1])
        cover_none = np.array([0,0,0,0])
        cover_half = np.array([1,1,1,0])
    else:
        X = torch.tensor([[0.],[1.],[2.],[3.]], dtype=torch.float32)
        cover_all = torch.tensor([1,1,1,1], dtype=torch.float32)
        cover_none = torch.tensor([0,0,0,0], dtype=torch.float32)
        cover_half = torch.tensor([1,1,1,0], dtype=torch.float32)

    estimator = WSC(delta=0.25)
    val_all = estimator.evaluate(X, cover_all, M=10)
    val_none = estimator.evaluate(X, cover_none, M=10)
    val_half = estimator.evaluate(X, cover_half, M=10)
    assert np.isclose(val_all, 1.0)
    assert np.isclose(val_none, 0.0)
    assert 0.0 <= val_half <= 1.0

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_1d_cases_unified(backend):
    """Unified 1D WSC test cases for numpy and torch."""
    
    # Define data as numpy first
    X_np = np.array([[0.], [1.], [2.], [3.]])
    covers = {
        "all": np.array([1,1,1,1]),
        "none": np.array([0,0,0,0]),
        "half1": np.array([1,1,1,0]),
        "half2": np.array([1,0,1,0]),
        "asym": np.array([1,1,0,0]),
        "strict": np.array([1,0,1,1])
    }
    
    deltas = {
        "all": 0.25,
        "none": 0.25,
        "half1": 0.26,
        "half2": 0.1,
        "half3": 0.26,
        "asym": 0.5,
        "strict": 0.75
    }
    
    expected = {
        "all": 1.0,
        "none": 0.0,
        "half1": 0.5,
        "half2": 0.0,
        "half3": 1/3,
        "asym": 0.0,
        "strict": 2/3
    }
    
    # Convert to torch if needed
    if backend == "torch":
        X = torch.tensor(X_np, dtype=torch.float32)
        covers = {k: torch.tensor(v, dtype=torch.float32) for k,v in covers.items()}
    else:
        X = X_np
    
    for case in covers:
        cover = covers[case]
        delta = deltas[case]
        estimator = WSC(delta=delta)
        val = estimator.evaluate(X, cover, M=10)
        assert np.isclose(val, expected[case]), f"{case} failed for backend={backend}"

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_return_type_is_float(backend):
    X, cover = make_data(backend=backend)
    estimator = WSC(delta=0.1)
    val = estimator.evaluate(X, cover, M=50)
    assert isinstance(val, float)

# ---------- ALPHA checks ----------
def test_delta_valid_and_invalid_numpy():
    cover = np.array([1,0,1,0])
    X = np.array([[0],[1],[2],[3]])
    # Valid alpha
    estimator = WSC(delta=0.1)
    estimator.evaluate(X, cover, M=50)
    for bad_delta in [-0.1, 1.5]:
        with pytest.raises(ValueError):
            estimator.evaluate(X, cover, M=50, delta=bad_delta)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_1d_invalid_inputs_evaluate(backend):
    """Errors should be raised by evaluate() itself."""

    # Valid inputs
    X_np = np.array([[0.], [1.], [2.], [3.]])
    cover_np = np.array([1,0,1,0])

    if backend == "torch":
        X = torch.tensor(X_np, dtype=torch.float32)
        cover = torch.tensor(cover_np, dtype=torch.float32)
    else:
        X = X_np
        cover = cover_np

    estimator = WSC(delta=0.25)

    # 1. Length mismatch
    cover_short = cover[:-1]
    with pytest.raises(ValueError):
        estimator.evaluate(X, cover_short, M=10)

    # 2. Non-binary cover
    if backend == "torch":
        cover_bad = torch.tensor([0,1,2,0], dtype=torch.float32)
    else:
        cover_bad = np.array([0,1,2,0])
    with pytest.raises(ValueError):
        estimator.evaluate(X, cover_bad, M=10)

    # 3. X not 2D
    if backend == "torch":
        X_bad = torch.tensor([0.,1.,2.,3.], dtype=torch.float32)
    else:
        X_bad = np.array([0.,1.,2.,3.])
    with pytest.raises(ValueError):
        estimator.evaluate(X_bad, cover, M=10)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_emptyness(backend):
    X = to_backend([], backend, "float")
    cover = to_backend([], backend, "int")
    estimator_small = WSC(delta=0.1)
    
    # Expect either ValueError or TypeError
    with pytest.raises((ValueError, TypeError)):
        estimator_small.evaluate(X, cover, M=200)
    