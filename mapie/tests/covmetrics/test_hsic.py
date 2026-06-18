import numpy as np
import torch
import pytest

from covmetrics.dependence_metrics import HSIC

def to_backend(array, backend, dtype="float"):
    """Helper to create numpy/torch/list arrays with correct types."""
    if backend == "numpy":
        return np.array(array, dtype=float if dtype == "float" else int)
    elif backend == "torch":
        return torch.tensor(array, dtype=torch.float32 if dtype == "float" else torch.int64)
    else:
        raise ValueError("Unsupported backend")


# -------------------- CORE TESTS --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_independent_inputs_small_value(backend):
    """If cover and sizes are independent, HSIC should be close to 0."""
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 2, size=50)   # binary cover
    sizes = rng.normal(size=50)           # random noise, independent

    cover_b = to_backend(cover, backend)
    sizes_b = to_backend(sizes, backend)

    estimator = HSIC(sigma_x=1, sigma_y=1)
    val = estimator.evaluate(sizes_b, cover_b)
    assert isinstance(val, float)
    assert val >= 0.0
    assert val < 1.0   # should be small-ish for independent


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_dependent_inputs_large_value(backend):
    """If cover is correlated with sizes, HSIC should be larger."""
    cover = [0]*25 + [1]*25
    sizes = [1]*25 + [10]*25   # strongly dependent

    cover_b = to_backend(cover, backend, dtype="float")
    sizes_b = to_backend(sizes, backend, dtype="float")

    estimator = HSIC(sigma_x=1, sigma_y=1)
    val = estimator.evaluate(sizes_b, cover_b)
    assert isinstance(val, float)
    assert val > 0.1   # must detect dependence


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_return_type_is_float(backend):
    cover = to_backend([0, 1, 0, 1], backend)
    sizes = to_backend([1.0, 2.0, 3.0, 4.0], backend)
    estimator = HSIC()
    val = estimator.evaluate(sizes, cover)
    assert isinstance(val, float)


# -------------------- BACKEND CONSISTENCY --------------------

def test_numpy_and_torch_equivalence():
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 2, size=30)
    sizes = rng.normal(size=30)

    estimator = HSIC()

    val_numpy = estimator.evaluate(np.array(sizes, dtype=float),
                                   np.array(cover, dtype=float))

    val_torch = estimator.evaluate(torch.tensor(sizes, dtype=torch.float32),
                                   torch.tensor(cover, dtype=torch.float32))

    # should be very close
    assert np.isclose(val_numpy, val_torch, atol=1e-6)


# -------------------- ERROR HANDLING --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_length_mismatch_raises(backend):
    cover = to_backend([0, 1, 1], backend)
    sizes = to_backend([1.0, 2.0], backend)  # shorter
    estimator = HSIC()
    with pytest.raises((ValueError, IndexError)):
        estimator.evaluate(sizes, cover)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_empty_inputs_raise(backend):
    cover = to_backend([], backend)
    sizes = to_backend([], backend)
    estimator = HSIC()
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(sizes, cover)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_invalid_cover_values_raise(backend):
    """Cover must be 0/1 only."""
    estimator = HSIC()
    sizes = to_backend([1.0, 2.0, 3.0, 4.0], backend)

    cover_bad = to_backend([0, 2, 1, -1], backend)
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(sizes, cover_bad)

    cover_float_bad = to_backend([0.0, 0.5, 1.0, 0.2], backend)
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(sizes, cover_float_bad)


# -------------------- RANDOMIZED TEST --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_random_binary_cover_runs(backend):
    """Just checks that HSIC runs and returns nonnegative for random data."""
    rng = np.random.default_rng(123)
    cover = rng.integers(0, 2, size=100)
    sizes = rng.normal(size=100)

    if backend == "torch":
        cover = torch.tensor(cover, dtype=torch.float32)
        sizes = torch.tensor(sizes, dtype=torch.float32)

    estimator = HSIC(sigma_x=1, sigma_y=1)
    val = estimator.evaluate(sizes, cover)
    assert isinstance(val, float)
    assert val >= 0.0
