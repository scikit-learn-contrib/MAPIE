import numpy as np
import torch
import pytest
from scipy import stats

from covmetrics.dependence_metrics import PearsonCorrelation


def to_backend(array, backend, dtype="float"):
    """Helper to create numpy or torch arrays with the right type."""
    if backend == "numpy":
        return np.array(array, dtype=float if dtype == "float" else int)
    elif backend == "torch":
        return torch.tensor(array, dtype=torch.float32 if dtype == "float" else torch.int64)
    else:
        raise ValueError("Unsupported backend")


# -------------------- CORE TESTS --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_perfect_positive_correlation(backend):
    cover = to_backend([0, 0, 1, 1, 1], backend)     # binary
    sizes = to_backend([1, 2, 3, 4, 5], backend)
    estimator = PearsonCorrelation()
    val = estimator.evaluate(sizes, cover)
    # higher sizes tend to match cover=1 → positive correlation
    assert val > 0.5


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_perfect_negative_correlation(backend):
    cover = to_backend([1, 1, 0, 0, 0], backend)     # binary
    sizes = to_backend([1, 2, 3, 4, 5], backend)
    estimator = PearsonCorrelation()
    val = estimator.evaluate(sizes, cover)
    # higher sizes tend to match cover=0 → negative correlation
    assert val < -0.5


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_zero_correlation_like(backend):
    cover = to_backend([0, 1, 0, 1], backend)
    sizes = to_backend([10, 10, 5, 5], backend)  # no variation linked to cover
    estimator = PearsonCorrelation()
    val = estimator.evaluate(sizes, cover)
    assert np.isclose(val, 0.0, atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_return_type_is_float(backend):
    cover = to_backend([0, 1, 0, 1], backend)
    sizes = to_backend([1, 2, 3, 4], backend)
    estimator = PearsonCorrelation()
    val = estimator.evaluate(sizes, cover)
    assert isinstance(val, float)


# -------------------- TORCH / NUMPY CONSISTENCY --------------------

def test_numpy_and_torch_equivalence():
    cover = np.array([0, 0, 1, 1, 1], dtype=float)
    sizes = np.array([5, 4, 3, 2, 1], dtype=float)
    estimator = PearsonCorrelation()
    val_numpy = estimator.evaluate(sizes, cover)

    cover_t = torch.tensor(cover, dtype=torch.float32)
    sizes_t = torch.tensor(sizes, dtype=torch.float32)
    val_torch = estimator.evaluate(sizes_t, cover_t)

    assert np.isclose(val_numpy, val_torch, atol=1e-6)


# -------------------- ERROR HANDLING --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_length_mismatch_raises(backend):
    cover = to_backend([0, 1, 1], backend)
    sizes = to_backend([1, 2], backend)  # shorter
    estimator = PearsonCorrelation()
    with pytest.raises((ValueError, IndexError)):
        estimator.evaluate(sizes, cover)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_empty_inputs_raise(backend):
    cover = to_backend([], backend)
    sizes = to_backend([], backend)
    estimator = PearsonCorrelation()
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(sizes, cover)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_invalid_cover_values_raise(backend):
    estimator = PearsonCorrelation()

    # Non-binary values in cover should fail
    cover_invalid = to_backend([0, 2, 1, -1], backend)
    sizes = to_backend([1, 2, 3, 4], backend)
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(sizes, cover_invalid)

    cover_float_invalid = to_backend([0.0, 0.5, 1.0, 0.2], backend)
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(sizes, cover_float_invalid)

# -------------------- RANDOMIZED TESTS --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_random_binary_cover_matches_scipy(backend):
    rng = np.random.default_rng(0)
    cover = rng.integers(0, 2, size=50)   # binary cover
    sizes = rng.normal(size=50)

    if backend == "torch":
        cover = torch.tensor(cover, dtype=torch.float32)
        sizes = torch.tensor(sizes, dtype=torch.float32)

    estimator = PearsonCorrelation()
    val = estimator.evaluate(sizes, cover)

    expected, _ = stats.pearsonr(
        np.asarray(sizes, dtype=float),
        np.asarray(cover, dtype=float),
    )
    assert np.isclose(val, expected, atol=1e-6)
