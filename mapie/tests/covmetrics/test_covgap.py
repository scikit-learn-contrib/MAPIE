import numpy as np
import torch
import pytest

from covmetrics.group_metrics import CovGap


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
    estimator = CovGap(alpha=0.1)
    val = estimator.evaluate(groups, cover)
    expected = abs(1.0 - (1 - 0.1))  # |1 - 0.9| = 0.1
    assert np.isclose(val, expected)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_weighted_not_boolean_raises(backend):
    cover = to_backend([1, 1, 1, 1], backend, "float")
    groups = to_backend([0, 0, 0, 0], backend, "int")
    estimator = CovGap(alpha=0.1)
    with pytest.raises(ValueError): 
        estimator.evaluate(groups, cover, weighted=2)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_single_group_no_cover(backend):
    cover = to_backend([0, 0, 0, 0], backend, "float")
    groups = to_backend([0, 0, 0, 0], backend, "int")
    estimator = CovGap(alpha=0.2)
    val = estimator.evaluate(groups, cover)
    expected = abs(0.0 - (1 - 0.2))  # |0 - 0.8| = 0.8
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_two_groups_balanced_cover(backend):
    cover = to_backend([1, 1, 1, 0, 0, 0], backend, "float")
    groups = to_backend([0, 0, 0, 1, 1, 1], backend, "int")
    estimator = CovGap(alpha=0.25)
    val = estimator.evaluate(groups, cover)
    expected = 0.5 * 0.25 + 0.5 * 0.75
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_alpha_override(backend):
    cover = to_backend([1, 0, 1, 0], backend, "float")
    groups = to_backend([0, 0, 1, 1], backend, "int")
    estimator = CovGap(alpha=0.1)
    val1 = estimator.evaluate(groups, cover)
    val2 = estimator.evaluate(groups, cover, alpha=0.5)
    assert not np.isclose(val1, val2)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_unbalanced_group_sizes(backend):
    cover = to_backend([1, 1, 0, 0, 0], backend, "float")
    groups = to_backend([0, 0, 1, 1, 1], backend, "int")
    estimator = CovGap(alpha=0.5)
    expected = (2/5)*0.5 + (3/5)*0.5
    val = estimator.evaluate(groups, cover)
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_return_type_is_float(backend):
    cover = to_backend([1, 0, 1, 1], backend, "float")
    groups = to_backend([0, 0, 1, 1], backend, "int")
    estimator = CovGap(alpha=0.3)
    val = estimator.evaluate(groups, cover)
    assert isinstance(val, float)


# -------------------- ERROR HANDLING --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_length_mismatch_raises(backend):
    cover = to_backend([1, 0, 1], backend, "float")
    groups = to_backend([0, 0], backend, "int")  # shorter
    estimator = CovGap(alpha=0.2)
    with pytest.raises((ValueError, IndexError)):
        estimator.evaluate(groups, cover)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_wrong_shape_inputs(backend):
    if backend == "numpy":
        cover = np.array([[1, 0], [1, 1]])
        groups = np.array([0, 1])
    else:
        cover = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)
        groups = torch.tensor([0, 1], dtype=torch.int64)
    estimator = CovGap(alpha=0.2)
    with pytest.raises(Exception):
        estimator.evaluate(groups, cover)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_cover_values_must_be_binary(backend):
    groups = to_backend([0, 0, 1, 1], backend, "int")
    estimator = CovGap(alpha=0.2)

    cover_invalid = to_backend([0, 2, 1, -1], backend, "float")
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(groups, cover_invalid)

    cover_float_invalid = to_backend([0.0, 0.5, 1.0, 0.2], backend, "float")
    with pytest.raises((ValueError, AssertionError)):
        estimator.evaluate(groups, cover_float_invalid)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_non_integer_group_labels_raises(backend):
    cover = to_backend([1, 0, 1, 0], backend, "float")
    estimator = CovGap(alpha=0.2)

    if backend == "numpy":
        groups_str = np.array(["a", "a", "b", "b"])
        groups_float = np.array([0.1, 0.1, 1.5, 1.5])
        with pytest.raises((TypeError, ValueError)):
            estimator.evaluate(groups_str, cover)
        with pytest.raises((TypeError, ValueError)):
            estimator.evaluate(groups_float, cover)
    else:  # torch
        groups_float = torch.tensor([0.1, 0.1, 1.5, 1.5], dtype=torch.float32)
        with pytest.raises((TypeError, ValueError)):
            estimator.evaluate(groups_float, cover)


# -------------------- MORE COMPLEX GROUP STRUCTURES --------------------

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_multiple_groups_three_classes(backend):
    cover = to_backend([1, 0, 1, 0, 1, 0], backend, "float")
    groups = to_backend([0, 0, 1, 1, 2, 2], backend, "int")
    estimator = CovGap(alpha=0.2)
    expected = (2/6)*0.3 + (2/6)*0.3 + (2/6)*0.3
    val = estimator.evaluate(groups, cover)
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_groups_with_noncontiguous_labels(backend):
    cover = to_backend([1, 1, 0, 0], backend, "float")
    groups = to_backend([10, 10, 42, 42], backend, "int")
    estimator = CovGap(alpha=0.5)
    expected = 0.5*0.5 + 0.5*0.5
    val = estimator.evaluate(groups, cover)
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_many_groups_small_sizes(backend):
    cover = to_backend([1, 0, 1, 0], backend, "float")
    groups = to_backend([0, 1, 2, 3], backend, "int")
    estimator = CovGap(alpha=0.25)
    expected = (0.25+0.75+0.25+0.75)/4
    val = estimator.evaluate(groups, cover)
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_large_group_and_small_groups(backend):
    cover = to_backend([1]*50 + [0,1,0], backend, "float")
    groups = to_backend([0]*50 + [1,2,3], backend, "int")
    estimator = CovGap(alpha=0.1)
    expected = (50/50-0.9)*0.25 + (0.9-0/1)*0.25 + (1/1-0.9)*0.25 + (0.9-0/1)*0.25
    val = estimator.evaluate(groups, cover)
    assert np.isclose(val, expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_large_group_and_small_groups_weithed(backend):
    cover = to_backend([1]*50 + [0,1,0], backend, "float")
    groups = to_backend([0]*50 + [1,2,3], backend, "int")
    estimator = CovGap(alpha=0.1)
    expected = (50/50-0.9)*50/53 + (0.9-0/1)*1/53 + (1/1-0.9)*1/53 + (0.9-0/1)*1/53
    val = estimator.evaluate(groups, cover, weighted=True)
    assert np.isclose(val, expected)

@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_alpha_valid_and_invalid(backend):
    # valid alpha
    if backend == "numpy":
        cover = np.array([1, 0, 1, 0])
        groups = np.array([0, 0, 1, 1])
    else:
        cover = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        groups = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    # valid alpha in (0,1)
    for good_alpha in [0.0, 0.5, 1.0]:
        estimator = CovGap(alpha=good_alpha)
        val = estimator.evaluate(groups, cover)
        assert isinstance(val, float)

    # invalid alphas
    for bad_alpha in [-0.1, 1.5]:
        with pytest.raises((ValueError, AssertionError, TypeError)):
            estimator = CovGap(alpha=bad_alpha)
            estimator.evaluate(groups, cover, alpha=bad_alpha)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_emptyness(backend):
    X = to_backend([], backend, "float")
    cover = to_backend([], backend, "int")
    estimator = CovGap(alpha=0.1)
            
    # Expect either ValueError or TypeError
    with pytest.raises((ValueError, TypeError)):
        estimator.evaluate(X, cover)