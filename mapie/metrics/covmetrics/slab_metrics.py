import torch 
import numpy as np

from covmetrics.check import *
from covmetrics.utils import seed_everything


# Original code from https://github.com/Shai128/oqr


def sample_sphere_numpy(n_samples, dim):
    """Sample n_samples points uniformly on a sphere in R^dim (NumPy)."""
    vectors = np.random.randn(n_samples, dim)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def sample_sphere_torch(n_samples, dim):
    """Sample n_samples points uniformly on a sphere in R^dim (PyTorch)."""
    vectors = torch.randn((n_samples, dim))
    vectors /= vectors.norm(dim=1, keepdim=True)
    return vectors

def wsc_all_numpy(x, coverage, delta, V):
    """Compute worst slab coverage (NumPy backend)."""
    n, p = x.shape
    min_points_required = int(np.ceil(delta * n))
    n_directions = V.shape[0]

    # projections: shape (n_directions, n)
    projection = (x @ V.T).T
    order = np.argsort(projection, axis=1)
    projection_sorted = np.take_along_axis(projection, order, axis=1)
    coverage_sorted = np.take_along_axis(
        np.broadcast_to(coverage, (n_directions, n)), order, axis=1
    )

    max_start_index = n - min_points_required + 1
    arange_cache = np.arange(1, n + 1)

    best_coverages = np.ones(n_directions)
    best_start = np.zeros(n_directions, dtype=int)
    best_end = np.full(n_directions, n, dtype=int)

    for start_index in range(max_start_index):
        min_end_index = min(min_points_required, n) - 1
        cumulative_sums = np.cumsum(coverage_sorted[:, start_index:], axis=1)
        denom = arange_cache[: n - start_index][None, :]
        coverage_vals = cumulative_sums / denom
        coverage_vals[:, :min_end_index] = 1.0

        relative_end_index = np.argmin(coverage_vals, axis=1)
        current_cover = coverage_vals[np.arange(n_directions), relative_end_index]

        update_mask = current_cover < best_coverages
        best_coverages[update_mask] = current_cover[update_mask]
        best_start[update_mask] = start_index
        best_end[update_mask] = start_index + relative_end_index[update_mask]

    a_vals = projection_sorted[np.arange(n_directions), best_start]
    b_vals = projection_sorted[np.arange(n_directions), best_end - 1]
    return best_coverages, a_vals, b_vals

def wsc_all_torch(x, coverage, delta, V):
    """Compute worst slab coverage (Torch backend)."""
    n, p = x.shape
    min_points_required = int(np.ceil(delta * n))
    n_directions = V.shape[0]

    # projections: shape (n_directions, n)
    projection = (x @ V.T).T
    projection_sorted, order = torch.sort(projection, dim=1)
    coverage_sorted = torch.gather(coverage.expand(n_directions, -1), 1, order)

    max_start_index = n - min_points_required + 1
    arange_cache = torch.arange(1, n + 1)

    best_coverages = torch.ones(n_directions)
    best_start = torch.zeros(n_directions, dtype=torch.long)
    best_end = torch.full((n_directions,), n, dtype=torch.long)

    for start_index in range(max_start_index):
        min_end_index = min(min_points_required, n) - 1
        cumulative_sums = torch.cumsum(coverage_sorted[:, start_index:], dim=1)
        denom = arange_cache[: n - start_index].unsqueeze(0)
        coverage_vals = cumulative_sums / denom
        coverage_vals[:, :min_end_index] = 1.0

        relative_end_index = torch.argmin(coverage_vals, dim=1)
        current_cover = coverage_vals[torch.arange(n_directions), relative_end_index]

        update_mask = current_cover < best_coverages
        best_coverages[update_mask] = current_cover[update_mask]
        best_start[update_mask] = start_index
        best_end[update_mask] = start_index + relative_end_index[update_mask]

    a_vals = projection_sorted[torch.arange(n_directions), best_start]
    b_vals = projection_sorted[torch.arange(n_directions), best_end - 1]
    return best_coverages, a_vals, b_vals


# ---------------- Main Class ----------------

class WSC:
    def __init__(self, delta=None):

        self.delta = delta

    def wsc(self, x, coverage, delta, M=1000):
        """
        Compute the worst slab coverage value for the input data.

        Parameters
            x: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            coverage: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as x. 
            delta: Tolerance parameter that controls the size of the slab. 
            M: Number of random directions drawn on the unit sphere for the search procedure.

        Returns
            A tuple with:
                coverage_value: Worst coverage value found.
                direction: The direction vector on the sphere that produces this value.
                a: Auxiliary scalar returned by the backend solver.
                b: Auxiliary scalar returned by the backend solver.

        Notes
            It draws M random unit vectors and evaluates the coverage for each. The minimum value is returned.
        """
        use_torch = isinstance(x, torch.Tensor) or isinstance(coverage, torch.Tensor)

        if use_torch:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=coverage.dtype)
            if not isinstance(coverage, torch.Tensor):
                coverage = torch.tensor(coverage, dtype=x.dtype)
        else:
            if isinstance(x, list):
                x = np.asarray(x, dtype=float)
            if isinstance(coverage, list):
                coverage = np.asarray(coverage, dtype=float)

        # main loop
        if use_torch:    
            V = sample_sphere_torch(M, x.shape[1])
            coverage_list, a_list, b_list = wsc_all_torch(x, coverage, delta, V)
            idx_star = torch.argmin(coverage_list).item()
            return (
                float(coverage_list[idx_star].item()),
                V[idx_star].detach().cpu().numpy(),
                float(a_list[idx_star].item()),
                float(b_list[idx_star].item()),
            )
        else:
            V = sample_sphere_numpy(M, x.shape[1])
            coverage_list, a_list, b_list = wsc_all_numpy(x, coverage, delta, V)
            idx_star = int(np.argmin(coverage_list))
            return (
                float(coverage_list[idx_star]),
                V[idx_star],
                float(a_list[idx_star]),
                float(b_list[idx_star]),
            )

    def evaluate(self, x, cover, delta=None, M=1000, seed=42):
        """
        Compute the worst slab coverage value for the input data.

        Parameters
            x: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as x. 
            delta: Tolerance parameter that controls the size of the slab. 
            M: Number of random directions drawn on the unit sphere for the search procedure.

        Returns
            Float: WSC estimated

        Notes
            The function detects whether numpy or torch is in use and dispatches to the matching backend.
            It draws M random unit vectors and evaluates the coverage for each. The minimum value is returned.
        """
        seed_everything(seed)
        if delta is None:
            delta = self.delta
        check_delta(delta)
        check_cover(cover)
        check_tabular_strict(x)
        check_consistency(cover, x)

        wsc_value, _, _, _ = self.wsc(
            x, cover, delta=delta, M=M
        )
        return float(wsc_value)

