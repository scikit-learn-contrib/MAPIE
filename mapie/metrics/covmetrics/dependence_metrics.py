import torch 
from scipy import stats
import numpy as np
from covmetrics.check import *


class PearsonCorrelation:
    def __init__(self):
        pass

    def evaluate(self, sizes, cover):
        """
        Compute the Pearson Correlation between the vector sizes and cover. 

        Parameters
            sizes: Input samples. Either a numpy array or a torch tensor with shape (n,).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as sizes. 
        
        Returns
            Float: PearsonCorrelation estimated
        """
        # Torch backend
        check_emptyness(cover)
        check_emptyness(sizes)
        check_cover(cover)
        check_tabular_1D(sizes)
        check_consistency(cover, sizes)


        if isinstance(sizes, torch.Tensor) or isinstance(cover, torch.Tensor):
            if not isinstance(sizes, torch.Tensor):
                sizes = torch.tensor(sizes, dtype=torch.float32)
            if not isinstance(cover, torch.Tensor):
                cover = torch.tensor(cover, dtype=torch.float32)

            sizes = sizes.float()
            cover = cover.float()

            # Center the variables
            sizes_mean = sizes.mean()
            cover_mean = cover.mean()

            cov = ((sizes - sizes_mean) * (cover - cover_mean)).mean()
            std_sizes = sizes.std(unbiased=False)
            std_cover = cover.std(unbiased=False)

            corr = cov / (std_sizes * std_cover)
            return corr.item()

        # Numpy / Python backend
        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=float)
        if isinstance(cover, list):
            cover = np.array(cover, dtype=float)

        corr, _ = stats.pearsonr(cover, sizes)
        return float(corr)


# Original code from https://github.com/danielgreenfeld3/XIC 


def pairwise_distances(x):
    """Compute squared pairwise distances (backend-agnostic)."""
    if isinstance(x, torch.Tensor):
        instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()
    else:
        if isinstance(x, list):
            x = np.array(x, dtype=float)
        instances_norm = np.sum(x**2, axis=-1).reshape((-1, 1))
        return -2 * np.dot(x, x.T) + instances_norm + instances_norm.T

def GaussianKernelMatrix(x, sigma):
    """Gaussian kernel Gram matrix (backend-agnostic)."""
    dists = pairwise_distances(x)
    if isinstance(dists, torch.Tensor):
        return torch.exp(-dists / sigma)
    else:
        return np.exp(-dists / sigma)

def hsic_statistic(x, y, sigma_x=1, sigma_y=1):
    """Compute HSIC statistic (backend-agnostic)."""
    m, _ = x.shape
    K = GaussianKernelMatrix(x, sigma_x)
    L = GaussianKernelMatrix(y, sigma_y)

    if isinstance(K, torch.Tensor):
        H = torch.eye(m, device=K.device) - (1.0 / m) * torch.ones((m, m), device=K.device)
        HSIC = torch.trace(L @ H @ K @ H) / ((m - 1) ** 2)
        return HSIC
    else:
        H = np.eye(m) - (1.0 / m) * np.ones((m, m))
        HSIC = np.trace(L @ H @ K @ H) / ((m - 1) ** 2)
        return HSIC

# ---------- HSIC Class ----------

class HSIC:
    def __init__(self, sigma_x=1, sigma_y=1):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y  

    def evaluate(self, sizes, cover, max_number_samples=5000):
        """
        Compute the Hilbert–Schmidt independence criterion (HSIC) between the vector sizes and cover. 

        Parameters
            sizes: Input samples. Either a numpy array or a torch tensor with shape (n,).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as sizes. 
            max_number_samples: Maximum number of observations used in the computation. If the input is larger, a random subset of this size is selected to limit the computational cost.

        Returns
            Float: PearsonCorrelation estimated
        """
        check_emptyness(cover)
        check_emptyness(sizes)
        check_cover(cover)
        check_tabular_1D(sizes)
        check_consistency(cover, sizes)
        
        # Normalize input types
        if isinstance(cover, list):
            cover = np.array(cover, dtype=float)
        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=float)

        n = len(cover)
        idx = np.random.permutation(n)[:max_number_samples]

        cover_sub = cover[idx].reshape(len(idx), 1)
        sizes_sub = sizes[idx].reshape(len(idx), 1)

        if isinstance(cover, torch.Tensor) or isinstance(sizes, torch.Tensor):
            if not isinstance(cover_sub, torch.Tensor):
                cover_sub = torch.tensor(cover_sub, dtype=torch.float32)
            if not isinstance(sizes_sub, torch.Tensor):
                sizes_sub = torch.tensor(sizes_sub, dtype=torch.float32)

            return hsic_statistic(cover_sub, sizes_sub, self.sigma_x, self.sigma_y).item()
        else:
            return float(hsic_statistic(cover_sub, sizes_sub, self.sigma_x, self.sigma_y))
