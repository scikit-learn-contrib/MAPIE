import torch 
from typing import Literal
import numpy as np
from covmetrics.check import *

class CovGap:
    def __init__(self, alpha=None):
        """
        Initialize the CovGap metric
        alpha: Float in (0,1). The targetet miscoverage value. 
        """
        self.alpha=alpha
        
    def evaluate(self, groups, cover, alpha=None, weighted=False):    
        """
        Compute the CovGap.

        Parameters
            groups: Groups memberships. Either a numpy array or a torch tensor with shape (n,) and values indicating the groups memberships.
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as sizes. 
            alpha: Float in (0,1). The targetet miscoverage value. 
            alpha: (optional) Boolean Default=False, should compute the weighted or unweighted version of covgap.
        Returns
            Float: FSC estimated

        Notes
            The function detects whether numpy or torch is in use and dispatches to the matching backend.
        """
        
        if alpha is None:
            alpha = self.alpha
        
        # --- checks ---
        check_alpha(alpha)
        check_boolean(weighted)
        check_emptyness(cover)
        check_emptyness(groups)
        check_cover(cover)
        check_groups(groups)
        check_consistency(cover, groups)

        # --- Torch branch ---
        if isinstance(cover, torch.Tensor) or isinstance(groups, torch.Tensor):
            if not isinstance(cover, torch.Tensor):
                cover = torch.tensor(cover, dtype=torch.float32)
            if not isinstance(groups, torch.Tensor):
                groups = torch.tensor(groups, dtype=torch.int64)

            unique_groups = torch.unique(groups)
            total_samples = cover.numel()
            cover_gaps = []

            if weighted:
                for group in unique_groups:
                    group_indices = (groups == group)
                    group_size = group_indices.sum()
                    if group_size == 0:
                        continue
                    group_cover = (cover[group_indices] >= 1).float().mean()
                    gap = torch.abs(group_cover - (1 - alpha))
                    weighted_gap = (group_size.float() / total_samples) * gap
                    cover_gaps.append(weighted_gap)
                return torch.stack(cover_gaps).sum().item()

            for group in unique_groups:
                group_indices = (groups == group)
                group_size = group_indices.sum()
                if group_size == 0:
                    continue
                group_cover = (cover[group_indices] >= 1).float().mean()
                gap = torch.abs(group_cover - (1 - alpha))
                cover_gaps.append(gap)

            return torch.stack(cover_gaps).mean().item()

        # --- Numpy branch ---
        if isinstance(cover, list):
            cover = np.array(cover, dtype=float)
        if isinstance(groups, list):
            groups = np.array(groups)

        unique_groups = np.unique(groups)
        total_samples = cover.size
        cover_gaps = []

        if weighted:
            for group in unique_groups:
                group_indices = (groups == group)
                group_size = group_indices.sum()
                if group_size == 0:
                    continue
                group_cover = (cover[group_indices] >= 1).astype(float).mean()
                gap = abs(group_cover - (1 - alpha))
                weighted_gap = (group_size / total_samples) * gap
                cover_gaps.append(weighted_gap)

            return float(np.sum(cover_gaps))

        for group in unique_groups:
            group_indices = (groups == group)
            group_size = group_indices.sum()
            if group_size == 0:
                continue
            group_cover = (cover[group_indices] >= 1).astype(float).mean()
            gap = abs(group_cover - (1 - alpha))
            cover_gaps.append(gap)

        return float(np.mean(cover_gaps))

class FSC:
    def __init__(self):
        pass

    def evaluate(self, groups, cover):
        """
        Compute the FSC. 

        Parameters
            groups: Groups memberships. Either a numpy array or a torch tensor with shape (n,) and values indicating the groups memberships.
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (y in C(X)). Same type and length as sizes. 
        Returns
            Float: FSC estimated
        """
        check_emptyness(cover)
        check_cover(cover)
        check_groups(groups)
        check_consistency(cover, groups)
        
        use_torch = isinstance(cover, torch.Tensor) or isinstance(groups, torch.Tensor)
        
        if use_torch:
            if not isinstance(cover, torch.Tensor):
                cover = torch.tensor(cover, dtype = groups.dtype)
            if not isinstance(groups, torch.Tensor):
                groups = torch.tensor(groups, dtype = cover.dtype)
            unique_groups = torch.unique(groups)
            fsc_value = torch.tensor(1.0, device=cover.device)
        else:
            if isinstance(cover, list):
                cover = np.array(cover)
            if isinstance(groups, list):
                groups = np.array(groups)
            
            unique_groups = np.unique(groups)
            fsc_value = 1.0  # float

        for group in unique_groups:
            if use_torch:
                group_indices = (groups == group)
                group_size = group_indices.sum()
                if group_size == 0:
                    continue
                group_cover = (cover[group_indices] >= 1).float().mean()
                fsc_value = torch.min(fsc_value, group_cover)
            else:
                group_indices = (groups == group)
                group_size = group_indices.sum()
                if group_size == 0:
                    continue
                group_cover = ((cover[group_indices] >= 1).astype(float)).mean()
                fsc_value = min(fsc_value, group_cover)

        return fsc_value.item() if use_torch else float(fsc_value)
    
class EOC:
    def __init__(self, alpha=None):
        self.alpha = alpha

    def grouping(self, y, number_max_groups):
        """
        Group the vectors y using k-means

        Parameters
            y: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            number_max_groups: Integer

        Returns
            Vector with groups memberships
        """
        check_emptyness(y)
        
        # ---- Torch backend ----
        if isinstance(y, torch.Tensor):
            if len(torch.unique(y)) >= number_max_groups:
                grouping = KMeansGrouping()    
                if len(y.shape) == 1:
                    y_ = y.unsqueeze(1)                
                    grouping.fit(y_, n_groups=number_max_groups)
                    groups = grouping(y_)
                else:
                    grouping.fit(y, n_groups=number_max_groups)
                    groups = grouping(y)
                return groups
            else:
                unique_values, y_int = torch.unique(y, return_inverse=True)
                if len(y_int.shape) == 2:
                    y_int = y_int.squeeze(1)
                return y_int

        # ---- NumPy / list backend ----
        if isinstance(y, list):
            y = np.array(y)

        if len(np.unique(y)) >= number_max_groups:
            grouping = KMeansGrouping()
            if len(y.shape) == 1:
                y_ = np.expand_dims(y, axis=1)
                grouping.fit(y_, n_groups=number_max_groups)
                groups = grouping(y_)
            else:
                grouping.fit(y, n_groups=number_max_groups)
                groups = grouping(y)
            return groups
        else:
            unique_values, y_int = np.unique(y, return_inverse=True)
            if len(y_int.shape) == 2:
                y_int = y_int.flatten()
            return y_int
        
    def evaluate_FSC(self, y, cover, number_max_groups=10):
        """
        Compute the EOC FSC between the vector sizes and cover. 

        Parameters
            y: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (y in C(X)). Same type and length as sizes. 
            number_max_groups: Max number of groups to estimate this metric
        Returns
            Float: FSC estimated
        """
        groups = self.grouping(y, number_max_groups)
        return FSC().evaluate(groups, cover)
    
    def evaluate_CovGap(self, y, cover, alpha=None, number_max_groups=10, weighted=False):
        """
        Compute the EOC CovGap between the vector sizes and cover. 

        Parameters
            y: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same length and type as sizes. 
            number_max_groups: Max number of groups to estimate this metric
        Returns
            Float: CovGap estimated
        """
        if alpha is None:
            alpha = self.alpha
        groups = self.grouping(y, number_max_groups)
        return CovGap(alpha).evaluate(groups, cover, weighted=weighted)
    
    def evaluate(self, y, cover, alpha=None, number_max_groups=10, weighted=False):
        """
        Compute the EOC CovGap between the vector sizes and cover. 

        Parameters
            y: Input samples. Either a numpy array or a torch tensor with shape (n, d).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as sizes. 
            number_max_groups: Max number of groups to estimate this metric
        Returns
            Float: CovGap estimated

        """
        if alpha is None:
            alpha = self.alpha
        return self.evaluate_CovGap(y, cover, alpha, number_max_groups=number_max_groups, weighted=weighted)

class SSC:
    def __init__(self, alpha=None):
        self.estimator = EOC(alpha)

    def evaluate_FSC(self, size, cover, number_max_groups=10):
        """
        Compute the SSC between the vector sizes and cover. 

        Parameters
            sizes: Input samples. Either a numpy array or a torch tensor with shape (n,).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as sizes. 

        Returns
            Float: FSC estimated
        """
        return self.estimator.evaluate_FSC(size, cover, number_max_groups=number_max_groups)
    
    def evaluate_CovGap(self, size, cover, alpha=None, number_max_groups=10, weighted=False):
        """
        Compute the SSC between the vector sizes and cover. 

        Parameters
            sizes: Input samples. Either a numpy array or a torch tensor with shape (n,).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same type and length as sizes. 

        Returns
            Float: CovGap estimated
        """
        return self.estimator.evaluate_CovGap(size, cover, alpha = alpha, number_max_groups=number_max_groups, weighted=weighted)
    
    def evaluate(self, size, cover, alpha=None, number_max_groups=10, weighted=False):
        """
        Compute the SSC between the vector sizes and cover. 

        Parameters
            sizes: Input samples. Either a numpy array or a torch tensor with shape (n,).
            cover: 1 and 0 vector containing the coverage values associated with each sample. 1 = (yin C(X)). Same length and type as sizes. 

        Returns
            Float: CovGap estimated
        """
        return self.estimator.evaluate(size, cover, alpha = alpha, number_max_groups=number_max_groups, weighted=weighted)

class KMeansGrouping:
    def __init__(self):
        pass

    def fit(self, x_train, n_groups=5, max_iter=100, tol=1e-4, device=None, seed=None):
        centroids, labels = ClusteringGroupingFunction(x_train, n_groups=n_groups, max_iter=max_iter, tol=tol, device=device, seed=seed)
        self.centroids = centroids
        self.labels = labels

    def to(self, device):
        self.centroids = self.centroids.to(device)
        self.labels = self.labels.to(device)
        return self

    def __call__(self, X):
        """
        Assign rows of X to nearest centroid.
        X: (m, k) or (k,) or (m,) depending on k.
        returns: (m,) long tensor or numpy array of cluster indices
        """
        is_tensor_input = torch.is_tensor(X)
        
        if not is_tensor_input:
            X = torch.as_tensor(X, device=self.centroids.device)
        
        # ensure 2D
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        # compute squared distances: (m, n_groups)
        dists = torch.cdist(X, self.centroids, p=2)  # (m, n_groups)
        labels_new = torch.argmin(dists, dim=1).to(torch.long)
        
        if is_tensor_input:
            return labels_new
        else:
            return labels_new.cpu().numpy()

def ClusteringGroupingFunction(x_train,
                              n_groups,
                              max_iter: int = 100,
                              tol: float = 1e-4,
                              device: torch.device | str | None = None,
                              seed: int | None = None):
    """
    Performs K-means clustering on x_train (rows = samples) and returns a grouping object.

    Args:
        x_train: tensor-like, shape (n, k)
        n_groups: int, number of clusters
        max_iter: maximum kmeans iterations
        tol: tolerance for centroid change to stop
        device: device to run on (None uses x_train's device or CPU)
        seed: random seed for centroid init (optional)

    Returns:
        _KMeansGrouping object with attributes:
          - labels: (n,) long tensor of cluster assignments for x_train
          - centroids: (n_groups, k) tensor of centroid coordinates
          and callable mapping new X -> labels
    """
    if seed is not None:
        torch.manual_seed(seed)

    X = torch.as_tensor(x_train)
    if device is None:
        device = X.device
    else:
        device = torch.device(device)
        X = X.to(device)

    if X.dim() == 1:
        X = X.unsqueeze(0)   # (1, k) treat single sample

    n, k = X.shape
    if n_groups <= 0 or n_groups > n:
        raise ValueError("n_groups must be in 1..n (number of samples)")

    # initialize centroids by sampling k points from X (kmeans++)
    # We'll implement a simple kmeans++ init for better stability.
    centroids = torch.empty((n_groups, k), device=device, dtype=X.dtype)
    # pick first centroid uniformly
    idx = torch.randint(low=0, high=n, size=(1,), device=device)
    centroids[0] = X[idx]

    # kmeans++ initialization
    for i in range(1, n_groups):
        # compute distance of each point to nearest current centroid
        dists = torch.cdist(X, centroids[:i], p=2)  # (n, i)
        min_dists, _ = torch.min(dists, dim=1)     # (n,)
        probs = min_dists.pow(2)
        # avoid zero-sum
        if probs.sum() <= 1e-12 or torch.any(torch.isnan(probs)):
            # fallback to uniform
            idx = torch.randint(low=0, high=n, size=(1,), device=device)
        else:
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, num_samples=1)
        centroids[i] = X[idx]

    # main k-means loop
    prev_centroids = centroids.clone()
    for it in range(max_iter):
        # assign each point to nearest centroid
        dists = torch.cdist(X, centroids, p=2)  # (n, n_groups)
        labels = torch.argmin(dists, dim=1)     # (n,)

        # compute new centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_groups, device=device, dtype=X.dtype)
        # accumulate sums
        for g in range(n_groups):
            mask = (labels == g)
            cnt = mask.sum().item()
            if cnt == 0:
                # empty cluster: reinitialize to a random point
                new_centroids[g] = X[torch.randint(0, n, (1,), device=device)]
                counts[g] = 1.0
            else:
                selected = X[mask]
                new_centroids[g] = selected.mean(dim=0)
                counts[g] = cnt

        # check centroid shift
        shift = torch.norm(new_centroids - centroids, dim=1).max().item()
        centroids = new_centroids
        if shift <= tol:
            break

    # final assignment
    dists = torch.cdist(X, centroids, p=2)
    labels = torch.argmin(dists, dim=1).to(torch.long)

    return centroids, labels

class BinaryGroupingFunction:
    def __init__(self, val: float, mode: Literal['all', 'any', 'mean'] = 'all'):
        """
        val : threshold value (float or a tensor broadcastable with rows)
        mode: how to reduce a row to a single boolean:
              'all'  -> True if all features in row <= val
              'any'  -> True if any feature in row <= val
              'mean' -> True if row.mean() <= val
        """
        if mode not in ('all', 'any', 'mean'):
            raise ValueError("mode must be one of 'all','any','mean'")
        self.val = val
        self.mode = mode

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: tensor of shape (n, k) (or (k,) / (n,) handled too)
        returns: tensor of shape (n,) with dtype torch.long (0 or 1)
        """
        if X.dim() == 0:
            # scalar -> single-element tensor
            comp = (X <= self.val)
            return comp.to(dtype=torch.long).reshape(())
        if X.dim() == 1:
            # treat single row (k,) -> return shape (1,) for consistency
            X = X.unsqueeze(0)  # (1,k)

        # now X has shape (n, k)
        if self.mode == 'all':
            row_bool = (X <= self.val).all(dim=1)
        elif self.mode == 'any':
            row_bool = (X <= self.val).any(dim=1)
        else:  # 'mean'
            # compute row mean and compare
            row_mean = X.mean(dim=1)
            # if val is tensor broadcastable this still works
            row_bool = (row_mean <= self.val)

        return row_bool.to(dtype=torch.long)  # shape (n,)

