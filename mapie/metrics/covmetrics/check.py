import torch 
import pandas as pd
import numpy as np
import random
from covmetrics.utils import seed_everything
import inspect


def check_boolean(weighted):
    """Ensure weighted is a Boolean"""
    if not isinstance(weighted, bool):
        raise ValueError("Input must be a Boolean")
    return True

def check_alpha(alpha):
    """Ensure alpha is a float in (0,1)."""
    try:
        alpha = float(alpha)
    except Exception:
        raise TypeError("alpha must be a float or int.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1 (inclusive).")
    
def check_alpha_tab_ok(alpha, cover):
    """
    Ensure alpha is either:
    - a float/int in [0, 1], or
    - an array-like of the same type and length as cover,
      with all values in [0, 1].
    """
    if np.isscalar(alpha):
        try:
            alpha = float(alpha)
        except Exception:
            raise TypeError("alpha must be a float, int, or array-like.")
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1 (inclusive).")
        return alpha

    # Convert alpha and cover to arrays
    try:
        alpha_arr = np.asarray(alpha, dtype=float)
    except Exception:
        raise TypeError("alpha must be a float, int, or array-like.")

    cover_arr = np.asarray(cover)

    # Vérifie que alpha et cover ont le même type d'objet
    if type(alpha) != type(cover):
        raise TypeError("alpha and cover must be of the same type when alpha is not a scalar.")

    if alpha_arr.shape[0] != cover_arr.shape[0]:
        raise ValueError("alpha must have the same length as cover.")

    if np.any(alpha_arr < 0) or np.any(alpha_arr > 1):
        raise ValueError("All values in alpha must be between 0 and 1 (inclusive).")

    return alpha_arr

    
def check_delta(alpha):
    """Ensure delta is a float in (0,1)."""
    try:
        alpha = float(alpha)
    except Exception:
        raise TypeError("delta must be a float or int.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1 (delta).")

def check_tabular_1D(x):
    """
    Check that x is a valid 1D tabular array/vector.
    - Must be 1D
    - No NaN values
    - No ±Inf values
    - All entries are finite numbers
    """
    # Check dimensionality
    if x.ndim != 1:
        raise ValueError(f"x must be 1D (tabular vector), got shape {x.shape}")

    # NumPy array
    if isinstance(x, np.ndarray):
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains NaN, Inf, or -Inf values")
    # Torch tensor
    elif isinstance(x, torch.Tensor):
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN, Inf, or -Inf values")
    else:
        raise TypeError(f"x must be np.ndarray or torch.Tensor, got {type(x)}")

def check_n_splits(n_splits):
    """
    Checks if n_splits is a valid integer greater than 0.
    
    Raises:
        TypeError: If n_splits is not an integer.
        ValueError: If n_splits is less than 1.
    """
    if not isinstance(n_splits, int):
        raise TypeError(f"n_splits must be an integer, got {type(n_splits).__name__}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for cross-validation.")
    return True

def check_tabular(X):
    """
    Check that X is a valid tabular array/matrix.
    - Must be 2D
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if not isinstance(X, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"X must be np.ndarray or torch.Tensor or dataframe, got {type(X)}"
    )

    if X.ndim != 2:
        raise ValueError(f"X must be 2D (tabular), got shape {X.shape}")
    

def check_tabular_strict(X):
    """
    Check that X is a valid tabular array/matrix.
    - Must be 2D
    - No NaN values
    - No ±Inf values
    - All entries are finite numbers
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (tabular), got shape {X.shape}")
    # NumPy array
    if isinstance(X, np.ndarray):
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN, Inf, or -Inf values")
    # Torch tensor
    elif isinstance(X, torch.Tensor):
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN, Inf, or -Inf values")
    else:
        raise TypeError(f"X must be np.ndarray or torch.Tensor, got {type(X)}")

def check_cover(cover):
    """Ensure cover values are binary (0 or 1)."""
    # Convert pandas Series to NumPy array
    if isinstance(cover, pd.Series) or isinstance(cover, pd.DataFrame):
        cover = cover.values

    if isinstance(cover, torch.Tensor):
        if cover.ndim != 1:
            raise ValueError(f"cover must be 1D, got {cover.shape}")
        if not torch.all((cover == 0) | (cover == 1)):
            raise ValueError("cover values must be 0 or 1.")
    elif isinstance(cover, (list, np.ndarray)):
        cover = np.array(cover)
        if cover.ndim != 1:
            raise ValueError(f"cover must be 1D, got {cover.shape}")
        if not np.all(np.isin(cover, [0, 1])):
            raise ValueError("cover values must be 0 or 1.")
    else:
        raise TypeError("cover must be numpy array, list, pandas Series, or torch tensor.")

def check_emptyness(array):
    """
    Check if the input array or tensor is empty.
    
    Parameters:
        array: Can be a list, numpy array, or torch tensor.
    
    Raises:
        ValueError: If the input array is empty.
    """
    # For PyTorch tensors
    if isinstance(array, torch.Tensor):
        if array.numel() == 0:
            raise ValueError("The tensor is empty.")
    
    # For NumPy arrays
    elif isinstance(array, np.ndarray):
        if array.size == 0:
            raise ValueError("The numpy array is empty.")
            
    else:
        raise TypeError("Input must be a list, tuple, numpy array, or torch tensor.")

    return True  # Optional: Return True if not empty

def check_groups(groups):
    """Ensure group labels are integers."""
    if isinstance(groups, torch.Tensor):
        if groups.ndim != 1:
            raise ValueError(f"groups must be 1D, got {groups.shape}")
        if not groups.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("Group labels must be integers (torch int type).")
    elif isinstance(groups, (list, np.ndarray)):
        groups = np.array(groups)
        if groups.ndim != 1:
            raise ValueError(f"groups must be 1D, got {groups.shape}")
        if not np.issubdtype(groups.dtype, np.integer):
            raise TypeError("group labels must be integers.")
    else:
        raise TypeError("groups must be numpy array, list, or torch tensor.")

def check_consistency(cover, groups):
    """Ensure cover and groups have the same length and type."""
    # Convert pandas Series to NumPy arrays
    if isinstance(cover, pd.Series) or isinstance(cover, pd.DataFrame):
        cover = cover.values
    if isinstance(groups, pd.Series) or isinstance(groups, pd.DataFrame):
        groups = groups.values

    # Check type
    if type(cover) is not type(groups):
        raise TypeError(f"cover and groups must be of the same type, got {type(cover)} and {type(groups)}")
    
    # Check length
    if len(cover) != len(groups):
        raise ValueError(
            f"cover and groups must have the same length, got cover of shape {getattr(cover, 'shape', len(cover))} "
            f"and groups of shape {getattr(groups, 'shape', len(groups))}"
        )


def check_loss(loss_fn):
    test_inputs = [
        (np.array([0.1, 0.9]), np.array([0, 1])),
        (torch.tensor([0.3, 0.7]), torch.tensor([1, 0]))
    ]
    
    for p, q in test_inputs:
        try:
            sig = inspect.signature(loss_fn)

            has_alpha_arg = "alpha" in sig.parameters

            if has_alpha_arg:
                result = loss_fn(p, q, alpha=0.1)
            else:
                result = loss_fn(p, q)
            
            if isinstance(result, (np.ndarray, torch.Tensor)):
                if result.shape != np.shape(p):
                    raise ValueError(f"Result shape is incorrect: {result.shape}, expected {np.shape(p)}")
            elif isinstance(result, (float, int)):
                pass
            else:
                raise TypeError(f"Unexpected return type: {type(result)}")
        
        except Exception as e:
            raise ValueError(f"Loss function failed for p={p}, q={q}. Details: {e}")
    
    return True
