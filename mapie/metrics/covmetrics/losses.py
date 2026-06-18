import torch
import numpy as np
import pandas as pd


def clip_under(x, val_max):
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, max=val_max)

    elif isinstance(x, np.ndarray):
        return np.clip(x, a_min=None, a_max=val_max)

    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.clip(upper=val_max)
    
    elif isinstance(x, (int, float)):
        return min(x, val_max)

    else:
        raise TypeError(f"Input must be a torch.Tensor, np.ndarray, pd.Series, or pd.DataFrame, got {type(x)}")
    
def clip_over(x, val_min):
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, min=val_min)

    elif isinstance(x, np.ndarray):
        return np.clip(x, a_min=val_min, a_max=None)

    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.clip(lower=val_min)

    elif isinstance(x, (int, float)):
        return max(x, val_min)

    else:
        raise TypeError(f"Input must be a torch.Tensor, np.ndarray, pd.Series, pd.DataFrame, or a number, got {type(x)}")


def brier_score(pred_proba, cover):
    return (pred_proba - cover)**2

def logloss(pred_proba, cover, eps=1e-6):
    if isinstance(cover, torch.Tensor):
        if not isinstance(pred_proba, torch.Tensor):
            pred_proba = torch.tensor(pred_proba, dtype=torch.float32)
        pred_proba = torch.clip(pred_proba, eps, 1-eps)
        return cover * torch.log(pred_proba) + (1-cover)*torch.log(1-pred_proba)
    else:
        pred_proba = np.clip(pred_proba, eps, 1-eps)
        return - cover * np.log(pred_proba) - (1-cover)*np.log(1-pred_proba)

def L1_miscoverage(pred_proba, cover, alpha):
    if isinstance(cover, pd.Series) or isinstance(cover, pd.DataFrame): cover = np.asarray(cover)
    if isinstance(pred_proba, pd.Series) or isinstance(pred_proba, pd.DataFrame): pred_proba = np.asarray(pred_proba)
    if isinstance(alpha, pd.Series) or isinstance(alpha, pd.DataFrame): alpha = np.asarray(alpha)
    threshold = 1 - alpha

    out = cover * 0.0
    
    pos = pred_proba < threshold
    neg = pred_proba > threshold

    out[pos] = (threshold - cover)[pos]
    out[neg] = -(threshold - cover)[neg]

    return - out

def brier_score_over(pred_proba, cover, alpha):
    return brier_score( clip_over(pred_proba, 1-alpha), cover)

def L1_miscoverage_over(pred_proba, cover, alpha):
    return L1_miscoverage( clip_over(pred_proba, 1-alpha), cover, alpha)

def logloss_over(pred_proba, cover, alpha, eps=1e-6):
    return logloss(  clip_over(pred_proba, 1-alpha), cover, eps=eps)

def brier_score_under(pred_proba, cover, alpha):
    return brier_score( clip_under(pred_proba, 1-alpha), cover)

def logloss_under(pred_proba, cover, alpha, eps=1e-6):
    return logloss( clip_under(pred_proba, 1-alpha), cover, eps=eps)

def L1_miscoverage_under(pred_proba, cover, alpha):
    return L1_miscoverage( clip_under(pred_proba, 1-alpha), cover, alpha)
