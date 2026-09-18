"""
Shared utilities.
Used by: several modules.
"""
from __future__ import annotations
import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Fix seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def pick_device(pref: str = "auto"):
    """Choose a torch device."""
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_tqdm(iterable, **kw):
    """Use tqdm progress bar if available, otherwise pass through."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except Exception:
        return iterable
