"""
공용 유틸.
사용 위치: 여러 모듈에서 import.
"""
from __future__ import annotations
import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    """재현성 고정."""
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
    """torch device 선택."""
    import torch
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_tqdm(iterable, **kw):
    """tqdm 있으면 진행바, 없으면 그대로."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except Exception:
        return iterable
