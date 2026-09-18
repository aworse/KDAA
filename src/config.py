"""
Configuration loading.
Used by: every script, via `from src.config import load_config` at startup.
"""
from __future__ import annotations
import copy
import os
from types import SimpleNamespace
import yaml


def _to_ns(d):
    """dict -> namespace with dotted access (recursive)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(v) for v in d]
    return d


def load_config(path: str = "config.yaml", overrides: dict | None = None):
    """
    Read config.yaml and return it as a namespace.
    overrides: a flat dict like {"train.epochs": 5, "data.root": "data2"}.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for key, val in overrides.items():
            parts = key.split(".")
            node = cfg
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val
    cfg["_raw"] = copy.deepcopy(cfg)
    return _to_ns(cfg)


def parse_overrides(pairs: list[str]) -> dict:
    """Turn CLI ['train.epochs=5', ...] into a dict; values parsed as YAML."""
    out = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out
