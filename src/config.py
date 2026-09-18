"""
설정 로딩.
사용 위치: 모든 스크립트가 맨 처음 `from src.config import load_config` 로 불러 씀.
"""
from __future__ import annotations
import copy
import os
from types import SimpleNamespace
import yaml


def _to_ns(d):
    """dict -> 점(.)으로 접근 가능한 네임스페이스(재귀)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_ns(v) for v in d]
    return d


def load_config(path: str = "config.yaml", overrides: dict | None = None):
    """
    config.yaml 을 읽어 네임스페이스로 반환.
    overrides: {"train.epochs": 5, "data.root": "data2"} 형태의 평면 dict.
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
    """CLI 에서 받은 ['train.epochs=5', ...] -> dict. 값은 YAML로 파싱."""
    out = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out
