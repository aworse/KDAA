# -*- coding: utf-8 -*-
"""
PyTorch Dataset + split logic.
Used by: train.py, evaluate.py.

Key point (reviewers always ask about this):
  * The default split is 'session-wise'. Randomly splitting the same recording
    session into train/test overestimates accuracy, so we separate by session.
  * split='participant' measures 'cross-user generalization' separately.
  * split='random' is (not recommended) for an upper-bound sanity check.
Labels (y) are jamo strings, mapped to integer indices by LabelSpace.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:                       # importable even without torch
    _HAS_TORCH = False
    Dataset = object

from .audio_io import load_audio
from .features import MelExtractor
from . import augment as A


class LabelSpace:
    """jamo label <-> integer index."""
    def __init__(self, labels):
        self.labels = list(labels)
        self.l2i = {l: i for i, l in enumerate(self.labels)}
        self.i2l = {i: l for l, i in self.l2i.items()}

    def __len__(self):
        return len(self.labels)

    @classmethod
    def from_metadata(cls, df, label_col="jamo"):
        labels = sorted(df[label_col].astype(str).unique().tolist())
        return cls(labels)

    def encode(self, lab):
        return self.l2i[str(lab)]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.labels))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls([l for l in f.read().splitlines() if l != ""])


def make_split(df, cfg, seed):
    """
    Return (train_df, test_df).
    session: split the set of sessions 8:2 (stratified by scenario/participant).
    participant: hold out the last participant (cross-user generalization).
    random: row-level random split.
    """
    rng = np.random.RandomState(seed)
    mode = cfg.train.split
    if mode == "random":
        idx = rng.permutation(len(df))
        n_test = int(len(df) * cfg.train.val_ratio)
        test_i = set(idx[:n_test].tolist())
        mask = df.index.isin([df.index[i] for i in test_i])
        return df[~mask].copy(), df[mask].copy()

    if mode == "participant":
        parts = sorted(df["participant"].unique())
        test_p = parts[-1]
        return df[df["participant"] != test_p].copy(), df[df["participant"] == test_p].copy()

    # default: session-wise (split sessions within each scenario, stratified)
    test_sessions = set()
    for scen, g in df.groupby("scenario"):
        sess = sorted(g["session"].unique())
        rng.shuffle(sess)
        n_test = max(1, int(round(len(sess) * cfg.train.val_ratio)))
        test_sessions.update(sess[:n_test])
    test_mask = df["session"].isin(test_sessions)
    return df[~test_mask].copy(), df[test_mask].copy()


class KeystrokeDataset(Dataset):
    """
    One row = one keystroke clip. __getitem__ -> (logmel tensor (1,M,T), label idx, meta dict)
    """
    def __init__(self, df, cfg, labelspace, train=False):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.ls = labelspace
        self.train = train
        self.root = cfg.data.root
        self.sr = cfg.data.sample_rate
        self.mel = MelExtractor(cfg.feature, self.sr)
        self.aug = cfg.augment

    def __len__(self):
        return len(self.df)

    def _load_wave(self, row):
        fp = row["filepath"]
        if not os.path.isabs(fp):
            fp = os.path.join(self.root, fp)
        wav, _ = load_audio(fp, self.sr)
        return wav

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = self._load_wave(row)
        if self.train and self.aug.enable:
            wav = A.time_shift(wav, self.sr, self.aug.time_shift_ms)
            wav = A.mix_noise(wav, self.aug.noise_snr_db, self.aug.noise_prob)
        logmel = self.mel(wav)                       # (1, M, T)
        if self.train and self.aug.enable:
            logmel = A.spec_augment(logmel, self.aug.specaug_time, self.aug.specaug_freq)
        y = self.ls.encode(row["jamo"])
        meta = dict(scenario=row["scenario"], participant=row["participant"],
                    session=row["session"], jamo=str(row["jamo"]))
        if _HAS_TORCH:
            return torch.from_numpy(logmel), int(y), meta
        return logmel, int(y), meta
