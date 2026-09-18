# -*- coding: utf-8 -*-
"""
Training script.
Usage:
  python -m src.train --config config.yaml
  python -m src.train --set train.epochs=5 model.arch=coatnet_lite
Outputs (under cfg.train.out_dir):
  best.pt (model+labels+cfg), labels.txt, train_df.csv/test_df.csv, history.json
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_config, parse_overrides
from .utils import set_seed, ensure_dir, pick_device, try_tqdm
from .dataset import KeystrokeDataset, LabelSpace, make_split
from .model import build_model


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    metas = [b[2] for b in batch]
    return xs, ys, metas


def run_epoch(model, loader, device, crit, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    for xs, ys, _ in try_tqdm(loader, leave=False):
        xs, ys = xs.to(device), ys.to(device)
        with torch.set_grad_enabled(train):
            logits = model(xs)
            loss = crit(logits, ys)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
        loss_sum += float(loss.detach()) * len(ys)
        correct += int((logits.argmax(1) == ys).sum())
        tot += len(ys)
    return loss_sum / max(tot, 1), correct / max(tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    set_seed(cfg.seed)
    device = pick_device(cfg.train.device)
    out = ensure_dir(cfg.train.out_dir)

    df = pd.read_csv(cfg.data.metadata)
    df["jamo"] = df["jamo"].astype(str)
    ls = LabelSpace.from_metadata(df)
    ls.save(os.path.join(out, "labels.txt"))

    tr_df, te_df = make_split(df, cfg, cfg.seed)
    tr_df.to_csv(os.path.join(out, "train_df.csv"), index=False)
    te_df.to_csv(os.path.join(out, "test_df.csv"), index=False)
    print(f"[split={cfg.train.split}] train={len(tr_df)} test={len(te_df)} "
          f"classes={len(ls)} device={device}")

    tr_ds = KeystrokeDataset(tr_df, cfg, ls, train=True)
    te_ds = KeystrokeDataset(te_df, cfg, ls, train=False)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.train.batch_size, shuffle=True,
                       num_workers=cfg.train.num_workers, collate_fn=collate)
    te_ld = DataLoader(te_ds, batch_size=cfg.train.batch_size, shuffle=False,
                       num_workers=cfg.train.num_workers, collate_fn=collate)

    model = build_model(cfg, len(ls)).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    best_acc, hist = -1.0, []
    for ep in range(1, cfg.train.epochs + 1):
        tl, ta = run_epoch(model, tr_ld, device, crit, opt)
        vl, va = run_epoch(model, te_ld, device, crit, None)
        sched.step()
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
        print(f"ep{ep:03d}  train {tl:.3f}/{ta:.3f}   val {vl:.3f}/{va:.3f}")
        if va > best_acc:
            best_acc = va
            torch.save(dict(model=model.state_dict(), labels=ls.labels,
                            cfg=cfg._raw, arch=cfg.model.arch), os.path.join(out, "best.pt"))
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[done] best val acc = {best_acc:.4f}  -> {out}/best.pt")


if __name__ == "__main__":
    main()
