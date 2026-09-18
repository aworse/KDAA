# -*- coding: utf-8 -*-
"""
평가 스크립트.
사용 위치 / 실행:
  python -m src.evaluate --config config.yaml --run runs/exp1
산출물(cfg.eval.figures_dir):
  confusion_matrix.png, topk_accuracy.png, per_scenario_accuracy.png,
  syllable_recovery.png, training_curve.png, metrics.json
지표(사용자 선호에 따라 표 대신 흑백 figure로 출력):
  * top-1 / top-5 정확도 (chance = 1/클래스수 기준선 표시)
  * 시나리오(근접/원거리/배경소음)별 정확도 (독립변인 효과)
  * 혼동행렬
  * '오토마타 필터 적용 전(greedy) vs 후(제약 빔서치)' 음절 복원율
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from .config import load_config, parse_overrides
from .utils import pick_device
from .dataset import KeystrokeDataset, LabelSpace, make_split
from .model import build_model
from .train import collate
from . import hangul as H
from . import figures as FIG


def _clip_order_key(cid: str):
    """clip_id '<sid>_<idx>' 에서 정렬용 (sid, idx)."""
    parts = str(cid).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return str(cid), 0


def load_run(run_dir, cfg, device):
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    ls = LabelSpace(ckpt["labels"])
    model = build_model(cfg, len(ls)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ls


@torch.no_grad()
def predict_logprobs(model, ds, device, batch_size):
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    all_lp, all_y, all_meta = [], [], []
    for xs, ys, metas in ld:
        logits = model(xs.to(device))
        lp = F.log_softmax(logits, dim=1).cpu().numpy()
        all_lp.append(lp); all_y.append(ys.numpy()); all_meta.extend(metas)
    return np.concatenate(all_lp), np.concatenate(all_y), all_meta


def topk_acc(logprobs, y, k):
    topk = np.argsort(-logprobs, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def syllable_recovery(test_df, logprobs, meta, ls, beam_width, orphan_penalty=2.0):
    """
    세션별로 타건 순서를 복원해 정답 텍스트를 만들고,
    (a) greedy(오토마타 미적용) (b) 제약 빔서치(오토마타 적용) 의
    문자 단위 음절 복원율을 계산.
    반환: dict(before=.., after=.., examples=[(gt, before, after), ...])
    """
    df = test_df.copy().reset_index(drop=True)
    df["_row"] = range(len(df))
    order = sorted(df["_row"], key=lambda r: _clip_order_key(df.iloc[r]["clip_id"]))
    idx2label = ls.i2l

    def char_acc(gt, pred):
        n = max(len(gt), 1)
        m = min(len(gt), len(pred))
        correct = sum(1 for i in range(m) if gt[i] == pred[i])
        return correct / n

    before_scores, after_scores, examples = [], [], []
    for sid, g in df.groupby("session"):
        rows = sorted(g["_row"].tolist(), key=lambda r: _clip_order_key(df.iloc[r]["clip_id"]))
        gt_jamo = [str(df.iloc[r]["jamo"]) for r in rows]
        gt_text = H.compose(gt_jamo)
        lp = logprobs[rows]
        _, before_text = H.greedy_decode(lp, idx2label)
        _, after_text, _ = H.constrained_beam_decode(
            lp, [idx2label[i] for i in range(len(ls))],
            beam_width=beam_width, orphan_penalty=orphan_penalty)
        before_scores.append(char_acc(gt_text, before_text))
        after_scores.append(char_acc(gt_text, after_text))
        examples.append((gt_text, before_text, after_text))
    return dict(before=float(np.mean(before_scores)),
                after=float(np.mean(after_scores)),
                examples=examples[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default=None, help="학습 산출 디렉토리(기본 cfg.train.out_dir)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    run_dir = a.run or cfg.train.out_dir
    device = pick_device(cfg.train.device)
    figdir = cfg.eval.figures_dir
    os.makedirs(figdir, exist_ok=True)

    # test 셋: 학습 때 저장한 test_df.csv 사용(동일 분할 보장)
    te_path = os.path.join(run_dir, "test_df.csv")
    if os.path.exists(te_path):
        te_df = pd.read_csv(te_path)
    else:
        df = pd.read_csv(cfg.data.metadata)
        _, te_df = make_split(df, cfg, cfg.seed)
    te_df["jamo"] = te_df["jamo"].astype(str)

    model, ls = load_run(run_dir, cfg, device)
    ds = KeystrokeDataset(te_df, cfg, ls, train=False)
    lp, y, meta = predict_logprobs(model, ds, device, cfg.train.batch_size)

    # --- top-k ---
    metrics = {"n_test": int(len(y)), "n_classes": len(ls),
               "chance": 1.0 / len(ls)}
    ks = list(cfg.eval.topk)
    for k in ks:
        metrics[f"top{k}"] = topk_acc(lp, y, k)
    FIG.bar_fig([f"top-{k}" for k in ks], [metrics[f"top{k}"] for k in ks],
                os.path.join(figdir, "topk_accuracy.png"),
                ylabel="accuracy", title="Key(jamo) classification accuracy",
                baseline=metrics["chance"])

    # --- 시나리오별 정확도 (독립변인) ---
    scen_names = sorted(set(m["scenario"] for m in meta))
    pred = lp.argmax(1)
    scen_acc = []
    for s in scen_names:
        idx = [i for i, m in enumerate(meta) if m["scenario"] == s]
        scen_acc.append(float(np.mean(pred[idx] == y[idx])) if idx else 0.0)
    metrics["per_scenario_top1"] = dict(zip(scen_names, scen_acc))
    FIG.bar_fig(scen_names, scen_acc, os.path.join(figdir, "per_scenario_accuracy.png"),
                ylabel="top-1 accuracy", title="Accuracy by attack scenario (IV)",
                baseline=metrics["chance"])

    # --- 혼동행렬 ---
    cm = confusion_matrix(y, pred, labels=list(range(len(ls))))
    FIG.confusion_matrix_fig(cm, ls.labels, os.path.join(figdir, "confusion_matrix.png"),
                             title="Confusion matrix (jamo)")

    # --- 음절 복원율: 오토마타 전/후 ---
    rec = syllable_recovery(te_df, lp, meta, ls, cfg.eval.beam_width)
    metrics["syllable_recovery_before_automata"] = rec["before"]
    metrics["syllable_recovery_after_automata"] = rec["after"]
    FIG.bar_fig(["greedy\n(no automata)", "beam\n(+automata)"],
                [rec["before"], rec["after"]],
                os.path.join(figdir, "syllable_recovery.png"),
                ylabel="syllable char accuracy",
                title="Syllable recovery: automata filter effect")

    # --- 학습곡선 ---
    hp = os.path.join(run_dir, "history.json")
    if os.path.exists(hp):
        with open(hp) as f:
            FIG.curve_fig(json.load(f), os.path.join(figdir, "training_curve.png"))

    with open(os.path.join(figdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n=== 음절 복원 예시 (정답 / greedy / +automata) ===")
    for gt, b, af in rec["examples"]:
        print(f"  GT     : {gt}")
        print(f"  greedy : {b}")
        print(f"  +auto  : {af}\n")
    print(f"[figures] -> {figdir}")


if __name__ == "__main__":
    main()
