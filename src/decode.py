# -*- coding: utf-8 -*-
"""
Attack demo: new audio (continuous session recording or clip folder) -> recovered text.
Usage:
  # attack one continuous recording
  python -m src.decode --run runs/exp1 --wav some_session.wav
  # attack a folder of pre-segmented clips (sorted by filename)
  python -m src.decode --run runs/exp1 --clips path/to/clips_dir
Chains the whole attack pipeline (segmentation -> features -> classification ->
automaton-constrained beam search) to demonstrate 'eavesdrop -> text' recovery.
"""
from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config, parse_overrides
from .utils import pick_device
from .dataset import LabelSpace
from .model import build_model
from .features import MelExtractor
from .audio_io import load_audio
from .segment import detect_onsets, cut_clip
from . import hangul as H


def load_run(run_dir, cfg, device):
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    ls = LabelSpace(ckpt["labels"])
    model = build_model(cfg, len(ls)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, ls


@torch.no_grad()
def clips_to_logprobs(clips, cfg, model, device):
    mel = MelExtractor(cfg.feature, cfg.data.sample_rate)
    X = torch.from_numpy(np.stack([mel(c)[0] for c in clips]))[:, None, :, :]
    X = X.squeeze(2) if X.dim() == 5 else X
    logits = model(X.to(device))
    return F.log_softmax(logits, 1).cpu().numpy()


def decode_wav(wav_path, run_dir, cfg, device):
    model, ls = load_run(run_dir, cfg, device)
    wav, sr = load_audio(wav_path, cfg.data.sample_rate)
    onsets = detect_onsets(wav, sr, cfg.segment)
    clips = [cut_clip(wav, sr, o, cfg.segment) for o in onsets]
    if not clips:
        return "", []
    lp = clips_to_logprobs(clips, cfg, model, device)
    labels = [ls.i2l[i] for i in range(len(ls))]
    seq, text, _ = H.constrained_beam_decode(lp, labels, cfg.eval.beam_width)
    return text, seq


def decode_clip_dir(clip_dir, run_dir, cfg, device):
    model, ls = load_run(run_dir, cfg, device)
    files = sorted(glob.glob(os.path.join(clip_dir, "*.wav")) +
                   glob.glob(os.path.join(clip_dir, "*.npy")))
    clips = [load_audio(f, cfg.data.sample_rate)[0] for f in files]
    if not clips:
        return "", []
    lp = clips_to_logprobs(clips, cfg, model, device)
    labels = [ls.i2l[i] for i in range(len(ls))]
    seq, text, _ = H.constrained_beam_decode(lp, labels, cfg.eval.beam_width)
    return text, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default=None)
    ap.add_argument("--wav", default=None, help="continuous session recording wav")
    ap.add_argument("--clips", default=None, help="folder of pre-segmented clips")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    run_dir = a.run or cfg.train.out_dir
    device = pick_device(cfg.train.device)
    if a.wav:
        text, seq = decode_wav(a.wav, run_dir, cfg, device)
    elif a.clips:
        text, seq = decode_clip_dir(a.clips, run_dir, cfg, device)
    else:
        raise SystemExit("specify one of --wav or --clips")
    print("recovered jamo:", " ".join(seq))
    print("recovered text:", text)


if __name__ == "__main__":
    main()
