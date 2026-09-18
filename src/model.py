# -*- coding: utf-8 -*-
"""
Classification model.
Used by: train.py, evaluate.py, decode.py.
Design: a CNN taking log-mel (1,M,T) as an image. Default SmallCNN trains on CPU.
        arch='coatnet_lite' adds an attention block on top of conv (optional).
Targets: 33 base jamo (number of classes determined from the data).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.net(x))


class SmallCNN(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class _SelfAttn2d(nn.Module):
    """Lightweight MHSA applied over flattened spatial tokens (for coatnet_lite)."""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)     # (b, h*w, c)
        tn = self.norm(t)
        a, _ = self.attn(tn, tn, tn)
        t = t + a
        return t.transpose(1, 2).reshape(b, c, h, w)


class CoAtNetLite(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super().__init__()
        self.stem = ConvBlock(1, 32)
        self.c2 = ConvBlock(32, 64)
        self.c3 = ConvBlock(64, 128)
        self.attn = _SelfAttn2d(128, heads=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.attn(x)
        return self.head(self.pool(x))


def build_model(cfg, n_classes):
    arch = cfg.model.arch
    if arch == "coatnet_lite":
        return CoAtNetLite(n_classes, cfg.model.dropout)
    return SmallCNN(n_classes, cfg.model.dropout)
