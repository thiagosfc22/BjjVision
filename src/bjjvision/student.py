"""The student: one frame in, two athlete masks out, no SAM2 and no colour pipeline.

The teacher costs ~77 min of rented GPU per match and only works because of a
per-shot mat model, a colour seed that excludes the mat and object-scale
selection. This is the attempt to compress all of that into weights.

**Why three classes and not two binary heads.** Identity matters as much as the
mask, so the output is a softmax over {background, A, B} and identity is simply
which channel fired. That is deliberately the *cheap* answer, and it has a
failure mode worth naming before any number is read: A is the blue gi in all
three available matches, so a 3-class softmax is free to learn "blue channel"
instead of "the athlete underneath". `metrics` reports assigned and
best-permutation IoU separately precisely so that failure is visible rather
than averaged away -- the gap between them IS the identity error.

**Why a plain U-Net.** Not because it is the best architecture but because it
is the one that produces a number today. The encoder is four stride-2 blocks on
a 320x180 input, which is enough: the measured ceiling from downsampling a
ground-truth mask to that resolution and back is 0.973 IoU, so resolution is
not what will limit this.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _block(cin: int, cout: int) -> nn.Sequential:
    # GroupNorm rather than BatchNorm: batches here are small and highly
    # correlated (consecutive frames of one shot), which is the case where
    # BatchNorm's running statistics are worst.
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
    )


class UNetStudent(nn.Module):
    def __init__(self, width: int = 32, n_classes: int = 3):
        super().__init__()
        w = width
        self.e1 = _block(3, w)
        self.e2 = _block(w, w * 2)
        self.e3 = _block(w * 2, w * 4)
        self.e4 = _block(w * 4, w * 8)
        self.bott = _block(w * 8, w * 8)
        self.d4 = _block(w * 16, w * 4)
        self.d3 = _block(w * 8, w * 2)
        self.d2 = _block(w * 4, w)
        self.d1 = _block(w * 2, w)
        self.head = nn.Conv2d(w, n_classes, 1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b = self.bott(self.pool(s4))
        def up(t, ref):
            return F.interpolate(t, size=ref.shape[-2:], mode="nearest")
        d = self.d4(torch.cat([up(b, s4), s4], 1))
        d = self.d3(torch.cat([up(d, s3), s3], 1))
        d = self.d2(torch.cat([up(d, s2), s2], 1))
        d = self.d1(torch.cat([up(d, s1), s1], 1))
        return self.head(d)


def normalise(img_u8: np.ndarray, device) -> torch.Tensor:
    """BGR uint8 NHWC -> normalised RGB float NCHW."""
    x = torch.from_numpy(np.ascontiguousarray(img_u8[..., ::-1])).to(device)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN.to(device)) / STD.to(device)


@torch.no_grad()
def metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-frame IoU, reported both as assigned and as best-permutation.

    `assigned` grades the student on the question it was asked -- is this pixel
    fighter A. `best` grades it on the easier question of whether it found two
    separate athletes at all, by taking the better of the two labellings. Their
    difference is the price of identity, and on this data that price is the
    whole reason the project is not already solved.
    """
    pred = logits.argmax(1)
    out = {}
    def iou(p, t):
        inter = (p & t).flatten(1).sum(1).float()
        union = (p | t).flatten(1).sum(1).float()
        # A frame where neither prediction nor target has the class scores 1:
        # the student correctly found nothing.
        return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    a_t, b_t = target == 1, target == 2
    a_p, b_p = pred == 1, pred == 2
    direct = 0.5 * (iou(a_p, a_t) + iou(b_p, b_t))
    swapped = 0.5 * (iou(b_p, a_t) + iou(a_p, b_t))
    out["assigned"] = direct.cpu().numpy()
    out["best"] = torch.maximum(direct, swapped).cpu().numpy()
    out["flipped"] = (swapped > direct).cpu().numpy()
    out["iou_A"] = iou(a_p, a_t).cpu().numpy()
    out["iou_B"] = iou(b_p, b_t).cpu().numpy()
    # Union-of-both: "did it find the athletes", identity ignored entirely.
    out["iou_union"] = iou(a_p | b_p, a_t | b_t).cpu().numpy()
    return out
