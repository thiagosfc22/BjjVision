"""Synthetic grappling scene used to validate the colour anchor without a GPU.

Deliberately adversarial: the two athletes spend most of the clip overlapping,
a referee circles them in a contrasting uniform, and a crowd fills the top of the
frame. The point is to reproduce the geometry that breaks box-based Re-ID.
"""
from __future__ import annotations

import cv2
import numpy as np

W, H = 640, 360
GI_A = (238, 236, 232)     # white gi        (BGR)
GI_B = (150, 70, 30)       # navy blue gi
REF = (40, 190, 235)       # referee: yellow shirt
MAT = (150, 105, 60)       # blue mat
SKIN = (150, 175, 205)


def _person(canvas, mask, cx, cy, w, h, colour, skin_head=True):
    cv2.ellipse(canvas, (int(cx), int(cy)), (int(w), int(h)), 0, 0, 360, colour, -1)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(w), int(h)), 0, 0, 360, 1, -1)
    if skin_head:
        cv2.circle(canvas, (int(cx), int(cy - h * 0.85)), int(w * 0.42), SKIN, -1)
        cv2.circle(mask, (int(cx), int(cy - h * 0.85)), int(w * 0.42), 1, -1)


def make_frame(t: float, overlap: float = 0.0):
    """overlap in [0,1]: 0 = standing apart, 1 = fully entangled."""
    img = np.full((H, W, 3), MAT, np.uint8)
    cv2.rectangle(img, (0, 0), (W, 96), (58, 58, 66), -1)          # stands
    rng = np.random.default_rng(7)
    for i in range(26):                                            # crowd
        x, y = int(rng.uniform(8, W - 8)), int(rng.uniform(16, 88))
        c = tuple(int(v) for v in rng.integers(40, 210, 3))
        cv2.circle(img, (x, y), int(rng.uniform(5, 9)), c, -1)
    cv2.ellipse(img, (W // 2, 250), (285, 118), 0, 0, 360, (168, 122, 74), -1)  # mat

    gap = (1.0 - overlap) * 110 + 24
    ax = W / 2 - gap / 2 + 16 * np.sin(t * 1.7)
    bx = W / 2 + gap / 2 + 16 * np.cos(t * 1.3)
    cy = 250 + 10 * np.sin(t * 0.9)

    ma = np.zeros((H, W), np.uint8)
    mb = np.zeros((H, W), np.uint8)
    # B drawn first so A occludes it -- mirrors a real top-position athlete
    _person(img, mb, bx, cy, 44, 58, GI_B)
    _person(img, ma, ax, cy, 44, 58, GI_A)
    ma_b = ma.astype(bool)
    mb_b = mb.astype(bool) & ~ma_b          # occlusion: A wins contested pixels

    mref = np.zeros((H, W), np.uint8)
    rx = W / 2 + 190 * np.cos(t * 0.55)
    _person(img, mref, rx, 232, 26, 62, REF)

    return img, {"A": ma_b, "B": mb_b}, mref.astype(bool)
