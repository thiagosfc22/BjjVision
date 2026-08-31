"""Gi recolouring, to stop colour from predicting who is on the bottom.

In this match `A` is the blue gi, `A` is Ribeiro, and Ribeiro is underneath in
78% of frames. Those three facts are perfectly confounded, so any model trained
on pixels learns "blue is the bottom one" and carries that into the first match
where the blue gi is on top. The masks are the labels and they do not move, so
we can hand the same labels back with the gi colours changed.

Two modes, and the measurement in `scripts/measure_gi_confound.py` says they are
not equivalent. `swap` exchanges the two real palettes and removes about a third
of the confound: the transferred distribution never quite matches the real one,
so a classifier still separates "blue painted white" from "white". `recolour`
draws from `PALETTE` at random and removes about two thirds, which is as far as
colour alone goes.

What neither touches, by design, is luminance from shadow. The athlete
underneath is genuinely darker, and that is physics rather than a labelling
artifact -- it survives every recolouring at ~64% and needs a second venue to
break, not a filter.
"""
from __future__ import annotations

import cv2
import numpy as np

from .appearance import lab_of, torso_mask

BAND = (0.05, 0.90)

# Plausible competition gi colours as (name, Lab mean, Lab std).
PALETTE: list[tuple[str, np.ndarray, np.ndarray]] = [
    ("branco",  np.float32([155, 126, 131]), np.float32([46, 4, 8])),
    ("cinza",   np.float32([132, 128, 128]), np.float32([40, 4, 6])),
    ("marinho", np.float32([40, 141, 98]),   np.float32([17, 6, 13])),
    ("azul",    np.float32([58, 146, 92]),   np.float32([20, 7, 14])),
    ("preto",   np.float32([32, 128, 128]),  np.float32([14, 4, 5])),
    ("verde",   np.float32([62, 112, 138]),  np.float32([19, 7, 11])),
]


def gi_stats(frames, readers, video_path, rng_seed: int = 0):
    """Lab mean/std of each gi, sampled from torso bands across many frames.

    The torso band is what `appearance.torso_mask` already carves out for the
    colour prototypes: sampling the whole silhouette drags in skin and mat.
    The L tails get trimmed on top of that, because limbs let the mat through.
    """
    want = set(frames)
    acc: dict[str, list] = {"A": [], "B": []}
    cap = cv2.VideoCapture(video_path)
    i, got = -1, 0
    while got < len(want):
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if i not in want:
            continue
        got += 1
        masks = readers[i].get(i) if i in readers else {}
        lab = lab_of(img)
        for fid in ("A", "B"):
            mk = masks.get(fid)
            if mk is None or mk.sum() < 4000:
                continue
            px = lab[torso_mask(mk, BAND).astype(bool)]
            if len(px) < 500:
                continue
            lo, hi = np.percentile(px[:, 0], [20, 80])
            px = px[(px[:, 0] >= lo) & (px[:, 0] <= hi)]
            rng = np.random.default_rng(rng_seed + i)
            acc[fid].append(px[rng.choice(len(px), min(len(px), 1500), replace=False)])
    cap.release()
    return {fid: (np.concatenate(v).astype(np.float32).mean(0),
                  np.concatenate(v).astype(np.float32).std(0) + 1e-6)
            for fid, v in acc.items()}


def skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Skin stays skin. A recoloured face is how this augmentation gets noticed."""
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycc[:, :, 1].astype(np.int16)
    cb = ycc[:, :, 2].astype(np.int16)
    return (cr >= 133) & (cr <= 180) & (cb >= 72) & (cb <= 130)


def dark_mask(lab: np.ndarray) -> np.ndarray:
    """Black belt and deep shadow: no chroma to speak of, and low light."""
    chroma = (np.abs(lab[:, :, 1].astype(np.int16) - 128)
              + np.abs(lab[:, :, 2].astype(np.int16) - 128))
    return (lab[:, :, 0] < 70) & (chroma < 24)


def random_targets(rng: np.random.Generator) -> dict:
    """Two distinguishable gis, assigned to A and B at random.

    Colour still separates the two athletes inside a frame, which is realistic.
    What it stops doing is telling you which of them is on top.
    """
    i, j = rng.choice(len(PALETTE), 2, replace=False)
    out = {}
    for fid, k in (("A", i), ("B", j)):
        name, mu, sd = PALETTE[k]
        out[fid] = (name,
                    mu + rng.normal(0, [6, 2, 2]).astype(np.float32),
                    sd * rng.uniform(0.85, 1.2))
    return out


def recolour(img_bgr: np.ndarray, masks: dict, stats: dict, targets: dict,
             gate: float = 2.4, l_lo: float = 0.7, l_hi: float = 1.6,
             c_lo: float = 0.5, c_hi: float = 2.2) -> np.ndarray:
    """Move each athlete's gi pixels onto a target palette.

    Two things this gets wrong if done naively. The luminance scale has to be
    clamped: the white gi varies three times more in L than the blue one, and
    copying that ratio straight blows every fold into a highlight. And the
    gi test has to be a soft weight rather than a gate -- a hard cut leaves
    shadowed folds in the old colour, which reads as holes punched in the new.
    """
    lab = lab_of(img_bgr).astype(np.float32)
    out = lab.copy()
    protect = skin_mask(img_bgr) | dark_mask(lab)
    for fid in ("A", "B"):
        mk = masks.get(fid)
        if mk is None or not mk.any():
            continue
        mu_s, sd_s = stats[fid]
        _, mu_d, sd_d = targets[fid]
        scale = np.empty(3, np.float32)
        scale[0] = np.clip(sd_d[0] / sd_s[0], l_lo, l_hi)
        scale[1:] = np.clip(sd_d[1:] / sd_s[1:], c_lo, c_hi)
        sel = mk.astype(bool) & (~protect)
        if not sel.any():
            continue
        px = lab[sel]
        z = (px - mu_s) / sd_s
        w = np.exp(-0.5 * (z[:, 1] ** 2 + z[:, 2] ** 2) / gate ** 2)[:, None]
        w = np.where(np.abs(z[:, 0:1]) > 3.6, w * 0.35, w)
        out[sel] = px * (1 - w) + ((px - mu_s) * scale + mu_d) * w
    out[:, :, 0] = np.clip(out[:, :, 0], 0, 255)
    out[:, :, 1:] = np.clip(out[:, :, 1:], 0, 255)
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_Lab2BGR)


def swap(img_bgr: np.ndarray, masks: dict, stats: dict, **kw) -> np.ndarray:
    """The exact counterfactual: each athlete wearing the other's gi."""
    targets = {"A": ("B_real", *stats["B"]), "B": ("A_real", *stats["A"])}
    return recolour(img_bgr, masks, stats, targets, **kw)
