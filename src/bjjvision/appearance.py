"""Gi colour modelling -- the identity anchor of the whole system.

Why colour and not a Re-ID embedding: in grappling the two athletes' bounding
boxes overlap almost completely, so box-cropped appearance features describe the
*pair*, not either fighter. A SAM2 mask gives a genuinely disjoint pixel set, and
the gi colour is invariant for the entire match. So colour is not one cue among
many -- it is the ground truth that every other cue gets corrected against.

The central object is a normalised CIELAB histogram, used two ways:
  * as a *signature*  -> distance between a mask and a fighter prototype
  * as a *likelihood* -> per-pixel P(fighter | lab), which lets us measure mask
    purity and split a contaminated mask back into two.
Both fall out of the same table, which is why contamination is cheap to detect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

EPS = 1e-8


# --------------------------------------------------------------------------
# Colour signature
# --------------------------------------------------------------------------
@dataclass
class ColorModel:
    """Normalised joint histogram over CIELAB, plus robust summary stats."""
    bins: tuple[int, int, int]
    hist: np.ndarray                 # shape == bins, sums to 1
    mean_lab: np.ndarray             # (3,) median L,a,b -- for human-readable swatch
    n_pixels: int

    @property
    def swatch_bgr(self) -> tuple[int, int, int]:
        lab = np.clip(self.mean_lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _lab_indices(lab: np.ndarray, bins: tuple[int, int, int]) -> np.ndarray:
    """Map Nx3 uint8 Lab pixels -> flat histogram bin index."""
    lab = lab.astype(np.float32)
    idx = np.empty((lab.shape[0], 3), dtype=np.int32)
    for c in range(3):
        idx[:, c] = np.clip((lab[:, c] / 256.0) * bins[c], 0, bins[c] - 1).astype(np.int32)
    return (idx[:, 0] * bins[1] + idx[:, 1]) * bins[2] + idx[:, 2]


def torso_mask(mask: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Restrict a person mask to a vertical band of its bbox.

    Sampling the whole silhouette drags in skin, hair, and mat showing through
    the limbs; the torso band is nearly all gi, which is what we want to measure.
    """
    mu8 = mask.view(np.uint8) if mask.dtype == np.bool_ else mask.astype(np.uint8)
    _, y0, _, bh = cv2.boundingRect(mu8)     # ~1.5 ms cheaper than np.nonzero
    if bh == 0:
        return mask
    y1 = y0 + bh - 1
    h = max(y1 - y0, 1)
    lo = int(y0 + band[0] * h)
    hi = int(y0 + band[1] * h)
    out = np.zeros_like(mask)
    out[lo:hi + 1] = mask[lo:hi + 1]
    return out if out.sum() >= 0.15 * mask.sum() else mask


def lab_of(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR -> CIELAB. Costs ~4 ms at 720p, and the hot path used to pay it six
    times per frame, so callers in a loop should compute it once and pass it in."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)


def build_color_model(frame_bgr: np.ndarray, mask: np.ndarray,
                      bins: tuple[int, int, int] = (8, 12, 12),
                      band: tuple[float, float] | None = (0.15, 0.60),
                      min_pixels: int = 400,
                      lab: np.ndarray | None = None) -> ColorModel | None:
    """Masked CIELAB histogram for one person in one frame."""
    m = mask.astype(bool)
    if band is not None:
        m = torso_mask(m, band).astype(bool)
    n = int(m.sum())
    if n < min_pixels:
        return None

    lab_img = lab_of(frame_bgr) if lab is None else lab
    px = lab_img[m]                                     # (N,3) uint8
    flat = _lab_indices(px, bins)
    hist = np.bincount(flat, minlength=int(np.prod(bins))).astype(np.float32)
    hist /= hist.sum() + EPS

    return ColorModel(bins=bins, hist=hist.reshape(bins),
                      mean_lab=np.median(px, axis=0).astype(np.float32), n_pixels=n)


def hist_distance(a: ColorModel, b: ColorModel) -> float:
    """Hellinger distance in [0,1]. Bin-wise, so it survives multi-tone gis
    (blue jacket + white patches stays a stable bimodal signature)."""
    bc = float(np.sum(np.sqrt(a.hist.ravel() * b.hist.ravel())))
    return float(np.sqrt(max(0.0, 1.0 - bc)))


# --------------------------------------------------------------------------
# Prototype bank -- rolling, anti-drift
# --------------------------------------------------------------------------
@dataclass
class FighterPrototype:
    """The persistent colour identity of one athlete.

    Exemplars only enter the bank on high-confidence frames. That gate is the
    whole anti-drift story: if we admitted every frame, one bad occluded frame
    would poison the prototype and the next frame would look 'consistent' with
    the poison. The prototype must be harder to move than the tracker.
    """
    fighter_id: str                      # "A" / "B"
    label: str = ""                      # human name once known ("Galvao")
    capacity: int = 24
    exemplars: list[ColorModel] = field(default_factory=list)
    _cache: ColorModel | None = None

    def add(self, cm: ColorModel) -> None:
        self.exemplars.append(cm)
        if len(self.exemplars) > self.capacity:
            # keep the first few (the clean calibration frames) as an anchor,
            # rotate the rest -- so late-match drift can never fully take over
            keep_anchor = self.capacity // 4
            self.exemplars = self.exemplars[:keep_anchor] + self.exemplars[-(self.capacity - keep_anchor):]
        self._cache = None

    @property
    def model(self) -> ColorModel:
        """Bank consensus: the mean histogram, weighted by exemplar pixel count."""
        if self._cache is not None:
            return self._cache
        if not self.exemplars:
            raise ValueError(f"prototype {self.fighter_id} is empty")
        w = np.array([e.n_pixels for e in self.exemplars], dtype=np.float32)
        w /= w.sum() + EPS
        hist = np.tensordot(w, np.stack([e.hist for e in self.exemplars]), axes=(0, 0))
        hist /= hist.sum() + EPS
        mean_lab = np.average(np.stack([e.mean_lab for e in self.exemplars]), axis=0, weights=w)
        self._cache = ColorModel(bins=self.exemplars[0].bins, hist=hist.astype(np.float32),
                                 mean_lab=mean_lab.astype(np.float32),
                                 n_pixels=int(sum(e.n_pixels for e in self.exemplars)))
        return self._cache

    def distance(self, cm: ColorModel) -> float:
        return hist_distance(self.model, cm)

    @property
    def ready(self) -> bool:
        return len(self.exemplars) >= 3


# --------------------------------------------------------------------------
# Per-pixel likelihood: purity, contamination, and mask splitting
# --------------------------------------------------------------------------
class PixelClassifier:
    """Turns the two prototypes into a dense per-pixel decision rule.

    Same histograms, read as likelihoods instead of signatures. This is what
    detects a SAM2 mask that has swallowed part of the other athlete, and what
    repairs it without re-running the segmenter.
    """

    def __init__(self, proto_a: FighterPrototype, proto_b: FighterPrototype,
                 background: ColorModel | None = None, smooth: float = 1e-4):
        self.bins = proto_a.model.bins
        self.lut = np.stack([
            proto_a.model.hist.ravel() + smooth,
            proto_b.model.hist.ravel() + smooth,
        ] + ([background.hist.ravel() + smooth] if background is not None else []))
        self.lut /= self.lut.sum(axis=0, keepdims=True)     # -> P(class | bin)
        self.n_classes = self.lut.shape[0]

    def posterior(self, frame_bgr: np.ndarray, mask: np.ndarray,
                  lab: np.ndarray | None = None) -> np.ndarray:
        """(N, n_classes) posterior for every pixel inside `mask`."""
        m = mask.astype(bool)
        if not m.any():
            return np.zeros((0, self.n_classes), dtype=np.float32)
        px = (lab_of(frame_bgr) if lab is None else lab)[m]
        flat = _lab_indices(px, self.bins)
        return self.lut[:, flat].T.astype(np.float32)

    def purity(self, frame_bgr: np.ndarray, mask: np.ndarray, fighter_idx: int,
               lab: np.ndarray | None = None) -> float:
        """Fraction of the mask's pixels that vote for the fighter it claims to be.

        This is the primary health signal. A clean mask sits near 0.85-0.95; a mask
        that has bled into the opponent collapses toward 0.5 long before the mask
        *shape* looks obviously wrong -- which is why it catches swaps early.
        """
        post = self.posterior(frame_bgr, mask, lab)
        if post.shape[0] == 0:
            return 0.0
        return float((np.argmax(post, axis=1) == fighter_idx).mean())

    def split(self, frame_bgr: np.ndarray, mask: np.ndarray,
              min_component_frac: float = 0.12,
              lab: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Repair a contaminated mask by re-assigning its pixels to A and B.

        Morphological opening + largest-component keeps the result anatomically
        plausible instead of a speckled per-pixel scatter.
        """
        m = mask.astype(bool)
        post = self.posterior(frame_bgr, m, lab)
        if post.shape[0] == 0:
            z = np.zeros_like(m)
            return z, z
        lab = np.argmax(post, axis=1)
        out = []
        for cls in (0, 1):
            sel = np.zeros_like(m)
            sel[m] = (lab == cls)
            sel = cv2.morphologyEx(sel.astype(np.uint8), cv2.MORPH_OPEN,
                                   np.ones((5, 5), np.uint8)).astype(bool)
            if sel.sum() < min_component_frac * max(m.sum(), 1):
                sel = np.zeros_like(m)
            out.append(sel)
        return out[0], out[1]

    def sample_prompt_points(self, frame_bgr: np.ndarray, mask: np.ndarray,
                             fighter_idx: int, k: int = 6,
                             rng: np.random.Generator | None = None,
                             lab: np.ndarray | None = None) -> np.ndarray:
        """Pick SAM2 prompt points from *colour-confident* pixels.

        Prompting at a box centre is how you re-seed the same error: in a tangle
        the geometric centre of A's box frequently sits on B's body. Seeding from
        pixels that the colour model is sure about is the difference between a
        re-anchor that heals and one that re-commits.
        """
        rng = rng or np.random.default_rng(0)
        m = mask.astype(bool)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        post = self.posterior(frame_bgr, m, lab)
        conf = post[:, fighter_idx]
        good = conf > max(0.55, float(np.quantile(conf, 0.60)))
        if good.sum() < k:
            good = conf >= np.sort(conf)[-min(k, conf.size)]
        gy, gx = ys[good], xs[good]
        # spread the points spatially so SAM2 sees the whole body, not one blob
        order = rng.permutation(gy.size)
        picked: list[tuple[int, int]] = []
        for i in order:
            p = (int(gx[i]), int(gy[i]))
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 400 for q in picked):
                picked.append(p)
            if len(picked) >= k:
                break
        return np.array(picked or [(int(gx[0]), int(gy[0]))], dtype=np.float32)


def separability(proto_a: FighterPrototype, proto_b: FighterPrototype) -> float:
    """How distinguishable the two gis are. Reported up-front as a feasibility gate.

    Near 1.0 (white vs navy) the colour anchor carries the match on its own.
    Below ~0.35 (two near-white gis) colour cannot arbitrate and the pipeline must
    lean on motion continuity and the LLM supervisor -- better to know at minute 0.
    """
    return hist_distance(proto_a.model, proto_b.model)
