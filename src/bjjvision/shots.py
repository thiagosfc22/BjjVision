"""Shot-boundary detection, and segmenting a broadcast into trackable pieces.

Measured on the IBJJF source: 127 cuts in 777 s, 74 of them inside the match
itself, median in-match shot length 3.0 s. A fixed 240-frame (8 s) SAM2 window
would therefore straddle a cut most of the time, and SAM2's memory attention
assumes temporal continuity -- across a hard cut it happily drags the previous
mask onto an unrelated camera angle and reports high confidence while doing it.

So propagation windows are bounded by *shots*, not by a frame count.

This lands well for the architecture rather than against it: because gi colour is
the identity anchor and not the tracker, a cut is simply another re-anchor point.
The prototypes survive the cut untouched; only the segmenter restarts.

Threshold calibration is deliberately asymmetric. A false positive costs one
extra re-anchor -- a few milliseconds. A false negative costs an entire window of
corrupted masks and a poisoned prototype. So the detector is tuned to be
sensitive rather than precise, which is the opposite of the usual default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Shot:
    start: int                 # inclusive frame index
    end: int                   # exclusive
    kind: str = "unknown"      # "match" | "closeup" | "offmat" | "unknown"
    n_persons_median: float = 0.0
    mat_frac_median: float = 0.0
    flat_frac_median: float = 0.0

    @property
    def length(self) -> int:
        return self.end - self.start

    def seconds(self, fps: float) -> tuple[float, float]:
        return self.start / fps, self.end / fps


def _signature(frame_bgr: np.ndarray) -> np.ndarray:
    """Small hue-saturation histogram: robust to motion blur and exposure ripple,
    sensitive to the wholesale colour-layout change that a camera cut produces."""
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h)


def flat_fraction(frame_bgr: np.ndarray, var_thresh: float = 12.0) -> float:
    """Fraction of the frame that is flat, textureless colour.

    Broadcast transitions in this footage are chroma-green plates carrying small
    video insets. Live action is textured everywhere; a graphics plate is not. So
    "how much of this frame has no local detail" separates them without having to
    hardcode the colour of any particular broadcaster's transition.
    """
    small = cv2.resize(frame_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(grey, (9, 9))
    var = cv2.blur(grey * grey, (9, 9)) - mean * mean
    return float((var < var_thresh).mean())


def detect_cuts(video_path: Path, z_threshold: float = 150.0,
                min_shot_frames: int = 10,
                progress: bool = False) -> tuple[list[int], np.ndarray, float]:
    """Return (cut frame indices, per-frame dissimilarity, fps).

    The threshold is adaptive: a static wide shot and a hand-held tracking shot
    have very different baseline inter-frame motion, so a fixed cutoff either
    misses cuts in busy footage or shreds calm footage. Median/MAD normalisation
    measures each frame against the video's own noise floor.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    diffs: list[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sig = _signature(frame)
        diffs.append(0.0 if prev is None else 1.0 - cv2.compareHist(prev, sig, cv2.HISTCMP_CORREL))
        prev = sig
        if progress and len(diffs) % 3000 == 0:
            print(f"  scanned {len(diffs)} frames", flush=True)
    cap.release()

    d = np.asarray(diffs, dtype=np.float32)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6
    z = (d - med) / (1.4826 * mad)

    raw = np.flatnonzero(z > z_threshold).tolist()
    cuts: list[int] = []
    for c in raw:
        # a dissolve trips several consecutive frames; keep only its start
        if cuts and c - cuts[-1] < min_shot_frames:
            continue
        cuts.append(c)
    return cuts, d, fps


def build_shots(n_frames: int, cuts: list[int]) -> list[Shot]:
    bounds = [0] + [c for c in cuts if 0 < c < n_frames] + [n_frames]
    return [Shot(start=a, end=b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def classify_shots(video_path: Path, shots: list[Shot], detector=None, mat_model=None,
                   samples_per_shot: int = 3, min_box_area_frac: float = 0.004,
                   flat_max: float = 0.55) -> list[Shot]:
    """Label each shot so the pipeline can skip what it cannot track.

    A close-up of one athlete's face contains one competitor, not two; a podium or
    crowd shot contains neither. Running the two-fighter tracker on those produces
    confident nonsense. Better to detect them and say so on screen than to invent
    a second athlete out of a cameraman's shoulder.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        for sh in shots:
            idxs = np.linspace(sh.start, sh.end - 1,
                               min(samples_per_shot, sh.length), dtype=int)
            counts, mat_fracs, flats = [], [], []
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    continue
                flats.append(flat_fraction(frame))
                mat_fracs.append(float(mat_model.mask(frame).mean()) if mat_model else 0.0)
                if detector is not None:
                    fa = float(frame.shape[0] * frame.shape[1])
                    persons = detector.detect(frame, persist=False)
                    counts.append(len([p for p in persons if p.area / fa >= min_box_area_frac]))

            sh.n_persons_median = float(np.median(counts)) if counts else -1.0
            sh.mat_frac_median = float(np.median(mat_fracs)) if mat_fracs else 0.0
            sh.flat_frac_median = float(np.median(flats)) if flats else 0.0

            if sh.flat_frac_median > flat_max:
                sh.kind = "graphic"         # transition plate / inset composite
            elif sh.mat_frac_median < 0.08:
                sh.kind = "offmat"          # podium, crowd, sponsor board
            elif sh.n_persons_median < 0:
                sh.kind = "mat"             # CPU-only scout: mat visible, count unknown
            elif sh.n_persons_median >= 2:
                sh.kind = "match"
            elif sh.n_persons_median == 1:
                sh.kind = "closeup"         # one athlete filling the frame
            else:
                sh.kind = "offmat"
    finally:
        cap.release()
    return shots


def windows(shots: list[Shot], max_window: int,
            kinds: tuple[str, ...] = ("match",)) -> list[tuple[int, int, bool]]:
    """Propagation windows as (start, end, starts_new_shot).

    Two different boundaries, with different consequences:
      * a window boundary INSIDE a shot -- the camera is continuous, so the
        previous masks are still meaningful and we re-seed from them.
      * a SHOT boundary -- the geometry is unrelated, so the previous masks are
        worthless and we re-detect from scratch. The colour prototypes survive
        and bind the fresh detections back to A and B, which is the whole reason
        a cut is survivable at all.
    """
    out: list[tuple[int, int, bool]] = []
    for sh in shots:
        if sh.kind not in kinds:
            continue
        s, first = sh.start, True
        while s < sh.end:
            e = min(s + max_window, sh.end)
            out.append((s, e, first))
            s, first = e, False
    return out


def summarise(shots: list[Shot], fps: float) -> dict:
    by_kind: dict[str, dict] = {}
    for sh in shots:
        k = by_kind.setdefault(sh.kind, {"shots": 0, "frames": 0})
        k["shots"] += 1
        k["frames"] += sh.length
    for k, v in by_kind.items():
        v["seconds"] = round(v["frames"] / fps, 1)
    lens = np.array([s.length for s in shots], dtype=np.float32) / fps
    return {
        "n_shots": len(shots),
        "median_shot_s": round(float(np.median(lens)), 2) if len(lens) else 0.0,
        "shortest_shot_s": round(float(lens.min()), 2) if len(lens) else 0.0,
        "longest_shot_s": round(float(lens.max()), 2) if len(lens) else 0.0,
        "by_kind": by_kind,
    }
