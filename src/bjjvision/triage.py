"""Triage: where does the current student already work on a match it never saw?

This is the front gate of the per-match loop. Before any teacher run is paid
for, the student is run over sampled windows of every trackable shot and each
shot is scored on signals a wrong segmentation cannot fake all at once:

  stability   consecutive-frame mask IoU. The one number that separated the
              healthy student from garbage on galvao-xande (0.93 vs collapse).
  flip_rate   A/B identity exchanged between consecutive frames.
  empty_rate  frames where an athlete class all but vanished.
  margin      mean softmax top1-top2 gap on athlete pixels -- how sure the
              student is of the pixels it did claim.
  frag        connected components per athlete mask. A body is 1, maybe 2
              under occlusion; a crowd hallucination is confetti.
  stray_frac  fraction of each class's pixels OUTSIDE its largest component.
              Separates a stray banner blob (core mask fine, cleanup enough)
              from structural fragmentation (the mask itself is confetti).
  dist        centroid distance A-B as a fraction of the frame diagonal.
              Grapplers touch; the dalpra failure mode (two seated
              photographers) lives far apart.
  area        athlete pixels as a fraction of the frame.
  border      fraction of the frame's border pixels claimed by an athlete
              class. On wide camera work (2009 multi-mat, Pan 2019) the
              student painted the MAT as athlete B -- and scored stability
              0.91, because a mat does not move. A body rarely touches much
              of the frame edge; a mat-fill hugs it, so border is the one
              cheap signal that catastrophe cannot fake.

Numbers alone do not decide anything -- the dalpra photographers scored
confidence 1.00 on a pipeline that trusted numbers. Every shot also gets an
evidence grid (frames with the student's overlay) for a pair of eyes: the VLM
supervisor when enabled, a human otherwise. Signals sort the queue; eyes rule.

Thresholds below are STARTING points chosen before calibration. `flags` names
what looks off; the verdict on whether a flagged shot is actually broken
belongs to whoever looks at the evidence grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .student import UNetStudent, normalise

SIZE = (320, 180)                 # what the student was trained on
MIN_CLASS_PX = 30                 # below this, at 320x180, the class is "empty"
MIN_COMP_PX = 15                  # ignore speckle when counting components

# Starting thresholds -- see module docstring. Recorded in every report so a
# dataset built on top of a triage run is reproducible from the report alone.
THRESHOLDS = {
    "stability_min": 0.70,
    "flip_rate_max": 0.02,
    "empty_rate_max": 0.10,
    "margin_min": 0.60,
    "frag_max": 2.5,
    "dist_max": 0.35,
    "area_min": 0.005,
    "area_max": 0.40,
    # Measured on buchecha-lo, the first unseen match: 47/54 shots tripped
    # frag_max, but the evidence grids showed the athletes' own masks were
    # mostly fine -- the extra components were hallucinated blobs on banner
    # graphics and mat logos. The decisive number is CORE stability, not
    # stray mass: buchecha's core_stability median is 0.887 against 0.894 on
    # the healthy galvao reference, while stray_frac runs to 0.20 (p90) even
    # on the reference. So the cleanup gate is "the largest component is a
    # stable object"; stray_max is only a sanity bound -- past half the mask,
    # 'largest component' stops meaning 'the athlete'.
    "stray_max": 0.50,
    # mat_fill: gracie-calasans put athlete B on the MAT across a whole
    # single-camera match and every motion-based gate passed (stability 0.91
    # -- mats hold still). Measured across all eight triaged matches: the two
    # catastrophes sit at border 0.271 / 0.305 with area 0.35+, while healthy
    # shots top out at border ~0.20 (galvao median 0.005) even in legitimate
    # close-ups whose area hits 0.46.
    "matfill_area_min": 0.25,
    "matfill_border_min": 0.25,
}


def load_student(ckpt_path: str | Path, device: torch.device) -> UNetStudent:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = UNetStudent(ck["width"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


# One sampling window per this many frames of shot. Calibrated by failure:
# 2009 single-camera footage (gracie-bastos) has 2-minute shots, and a flat
# 3 runs sampled 2% of an 8-minute match -- a verdict on footage barely seen.
RUN_DENSITY_FRAMES = 360          # ~12 s at 30 fps
MAX_RUNS_PER_SHOT = 10


def sample_runs(start: int, end: int, n_runs: int, run_len: int) -> list[tuple[int, int]]:
    """Runs of consecutive frames spread across a shot, more for longer shots.

    Consecutive frames, not a stride: stability and flip_rate are pair
    statistics and only mean something on truly adjacent frames. `n_runs` is
    the floor; density adds runs up to MAX_RUNS_PER_SHOT so a two-minute
    locked-off shot is not judged on the same three windows as an 8-second
    broadcast cut.
    """
    length = end - start
    if length < run_len:
        return [(start, length)] if length >= 2 else []
    n = min(max(n_runs, length // RUN_DENSITY_FRAMES), MAX_RUNS_PER_SHOT)
    n = min(n, max(1, length // run_len))
    anchors = np.linspace(start, end - run_len, n).astype(int)
    return [(int(a), run_len) for a in anchors]


def read_run(cap: cv2.VideoCapture, start: int, length: int) -> tuple[np.ndarray, list[int]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, ids = [], []
    for i in range(length):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA))
        ids.append(start + i)
    if not frames:
        return np.zeros((0, SIZE[1], SIZE[0], 3), np.uint8), []
    return np.stack(frames), ids


@torch.no_grad()
def infer(model: UNetStudent, imgs: np.ndarray, device: torch.device,
          batch: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """-> (pred plane uint8 [N,H,W], per-frame mean softmax margin on athlete px)."""
    preds, margins = [], []
    for s in range(0, len(imgs), batch):
        x = normalise(imgs[s:s + batch], device)
        p = F.softmax(model(x), 1)
        top2 = p.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1])
        pred = p.argmax(1)
        fg = pred != 0
        m = torch.where(fg.flatten(1).any(1),
                        (margin * fg).flatten(1).sum(1) / fg.flatten(1).sum(1).clamp(min=1),
                        torch.zeros(len(pred), device=device))
        preds.append(pred.cpu().numpy().astype(np.uint8))
        margins.append(m.cpu().numpy())
    return np.concatenate(preds), np.concatenate(margins)


def _components(mask: np.ndarray) -> tuple[int, np.ndarray, float]:
    """-> (component count, largest-component mask, stray pixel fraction)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 0, np.zeros_like(mask), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(areas.argmax())
    core = lab == big
    total = int(areas.sum())
    stray = 0.0 if total == 0 else 1.0 - float(areas.max()) / total
    return int((areas >= MIN_COMP_PX).sum()), core, stray


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


@dataclass
class ShotSignals:
    shot_id: int
    start: int
    end: int
    kind: str
    n_frames: int = 0
    stability: float = np.nan
    core_stability: float = np.nan
    flip_rate: float = np.nan
    empty_rate: float = np.nan
    margin: float = np.nan
    frag: float = np.nan
    stray_frac: float = np.nan
    dist: float = np.nan
    area: float = np.nan
    border: float = np.nan
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4) if np.isfinite(v) else None
        return d


def measure_shot(model: UNetStudent, cap: cv2.VideoCapture, device: torch.device,
                 shot_id: int, start: int, end: int, kind: str,
                 n_runs: int = 3, run_len: int = 24
                 ) -> tuple[ShotSignals, list[np.ndarray], list[np.ndarray], list[int]]:
    """-> (signals, sampled imgs, preds, frame ids) for the evidence grid."""
    sig = ShotSignals(shot_id, start, end, kind)
    all_imgs, all_preds, all_ids = [], [], []
    stab, core_stab, flips = [], [], []
    empt, margs, frags, strays, dists, areas, borders = [], [], [], [], [], [], []
    npx = SIZE[0] * SIZE[1]
    diag = float(np.hypot(*SIZE))
    n_border_px = 2 * (SIZE[0] + SIZE[1]) - 4

    for rs, rl in sample_runs(start, end, n_runs, run_len):
        imgs, ids = read_run(cap, rs, rl)
        if len(imgs) < 2:
            continue
        preds, margin = infer(model, imgs, device)
        all_imgs.append(imgs); all_preds.append(preds); all_ids.extend(ids)
        margs.extend(margin.tolist())
        prev = None                              # (a, b, a_core, b_core)
        for t in range(len(preds)):
            a, b = preds[t] == 1, preds[t] == 2
            sa, sb = int(a.sum()), int(b.sum())
            empt.append(sa < MIN_CLASS_PX or sb < MIN_CLASS_PX)
            areas.append((sa + sb) / npx)
            fg = a | b
            borders.append((int(fg[0].sum()) + int(fg[-1].sum())
                            + int(fg[1:-1, 0].sum()) + int(fg[1:-1, -1].sum()))
                           / n_border_px)
            na, a_core, stray_a = _components(a)
            nb, b_core, stray_b = _components(b)
            if sa >= MIN_CLASS_PX and sb >= MIN_CLASS_PX:
                frags.append(0.5 * (na + nb))
                strays.append(0.5 * (stray_a + stray_b))
                # Centroid of the CORE component: a banner blob must not drag
                # the athlete's centre of mass into the crowd.
                ya, xa = np.nonzero(a_core); yb, xb = np.nonzero(b_core)
                dists.append(np.hypot(xa.mean() - xb.mean(), ya.mean() - yb.mean()) / diag)
            if prev is not None:
                pa, pb, pa_core, pb_core = prev
                direct = 0.5 * (_iou(a, pa) + _iou(b, pb))
                cross = 0.5 * (_iou(a, pb) + _iou(b, pa))
                stab.append(direct)
                core_stab.append(0.5 * (_iou(a_core, pa_core) + _iou(b_core, pb_core)))
                flips.append(cross > direct)
            prev = (a, b, a_core, b_core)

    if not stab:
        return sig, all_imgs, all_preds, all_ids
    sig.n_frames = len(margs)
    sig.stability = float(np.mean(stab))
    sig.core_stability = float(np.mean(core_stab))
    sig.flip_rate = float(np.mean(flips))
    sig.empty_rate = float(np.mean(empt))
    sig.margin = float(np.mean(margs))
    sig.frag = float(np.mean(frags)) if frags else np.nan
    sig.stray_frac = float(np.mean(strays)) if strays else np.nan
    sig.dist = float(np.median(dists)) if dists else np.nan
    sig.area = float(np.median(areas))
    sig.border = float(np.mean(borders))

    t = THRESHOLDS
    if sig.area > t["matfill_area_min"] and sig.border > t["matfill_border_min"]:
        sig.flags.append("mat_fill")
    if sig.stability < t["stability_min"]:
        sig.flags.append("unstable")
    if sig.flip_rate > t["flip_rate_max"]:
        sig.flags.append("id_flips")
    if sig.empty_rate > t["empty_rate_max"]:
        sig.flags.append("athlete_missing")
    if sig.margin < t["margin_min"]:
        sig.flags.append("low_margin")
    if np.isfinite(sig.frag) and sig.frag > t["frag_max"]:
        # Cosmetic vs structural: a stable core is a cleanup job, not a
        # broken shot -- see the stray_max note in THRESHOLDS.
        if np.isfinite(sig.stray_frac) and sig.stray_frac <= t["stray_max"] \
                and sig.core_stability >= t["stability_min"]:
            sig.flags.append("stray_blobs")
        else:
            sig.flags.append("fragmented")
    if np.isfinite(sig.dist) and sig.dist > t["dist_max"]:
        sig.flags.append("far_apart")
    if sig.area < t["area_min"]:
        sig.flags.append("area_tiny")
    if sig.area > t["area_max"]:
        sig.flags.append("area_huge")
    return sig, all_imgs, all_preds, all_ids


# -- evidence -------------------------------------------------------------
_COL = {1: (0, 0, 255), 2: (0, 255, 0)}          # A red, B green (BGR)


def paint(img: np.ndarray, plane: np.ndarray) -> np.ndarray:
    out = img.copy()
    for cid, col in _COL.items():
        m = plane == cid
        if not m.any():
            continue
        out[m] = (0.45 * np.array(col) + 0.55 * out[m]).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, 1)
    return out


def evidence_grid(imgs: list[np.ndarray], preds: list[np.ndarray], ids: list[int],
                  sig: ShotSignals, n_tiles: int = 6, scale: int = 2,
                  cols: int = 2) -> np.ndarray:
    """Overlay tiles at the student's own resolution, upscaled for reading.

    Rendering exactly what the student saw (320x180) keeps the evidence honest:
    a judge should not grade the mask against detail the model never had.
    """
    I = np.concatenate(imgs); P = np.concatenate(preds)
    pick = np.linspace(0, len(I) - 1, min(n_tiles, len(I))).astype(int)
    tiles = []
    for i in pick:
        t = paint(I[i], P[i])
        t = cv2.resize(t, (SIZE[0] * scale, SIZE[1] * scale),
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"f{ids[i]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    grid = np.vstack(rows)

    hdr = np.zeros((34, grid.shape[1], 3), np.uint8)
    txt = (f"shot {sig.shot_id}  f{sig.start}-{sig.end}  "
           f"stab {sig.stability:.2f}/{sig.core_stability:.2f}core  "
           f"flip {sig.flip_rate:.3f}  empty {sig.empty_rate:.2f}  "
           f"marg {sig.margin:.2f}  frag {sig.frag:.1f}  "
           f"stray {sig.stray_frac:.2f}  dist {sig.dist:.2f}  "
           f"area {sig.area:.3f}  bord {sig.border:.2f}")
    cv2.putText(hdr, txt, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hdr, "flags: " + (",".join(sig.flags) or "none"), (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 200, 255) if sig.flags else (0, 255, 0), 1, cv2.LINE_AA)
    return np.vstack([hdr, grid])
