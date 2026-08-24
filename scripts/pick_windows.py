"""Choose the clip windows worth spending rented GPU on.

The instinct is to pick a stretch where the footage looks clean. That is the
wrong pick twice over: a clean stretch is where the pipeline has nothing to prove,
and it makes a boring clip. The interesting window is where the two athletes are
entangled on the ground, because that is where mask separation is hard, where
recalibration fires, and where the recovery is visible.

Proxy, computable on CPU with no detector: separate upright bodies produce two
tall foreground blobs; entangled ground grappling produces ONE wide merged blob.
Score a window by how much of it is merged-and-wide.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bjjvision.roles import MatModel

STRIDE = 10          # sample every Nth frame
WIN_S = 60.0


def frame_score(frame, mat: MatModel):
    """(entangled, wide, area_frac) for one frame."""
    mm = mat.mask(frame)
    if mm.mean() < 0.05:
        return 0.0, 0.0, 0.0
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab).astype(np.float32)
    dist = np.linalg.norm(lab - mat.center_lab.reshape(1, 1, 3), axis=2)
    fg = (dist > 34.0) & mm                        # on the mat, not mat-coloured
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n <= 1:
        return 0.0, 0.0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = np.argsort(areas)[::-1]
    total = float(frame.shape[0] * frame.shape[1])
    a0 = float(areas[big[0]]) / total
    if a0 < 0.02:
        return 0.0, 0.0, a0
    w0 = stats[1 + big[0], cv2.CC_STAT_WIDTH]
    h0 = stats[1 + big[0], cv2.CC_STAT_HEIGHT]
    aspect = w0 / max(h0, 1)
    a1 = float(areas[big[1]]) / total if len(areas) > 1 else 0.0
    merged = 1.0 if a1 < 0.35 * a0 else 0.0        # one dominant blob
    wide = 1.0 if aspect > 1.15 else 0.0           # lying down, not standing
    return merged * wide, aspect, a0


def main(slug: str, top_k: int = 3):
    data = json.loads((ROOT / "data" / "interim" / f"{slug}_shots.json").read_text())
    fps, n_frames = data["fps"], data["n_frames"]
    shots = [s for s in data["shots"] if s["kind"] in ("match", "mat")]
    video = ROOT / "data" / "interim" / f"{slug}_norm.mp4"

    cap = cv2.VideoCapture(str(video))
    probe = []
    for i in np.linspace(0, n_frames - 1, 60, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            probe.append(fr)
    mat = MatModel().fit(probe)

    scores: dict[int, float] = {}
    for sh in shots:
        if sh["end"] - sh["start"] < int(WIN_S * fps * 0.5):
            continue
        for i in range(sh["start"], sh["end"], STRIDE):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                continue
            s, _, _ = frame_score(fr, mat)
            scores[i] = s
    cap.release()

    win = int(WIN_S * fps)
    cands = []
    for sh in shots:
        if sh["end"] - sh["start"] < win:
            continue
        for start in range(sh["start"], sh["end"] - win, STRIDE * 6):
            keys = [k for k in range(start, start + win, STRIDE) if k in scores]
            if len(keys) < win // STRIDE * 0.6:
                continue
            cands.append((float(np.mean([scores[k] for k in keys])), start, start + win,
                          sh["start"], sh["end"]))
    cands.sort(reverse=True)

    picked = []
    for sc, a, b, ss, se in cands:
        if any(a < pb and pa < b for _, pa, pb, _, _ in picked):
            continue                                   # no overlap between picks
        picked.append((sc, a, b, ss, se))
        if len(picked) >= top_k:
            break

    print(f"\n=== {slug} ===  {n_frames} frames @ {fps:.0f}fps, {len(shots)} trackable shots")
    print(f"{'rank':<5}{'frames':<16}{'time':<16}{'ground-grapple':<16}{'shot span (s)'}")
    for r, (sc, a, b, ss, se) in enumerate(picked, 1):
        print(f"{r:<5}{f'{a}:{b}':<16}"
              f"{f'{a/fps//60:.0f}:{a/fps%60:04.1f}-{b/fps//60:.0f}:{b/fps%60:04.1f}':<16}"
              f"{sc:<16.0%}{ss/fps:.0f}-{se/fps:.0f}")
    return picked


if __name__ == "__main__":
    for slug in sys.argv[1:] or ["galvao-xande", "buchecha-lo"]:
        main(slug)
