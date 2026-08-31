"""Pseudo-labels: turn the student's own masks into training data -- gated.

    python scripts/make_pseudolabels.py buchecha-lo

The teacher costs a GPU rental per match; the student runs at 250 fps on this
laptop. On shots the triage cleared (verdict ok/cleanup, with the evidence
grids inspected), the student's masks are good enough to BE labels -- but only
behind safeguards, because training a model on its own raw output launders its
mistakes into the next generation with nothing left to catch them:

  1. The shot gate. Only shots the triage cleared enter at all; review and
     skip shots are teacher work. The triage verdicts were validated by eye
     against the failure modes numbers missed (mat-as-athlete scored
     stability 0.91).
  2. Largest-component cleanup. Deterministic geometry, not model output:
     one athlete is one connected body, so stray banner blobs are deleted
     rather than learned. Applied before every other per-frame gate so the
     gates judge the mask that would actually be trained on.
  3. Per-frame gates: both athletes present, confident (softmax margin),
     off the frame border (the mat_fill tell), temporally consistent with
     the neighbouring frame, and -- the one gate the student cannot vote on
     -- the COLOUR ARBITER: fighter A wears the darker/bluer gi in every
     fight in this campaign, so a frame where mask A is not bluer than mask
     B has the identity wrong, whatever the model thinks. Same arbiter that
     caught the teacher's broken labels in clean_labels.py.

Frames that fail any gate are DROPPED, never corrected. Output is
StudentSet-compatible (img.npy / lab.npy / manifest.json); `occlusion` is
zeros because the teacher features that measured it do not exist here, which
the manifest records.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.student import UNetStudent  # noqa: E402
from bjjvision.triage import SIZE, infer, load_student  # noqa: E402

GATES = {
    "min_px": 30,          # per class, after cleanup, at 320x180
    "margin_min": 0.50,    # mean softmax top1-top2 gap on athlete pixels
    "border_max": 0.25,    # athlete pixels on the frame border: mat_fill tell
    "contrast_min": 0.0,   # (blue-red of A) - (blue-red of B) must be positive
    "stab_min": 0.50,      # IoU with the previous candidate frame, same shot
    # The solo gate. Rendering the kept frames by class-area ratio showed two
    # populations below 0.15: solo-athlete framings where the "other athlete"
    # is a belt on the floor (garbage as labels), and deep occlusion where the
    # bottom athlete is genuinely a sliver (the frames this project exists
    # for). Area cannot separate them -- CONTACT can: in occlusion the sliver
    # touches the big mask, the stray belt does not. Ratio below the threshold
    # without contact (5 px dilation) is dropped; with contact it stays.
    "solo_ratio": 0.15,
    "solo_dilate_px": 5,
}
ACCEPT = ("ok", "cleanup", "student_ok")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except Exception:                              # noqa: BLE001
        return "unknown"


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    return lab == (1 + int(stats[1:, cv2.CC_STAT_AREA].argmax()))


def cleaned_plane(pred: np.ndarray) -> np.ndarray:
    plane = np.zeros_like(pred)
    plane[largest_component(pred == 2)] = 2
    plane[largest_component(pred == 1)] = 1      # A wins overlap, as in studentdata
    return plane


def solo_frame(plane: np.ndarray, ratio: float, g: dict) -> bool:
    """True when the smaller class is a detached token, not an occluded athlete."""
    if ratio >= g["solo_ratio"]:
        return False
    a, b = plane == 1, plane == 2
    small, big = (a, b) if a.sum() < b.sum() else (b, a)
    k = np.ones((3, 3), np.uint8)
    grown = cv2.dilate(big.astype(np.uint8), k, iterations=g["solo_dilate_px"])
    return not bool((grown.astype(bool) & small).any())


def frame_metrics(img: np.ndarray, plane: np.ndarray, prev: np.ndarray | None,
                  n_border_px: int) -> dict:
    a, b = plane == 1, plane == 2
    sa, sb = int(a.sum()), int(b.sum())
    fg = a | b
    border = (int(fg[0].sum()) + int(fg[-1].sum())
              + int(fg[1:-1, 0].sum()) + int(fg[1:-1, -1].sum())) / n_border_px
    contrast = np.nan
    if sa >= 50 and sb >= 50:
        f = img.astype(np.float32)
        contrast = float((f[a][:, 0].mean() - f[a][:, 2].mean())
                         - (f[b][:, 0].mean() - f[b][:, 2].mean()))
    stab = np.nan
    if prev is not None:
        pa, pb = prev == 1, prev == 2
        ua, ub = (a | pa).sum(), (b | pb).sum()
        ia = (a & pa).sum() / ua if ua else 1.0
        ib = (b & pb).sum() / ub if ub else 1.0
        stab = 0.5 * (ia + ib)
    return {"pxA": sa, "pxB": sb, "border": border, "contrast": contrast, "stab": stab}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v2/student.pt")
    ap.add_argument("--stride", type=int, default=2,
                    help="frames 0.03s apart are near-duplicates; 2 halves the "
                         "disk for almost no information loss")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    video = root / "data" / "interim" / f"{a.slug}_norm.mp4"
    report_p = root / "data" / "out" / f"{a.slug}_triage" / "report.json"
    if not report_p.exists():
        raise SystemExit(f"sem triagem para {a.slug} -- rode o loop antes")
    out = Path(a.out or root / "data" / "out" / f"pseudo_{a.slug}_320")
    out.mkdir(parents=True, exist_ok=True)

    from bjjvision.triage import apply_overrides
    report = json.loads(report_p.read_text())
    shots = [r for r in apply_overrides(report["shots"], report_p.parent)
             if r.get("verdict") in ACCEPT]
    if not shots:
        raise SystemExit(f"{a.slug}: nenhum shot aprovado na triagem")

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_student(a.ckpt, dev)
    W, H = SIZE
    n_border_px = 2 * (W + H) - 4
    t0 = time.time()

    # -- pass 1: infer everything, gate on metrics, keep labels as PNG ------
    cap = cv2.VideoCapture(str(video))
    frames_kept: list[int] = []
    shot_of: list[int] = []
    lab_png: list[bytes] = []
    drops = {k: 0 for k in ("empty", "margin", "border", "contrast", "stab", "solo")}
    n_cand = 0
    for r in shots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["start"])
        prev_plane = None
        for base in range(r["start"], r["end"], 32 * a.stride):
            batch_ids, batch_imgs = [], []
            for i in range(base, min(base + 32 * a.stride, r["end"])):
                ok, fr = cap.read()
                if not ok:
                    break
                if (i - r["start"]) % a.stride:
                    continue
                batch_ids.append(i)
                batch_imgs.append(cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA))
            if not batch_imgs:
                break
            preds, margins = infer(model, np.stack(batch_imgs), dev)
            for j, fi in enumerate(batch_ids):
                n_cand += 1
                plane = cleaned_plane(preds[j])
                m = frame_metrics(batch_imgs[j], plane, prev_plane, n_border_px)
                prev_plane = plane
                g = GATES
                if m["pxA"] < g["min_px"] or m["pxB"] < g["min_px"]:
                    drops["empty"] += 1; continue
                if margins[j] < g["margin_min"]:
                    drops["margin"] += 1; continue
                if m["border"] > g["border_max"]:
                    drops["border"] += 1; continue
                if not np.isnan(m["contrast"]) and m["contrast"] <= g["contrast_min"]:
                    drops["contrast"] += 1; continue
                if not np.isnan(m["stab"]) and m["stab"] < g["stab_min"]:
                    drops["stab"] += 1; continue
                ratio = min(m["pxA"], m["pxB"]) / max(m["pxA"], m["pxB"], 1)
                if solo_frame(plane, ratio, g):
                    drops["solo"] += 1; continue
                frames_kept.append(fi)
                shot_of.append(r["shot_id"])
                okp, buf = cv2.imencode(".png", plane)
                if not okp:
                    raise RuntimeError("PNG encode failed")
                lab_png.append(buf.tobytes())
    cap.release()
    n = len(frames_kept)
    print(f"{a.slug}: {n_cand} candidatos -> {n} mantidos "
          f"({100 * n / max(n_cand, 1):.1f}%)  drops: "
          + "  ".join(f"{k} {v}" for k, v in drops.items()))

    # -- pass 2: decode once more, write only the keepers -------------------
    img = np.lib.format.open_memmap(out / "img.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, H, W, 3))
    lab = np.lib.format.open_memmap(out / "lab.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, H, W))
    wanted = {f: k for k, f in enumerate(frames_kept)}
    cap = cv2.VideoCapture(str(video))
    idx, written = -1, 0
    while written < n:
        ok, fr = cap.read()
        if not ok:
            break
        idx += 1
        k = wanted.get(idx)
        if k is None:
            continue
        img[k] = cv2.resize(fr, SIZE, interpolation=cv2.INTER_AREA)
        lab[k] = cv2.imdecode(np.frombuffer(lab_png[k], np.uint8), cv2.IMREAD_GRAYSCALE)
        written += 1
    cap.release()
    img.flush(); lab.flush()

    manifest = {
        "source": "student-pseudo",
        "video": str(video), "triage": str(report_p), "ckpt": a.ckpt,
        "git_sha": git_sha(), "size": [W, H], "stride": a.stride,
        "n": int(written), "gates": GATES, "drops": drops,
        "candidates": n_cand,
        "frames": [int(f) for f in frames_kept],
        "shots": [int(s) for s in shot_of],
        # No teacher features exist here, so no occlusion measurement; zeros
        # keep StudentSet compatibility and are recorded as absent.
        "occlusion": [0.0] * written,
        "occlusion_available": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"-> {out}  ({written} frames, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
