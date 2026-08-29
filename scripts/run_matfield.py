#!/usr/bin/env python
"""Validate mat delimitation + athlete seeding across every mat shot of a match.

Runs the CPU/MPS half only: mat per shot, gi seeding, and SAM2's object-scale
answer at ONE frame per shot. That is the whole bootstrap, and it is the part
that decides whether a paid propagation run is worth starting -- a wrong lock-on
poisons identity for the rest of the match, so it is measured first, here, for
free.

Full video propagation is NOT this script's job: sam2.1_tiny runs at ~1 fps on
MPS, so a 20k-frame match is ~5.7 hours on the laptop. That is the rented box's
work, and only after this reports clean.

    PYTHONPATH=src python scripts/run_matfield.py dalpra-dorsey [--ckpt tiny|large]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bjjvision.matfield import (  # noqa: E402
    choose_scale, fit_mat, gi_votes, mat_confusable_with, seed_point, static_graphics,
)

CKPTS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
}


def grab(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    return f if ok else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--ckpt", default="tiny", choices=sorted(CKPTS))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--sheet", type=Path, default=None, help="write a contact sheet here")
    args = ap.parse_args()

    video = args.data_dir / "interim" / f"{args.slug}_norm.mp4"
    shots_path = args.data_dir / "interim" / f"{args.slug}_shots.json"
    if not video.exists() or not shots_path.exists():
        print(f"missing {video} or {shots_path} -- run `bjj fetch` and `bjj scout` first")
        return 2
    shots = [s for s in json.loads(shots_path.read_text())["shots"] if s["kind"] == "mat"]

    cap = cv2.VideoCapture(str(video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    graphics = static_graphics(
        [f for f in (grab(cap, i) for i in np.linspace(0, n_frames - 1, 40).astype(int))
         if f is not None])
    print(f"{args.slug}: {n_frames} frames, {len(shots)} mat shots, "
          f"burned-in graphics {graphics.mean():.3%} of frame")

    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    cfg, ckpt = CKPTS[args.ckpt]
    predictor = build_sam2_video_predictor(cfg, str(ROOT / ckpt), device=args.device)
    imgp = SAM2ImagePredictor(predictor)

    rows, tiles = [], []
    hdr = f"{'shot':>4} {'frames':>13} {'mat':>6} {'blue_px':>8} {'sc':>3} {'ctm':>5} {'white_px':>9} {'sc':>3} {'ctm':>5}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for si, s in enumerate(shots):
        a, b = s["start"], s["end"]
        frames = [f for f in (grab(cap, i) for i in np.linspace(a, b - 1, 15).astype(int))
                  if f is not None]
        mat = fit_mat(frames, graphics)
        if mat is None:
            print(f"{si:4d} {a:6d}-{b:6d}   MAT FAIL")
            rows.append(dict(shot=si, start=a, end=b, mat=None))
            continue

        # Do not stake the shot on one arbitrary frame. Every threshold change
        # made so far fixed one shot and broke another -- 28 then 6, 6 then 30,
        # 30 then 26 -- because the per-shot colour model couples both athletes
        # and a single mid-frame is a single point of failure. A shot is 150 to
        # 1700 frames long; take the first candidate frame that yields both.
        exclude = {gi: mat_confusable_with(mat, gi) for gi in ("blue", "white")}
        near_mat = cv2.dilate(mat.hull.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
        row = dict(shot=si, start=a, end=b, mat=round(float(mat.hull.mean()), 3),
                   modes=len(mat.modes))
        def sweep(exc, want=("blue", "white")):
            best = None
            for frac in (0.50, 0.30, 0.70, 0.15, 0.85):
                cand = grab(cap, a + int(frac * (b - a)))
                if cand is None:
                    continue
                votes = gi_votes(cand, mat, graphics, exclude_mat=exc)
                with torch.inference_mode():
                    imgp.set_image(cv2.cvtColor(cand, cv2.COLOR_BGR2RGB))
                attempt, got = {}, {}
                for key in want:
                    other = "white" if key == "blue" else "blue"
                    pt, area = seed_point(votes[key], near_mat)
                    if pt is None:
                        attempt[key] = None
                        continue
                    with torch.inference_mode():
                        masks, _, _ = imgp.predict(point_coords=pt,
                                                   point_labels=np.ones(1, np.int32),
                                                   multimask_output=True)
                    m, idx, contam = choose_scale(masks, votes[key], votes[other])
                    if m is None:
                        attempt[key] = None
                        continue
                    got[key] = m
                    attempt[key] = dict(px=int(m.sum()), scale=int(idx),
                                        contam=round(float(contam), 3),
                                        seed=[int(pt[0][0]), int(pt[0][1])], colour_px=area)
                n_found = sum(1 for v in attempt.values() if v)
                if best is None or n_found > best[0]:
                    best = (n_found, attempt, frac, got, cand)
                if n_found == len(want):
                    break
            return best

        best_try = sweep(exclude)
        picked = dict(best_try[3]) if best_try else {}
        mid = best_try[4] if best_try else None
        found = dict(best_try[1]) if best_try else {"blue": None, "white": None}

        # LAST RESORT, and only for a gi that found nothing on any frame. Dropping
        # the mat subtraction is what rescues shot 30, where the athlete was
        # absorbed into the mat model itself (952 colour px with the subtraction,
        # 138,238 without). Every earlier attempt at this was a change to the
        # shared mat model, which couples all 37 shots and traded one failure for
        # another four times over. This cannot: it runs only where the normal path
        # already returned nothing.
        missing = [gi for gi in ("blue", "white") if not found.get(gi)]
        if missing:
            relaxed = dict(exclude)
            for gi in missing:
                relaxed[gi] = False
            alt = sweep(relaxed, want=tuple(missing))
            if alt:
                for gi in missing:
                    if alt[1].get(gi):
                        found[gi] = dict(alt[1][gi], via="no_mat_subtraction")
                        picked[gi] = alt[3][gi]
                        if mid is None:
                            mid = alt[4]

        row.update(found)
        row["seed_frac"] = best_try[2] if best_try else None

        def fmt(k):
            v = row.get(k)
            return (f"{'--':>8} {'-':>3} {'-':>5}" if not v
                    else f"{v['px']:8d} {v['scale']:3d} {v['contam']:5.3f}")
        print(f"{si:4d} {a:6d}-{b:6d} {row['mat']:6.3f} {fmt('blue')} {fmt('white')}")
        rows.append(row)

        ov = mid.copy()
        ov[mat.hull] = (0.90 * ov[mat.hull] + 0.10 * np.array([0, 255, 0])).astype(np.uint8)
        for key, col in (("blue", (255, 60, 0)), ("white", (0, 160, 255))):
            if key in picked:
                m = picked[key]
                ov[m] = (0.40 * ov[m] + 0.60 * np.array(col)).astype(np.uint8)
        cv2.putText(ov, f"shot {si}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        tiles.append(cv2.resize(ov, (426, 240)))
    cap.release()

    out = args.data_dir / "out" / f"{args.slug}_matfield.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    ok = [r for r in rows if r.get("mat") is not None]
    both = [r for r in ok if r.get("blue") and r.get("white")]
    print(f"\nmat resolved      : {len(ok)}/{len(shots)}")
    print(f"both athletes seeded: {len(both)}/{len(shots)}")
    if both:
        bp = np.array([r["blue"]["px"] for r in both])
        wp = np.array([r["white"]["px"] for r in both])
        ct = np.array([max(r["blue"]["contam"], r["white"]["contam"]) for r in both])
        print(f"  blue  px : median {np.median(bp):7.0f}  min {bp.min():6d}  max {bp.max():6d}")
        print(f"  white px : median {np.median(wp):7.0f}  min {wp.min():6d}  max {wp.max():6d}")
        print(f"  worst contamination per shot: median {np.median(ct):.3f}  max {ct.max():.3f}")
    for r in rows:
        if r.get("mat") is None or not (r.get("blue") and r.get("white")):
            print(f"  !! shot {r['shot']} ({r['start']}-{r['end']}) incomplete")
    print(f"\n-> {out}")

    if tiles:
        sheet = args.sheet or (args.data_dir / "out" / f"{args.slug}_matfield.jpg")
        cols = 4
        rows_n = (len(tiles) + cols - 1) // cols
        blank = np.zeros_like(tiles[0])
        grid = np.vstack([np.hstack((tiles[r * cols:(r + 1) * cols]
                                     + [blank] * cols)[:cols]) for r in range(rows_n)])
        cv2.imwrite(str(sheet), grid)
        print(f"-> {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
