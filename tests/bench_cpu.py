"""Measure the CPU-bound stages at production resolution.

Relevant because instance selection usually fixates on the GPU and accepts
whatever vCPU count comes bundled. In this pipeline the colour audit and the
report compositing are pure NumPy/OpenCV on the host -- if they are slower than
the GPU stages, extra CUDA cores buy nothing and the rental is wasted on a
bottleneck that lives on the other side of the PCIe bus.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import make_frame

from bjjvision.appearance import build_color_model
from bjjvision.identity import IdentityManager
from bjjvision.render import RenderState, ReportRenderer

CFG = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
W, H = 1280, 720          # production resolution


def upscale(img, masks, mref):
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    up = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return img, {k: up(v) for k, v in masks.items()}, up(mref)


def timeit(fn, n=25):
    fn()                                   # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000.0


def main():
    mgr = IdentityManager(CFG)
    cal = []
    for i in range(10):
        img, masks, mref = upscale(*make_frame(i * 0.3, 0.0))
        cal.append((img, masks))
    mgr.calibrate(cal)

    img, masks, mref = upscale(*make_frame(5.0, 0.85))
    rend = ReportRenderer(CFG, W, H, 600.0)
    st = RenderState(t_s=5.0, duration_s=600.0, confidence=0.82,
                     purity={"A": 0.9, "B": 0.85}, proto_dist={"A": 0.2, "B": 0.25},
                     labels={"A": "Galvao", "B": "Opponent"},
                     swatches={k: v.model.swatch_bgr for k, v in mgr.protos.items()},
                     position="side control", commentary="A consolidates side control.")

    ap = CFG["appearance"]
    rows = [
        ("build_color_model (x2)", lambda: [build_color_model(
            img, masks[f], tuple(ap["hist_bins"]), tuple(ap["torso_band"]),
            ap["min_mask_pixels"]) for f in ("A", "B")]),
        ("identity.audit", lambda: mgr.audit(1, img, masks, {"A": .9, "B": .9})),
        ("soft_repair (colour split)", lambda: mgr.soft_repair(img, masks)),
        ("reanchor_prompts", lambda: mgr.reanchor_prompts(img, masks, 6)),
        ("render.compose (HUD)", lambda: rend.compose(img, masks, st, mref)),
    ]

    print(f"\n=== CPU stages @ {W}x{H} (this machine: arm64) ===\n")
    total = 0.0
    for name, fn in rows:
        ms = timeit(fn)
        total += ms
        print(f"  {name:<28} {ms:7.2f} ms   ({1000/ms:6.1f} fps if alone)")
    print(f"  {'-'*28} {'-'*7}")
    print(f"  {'per-frame CPU total':<28} {total:7.2f} ms   ({1000/total:6.1f} fps single-core)\n")

    for label, mins in (("5 min match", 5), ("10 min match", 10)):
        n = mins * 60 * 30
        cpu_h = n * total / 1000 / 3600
        jpg_gb = n * 190_000 / 1e9        # ~190 KB per 720p q2 jpeg
        print(f"  {label}: {n:,} frames | frames on disk ~{jpg_gb:.1f} GB "
              f"| CPU work {cpu_h*60:.0f} min single-core")
    print()


if __name__ == "__main__":
    main()
