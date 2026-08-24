"""Does the colour anchor actually detect and repair an identity swap?

This is the load-bearing claim of the whole design, so it gets tested directly
rather than inferred from a clean-video demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from synth import make_frame

from bjjvision.appearance import (PixelClassifier, build_color_model,
                                  separability)
from bjjvision.identity import Health, IdentityManager

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text())


def _calibrate(mgr):
    samples = []
    for i in range(10):
        img, masks, _ = make_frame(i * 0.3, overlap=0.0)
        samples.append((img, masks))
    return mgr.calibrate(samples)


def test_separability():
    mgr = IdentityManager(CFG)
    sep = _calibrate(mgr)
    print(f"  gi separability            = {sep:.3f}")
    assert sep > 0.6, f"white vs navy should separate strongly, got {sep:.3f}"
    return sep


def test_purity_under_occlusion():
    mgr = IdentityManager(CFG)
    _calibrate(mgr)
    img, masks, _ = make_frame(5.0, overlap=0.92)
    fh = mgr.audit(100, img, masks, {"A": 0.9, "B": 0.9})
    print(f"  purity @92% overlap        = A {fh.purity['A']:.2f}  B {fh.purity['B']:.2f}")
    assert fh.purity["A"] > 0.8 and fh.purity["B"] > 0.8, "clean masks must stay pure"
    return fh


def test_swap_is_detected():
    """The failure we actually fear: SAM2 keeps two good masks but swaps the labels."""
    mgr = IdentityManager(CFG)
    _calibrate(mgr)
    img, masks, _ = make_frame(6.0, overlap=0.85)
    swapped = {"A": masks["B"], "B": masks["A"]}
    fh = mgr.audit(200, img, swapped, {"A": 0.93, "B": 0.93})
    print(f"  purity after label swap    = A {fh.purity['A']:.2f}  B {fh.purity['B']:.2f}")
    print(f"  state                      = {fh.state.value}")
    print(f"  triggers                   = {fh.triggers}")
    assert fh.purity["A"] < 0.3 and fh.purity["B"] < 0.3, "swap must collapse purity"
    assert fh.state is not Health.HEALTHY, "swap must not be reported healthy"
    assert any("both_masks_disown" in t for t in fh.triggers), "swap signature must fire"
    return fh


def test_contaminated_mask_is_split():
    """One mask swallows both athletes -- can the posterior put it back?"""
    mgr = IdentityManager(CFG)
    _calibrate(mgr)
    img, masks, _ = make_frame(7.0, overlap=0.80)
    blob = masks["A"] | masks["B"]
    merged = {"A": blob, "B": np.zeros_like(blob)}
    fh = mgr.audit(300, img, merged, {"A": 0.7, "B": 0.1})
    repaired = mgr.soft_repair(img, merged)

    iou_a = (repaired["A"] & masks["A"]).sum() / max((repaired["A"] | masks["A"]).sum(), 1)
    iou_b = (repaired["B"] & masks["B"]).sum() / max((repaired["B"] | masks["B"]).sum(), 1)
    print(f"  merged-blob detected       = {fh.state.value}, triggers={fh.triggers[:2]}")
    print(f"  IoU after colour split     = A {iou_a:.2f}  B {iou_b:.2f}")
    assert iou_a > 0.6 and iou_b > 0.6, "split should recover both athletes"
    return iou_a, iou_b


def test_referee_is_a_colour_outlier():
    mgr = IdentityManager(CFG)
    _calibrate(mgr)
    img, _, mref = make_frame(3.0, overlap=0.3)
    ap = CFG["appearance"]
    cm = build_color_model(img, mref, tuple(ap["hist_bins"]),
                           tuple(ap["torso_band"]), ap["min_mask_pixels"])
    d_a = mgr.protos["A"].distance(cm)
    d_b = mgr.protos["B"].distance(cm)
    margin = CFG["roles"]["referee_color_margin"]
    print(f"  referee dist to A/B        = {d_a:.2f} / {d_b:.2f}  (reject above {margin})")
    assert min(d_a, d_b) > margin, "referee must not look like either gi"
    return min(d_a, d_b)


def test_prototype_resists_poisoning():
    """Feed a deliberately wrong exemplar through the gate; the bank must reject it."""
    mgr = IdentityManager(CFG)
    _calibrate(mgr)
    before = len(mgr.protos["A"].exemplars)
    img, masks, _ = make_frame(8.0, overlap=0.9)
    poisoned = {"A": masks["B"], "B": masks["A"]}
    fh = mgr.audit(400, img, poisoned, {"A": 0.9, "B": 0.9})
    mgr.maybe_update_prototypes(img, poisoned, fh)
    after = len(mgr.protos["A"].exemplars)
    print(f"  prototype exemplars        = {before} -> {after} (must not grow)")
    assert after == before, "a low-confidence frame must never enter the bank"


if __name__ == "__main__":
    print("\n=== colour anchor validation (synthetic, CPU) ===\n")
    tests = [
        ("separability", test_separability),
        ("purity under 92% occlusion", test_purity_under_occlusion),
        ("identity swap detection", test_swap_is_detected),
        ("contaminated mask split", test_contaminated_mask_is_split),
        ("referee colour outlier", test_referee_is_a_colour_outlier),
        ("prototype poisoning resistance", test_prototype_resists_poisoning),
    ]
    failed = 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn(); print("  PASS\n")
        except AssertionError as e:
            failed += 1; print(f"  FAIL: {e}\n")
        except Exception as e:
            failed += 1; print(f"  ERROR: {type(e).__name__}: {e}\n")
    print(f"=== {len(tests) - failed}/{len(tests)} passed ===")
    sys.exit(1 if failed else 0)
