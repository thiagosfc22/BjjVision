"""Does scale selection prefer the athlete over the garment?

The failure this guards against is silent by construction: a jacket-only mask
scores 0.997 on purity and passes every health check in the pipeline while
missing half its athlete. So the criterion is tested on the property purity
cannot express -- completeness -- and on the two ways a size-seeking rule can
go wrong: swallowing the opponent, and handing back less than it was given.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
from synth import make_frame

from bjjvision.appearance import build_color_model
from bjjvision.identity import IdentityManager

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "default.yaml").read_text())


def _calibrated():
    mgr = IdentityManager(CFG)
    samples = []
    for i in range(10):
        img, masks, _ = make_frame(i * 0.3, overlap=0.0)
        samples.append((img, masks))
    mgr.calibrate(samples)
    return mgr


def _truncate_top(mask, frac=0.5):
    """Keep only the upper `frac` of a mask -- a stand-in for 'jacket only'."""
    ys, xs = np.nonzero(mask)
    cut = ys.min() + int(frac * (ys.max() - ys.min()))
    out = np.zeros_like(mask)
    out[:cut] = mask[:cut]
    return out


def test_prefers_whole_body_over_part():
    mgr = _calibrated()
    img, masks, _ = make_frame(1.0, overlap=0.0)
    whole = masks["A"].astype(bool)
    jacket = _truncate_top(whole)
    # candidates offered in the worst order: the part first, as SAM2 often does
    cands = np.stack([jacket, whole])
    pick = mgr.choose_scale(img, cands, "A", held=jacket)
    frac = pick.sum() / whole.sum()
    print(f"  jacket {jacket.sum()/whole.sum():.2f} of body, chosen {frac:.2f} of body")
    assert frac > 0.95, "criterion kept the garment instead of the athlete"


def test_rejects_a_scale_that_swallowed_the_opponent():
    mgr = _calibrated()
    img, masks, _ = make_frame(1.0, overlap=0.0)
    a, b = masks["A"].astype(bool), masks["B"].astype(bool)
    merged = a | b                       # the "both athletes" scale
    cands = np.stack([a, merged])
    pick = mgr.choose_scale(img, cands, "A", held=a)
    iou_with_b = (pick & b).sum() / max(b.sum(), 1)
    print(f"  merged scale is {merged.sum()/a.sum():.1f}x bigger; "
          f"chosen overlaps B by {iou_with_b:.2f}")
    assert iou_with_b < 0.25, "criterion took the merged blob just because it was larger"


def test_never_returns_less_than_it_was_given():
    """Measured regression: on a frame where every candidate was contaminated,
    the criterion returned a 0.0036 fragment over the 0.0347 mask in hand."""
    mgr = _calibrated()
    img, masks, _ = make_frame(1.0, overlap=0.0)
    a, b = masks["A"].astype(bool), masks["B"].astype(bool)
    ys, xs = np.nonzero(a)
    fragment = np.zeros_like(a)
    fragment[ys.min():ys.min() + 12, xs.min():xs.min() + 12] = True
    cands = np.stack([fragment, a | b])          # a scrap, and a contaminated blob
    pick = mgr.choose_scale(img, cands, "A", held=a, min_px=100)
    print(f"  held {a.sum()} px, candidates {fragment.sum()} and {(a|b).sum()}, "
          f"chosen {pick.sum()} px")
    assert pick.sum() >= a.sum(), "a seed handed back less than it started with"


def test_purity_cannot_see_the_truncation():
    """Why a size-seeking criterion had to be added at all.

    Purity is a precision measure with no recall term, so it cannot separate a
    complete mask from half of one -- on this uniform synthetic gi it scores
    both at 1.000, and on real footage it rated a jacket-only mask 0.997 while
    that mask's own athlete's legs were missing. Whatever chooses between
    object scales, it cannot be this number.
    """
    mgr = _calibrated()
    img, masks, _ = make_frame(1.0, overlap=0.0)
    whole = masks["A"].astype(bool)
    jacket = _truncate_top(whole)
    p_jacket = mgr.classifier.purity(img, jacket, 0)
    p_whole = mgr.classifier.purity(img, whole, 0)
    print(f"  mask covers {jacket.sum()/whole.sum():.0%} vs 100% of the athlete;"
          f"  purity says {p_jacket:.3f} vs {p_whole:.3f}")
    assert p_jacket >= p_whole - 0.02, (
        "purity now penalises truncation -- re-check whether this change is still needed")


if __name__ == "__main__":
    print("\n=== object-scale selection (synthetic, CPU) ===\n")
    tests = [
        ("prefers whole body over garment", test_prefers_whole_body_over_part),
        ("rejects a merged-athletes scale", test_rejects_a_scale_that_swallowed_the_opponent),
        ("never regresses below the held mask", test_never_returns_less_than_it_was_given),
        ("purity cannot see the truncation", test_purity_cannot_see_the_truncation),
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
