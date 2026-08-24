"""Skeleton attribution is the load-bearing step of the deliverable.

If it silently assigns the wrong athlete, every downstream label is corrupt and
nothing about the pose table looks obviously wrong. So it gets tested against the
case it exists for: two overlapping bodies plus a stitched skeleton that belongs
to neither.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import make_frame

from bjjvision.features import attribute_skeletons, extract


def fake_skeleton(mask, jitter=0.0, rng=None):
    """17 keypoints scattered inside a mask, like a pose estimator would emit."""
    rng = rng or np.random.default_rng(0)
    ys, xs = np.nonzero(mask)
    idx = rng.choice(len(ys), size=17, replace=True)
    kp = np.stack([xs[idx].astype(float), ys[idx].astype(float),
                   np.full(17, 0.9)], axis=1)
    if jitter:
        kp[:, :2] += rng.normal(0, jitter, size=(17, 2))
    return kp


def test_attribution_separated():
    _, masks, _ = make_frame(1.0, overlap=0.0)
    rng = np.random.default_rng(1)
    ka, kb = fake_skeleton(masks["A"], rng=rng), fake_skeleton(masks["B"], rng=rng)
    got = attribute_skeletons([ka, kb], masks)
    assert got["A"] is not None and got["B"] is not None, "both should attribute"
    assert np.allclose(got["A"], ka), "A got the wrong skeleton"
    assert np.allclose(got["B"], kb), "B got the wrong skeleton"
    print("  separated bodies       -> both attributed correctly")


def test_attribution_under_occlusion():
    _, masks, _ = make_frame(5.0, overlap=0.92)
    rng = np.random.default_rng(2)
    ka, kb = fake_skeleton(masks["A"], rng=rng), fake_skeleton(masks["B"], rng=rng)
    got = attribute_skeletons([ka, kb], masks)
    ok_a = got["A"] is not None and np.allclose(got["A"], ka)
    ok_b = got["B"] is not None and np.allclose(got["B"], kb)
    print(f"  92% overlap            -> A {'ok' if ok_a else 'FAILED'}, "
          f"B {'ok' if ok_b else 'FAILED'}")
    assert ok_a and ok_b


def test_stitched_skeleton_is_refused():
    """The failure a pose estimator actually makes: one skeleton built from
    both athletes. Writing that in as a real athlete is worse than writing nothing."""
    _, masks, _ = make_frame(6.0, overlap=0.85)
    rng = np.random.default_rng(3)
    ka, kb = fake_skeleton(masks["A"], rng=rng), fake_skeleton(masks["B"], rng=rng)
    stitched = np.vstack([ka[:9], kb[9:]])          # half from each body
    got = attribute_skeletons([stitched], masks)
    n = sum(v is not None for v in got.values())
    print(f"  stitched skeleton      -> attributed to {n} athlete(s) (want 0)")
    assert n == 0, "a skeleton spanning both masks must be refused, not guessed"


def test_geometry_signs_are_meaningful():
    # overlap must be high enough that the bodies genuinely intersect; at 0.5 the
    # synthetic ellipses barely touch (mask_iou ~ 0) and there is no occlusion to
    # measure, which is what made an earlier version of this test vacuous
    _, masks, _ = make_frame(6.0, overlap=0.88)
    rng = np.random.default_rng(4)
    sk = {"A": fake_skeleton(masks["A"], rng=rng), "B": fake_skeleton(masks["B"], rng=rng)}
    ff = extract(42, 1.4, masks, sk)
    row = ff.to_row(640, 360)
    print(f"  contact_len={row['contact_len']:.3f}  mask_iou={row['mask_iou']:.3f}  "
          f"occl A<-B={row['occl_a_by_b']:.3f} B<-A={row['occl_b_by_a']:.3f}")
    assert 0.0 <= row["contact_len"] <= 1.0
    assert row["A_attributed"] and row["B_attributed"]
    assert row["A_nose_x"] is not None and 0.0 <= row["A_nose_x"] <= 1.0, "normalised"
    # A is drawn over B in the synthetic scene, so B's outline should be the one
    # bounded by the other body
    assert row["mask_iou"] == 0.0, "synth carves A out of B, so masks are disjoint"
    assert row["occl_b_by_a"] > row["occl_a_by_b"], (
        f"occlusion evidence backwards: B<-A={row['occl_b_by_a']:.3f} "
        f"should exceed A<-B={row['occl_a_by_b']:.3f}")
    print("  occlusion evidence points at the athlete drawn on top")


def test_no_masks_degrades_quietly():
    empty = {"A": np.zeros((360, 640), bool), "B": np.zeros((360, 640), bool)}
    ff = extract(0, 0.0, empty, {"A": None, "B": None})
    row = ff.to_row(640, 360)
    assert row["A_mask_area"] == 0.0 and row["A_nose_x"] is None
    print("  empty frame            -> row emitted with nulls, no crash")


if __name__ == "__main__":
    print("\n=== pose attribution + geometry ===\n")
    tests = [test_attribution_separated, test_attribution_under_occlusion,
             test_stitched_skeleton_is_refused, test_geometry_signs_are_meaningful,
             test_no_masks_degrades_quietly]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1; print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1; print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n=== {len(tests)-failed}/{len(tests)} passed ===")
    sys.exit(1 if failed else 0)
