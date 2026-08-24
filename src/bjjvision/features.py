"""The deliverable: per-frame, per-athlete pose and contact geometry.

This module is the reason the mask pipeline exists. Off-the-shelf pose estimators
fail in grappling for one specific reason: once two bodies interlock, the
estimator cannot tell whose limb is whose, and it happily assembles a skeleton
from both athletes. The gi-colour-anchored mask solves exactly that, so a
detected skeleton can be attributed by asking which mask its keypoints fall in.

Design rule for everything below: **emit measurements, not verdicts.**

It is tempting to write `top_athlete = "A"` into the table. Resist it. Whichever
heuristic decides that gets frozen into the dataset, and every model trained on
it inherits the heuristic's errors with no way to recover the underlying signal.
So the table carries centroid separation, area ratio, keypoint visibility counts
and occlusion evidence, and the downstream classifier is allowed to weigh them.
The one exception is `attributed_to`, which is a measurement (which mask holds
this skeleton), not an interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# COCO-17 indices, named because magic numbers in geometry code are unreadable
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

KP_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
            "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
            "l_knee", "r_knee", "l_ankle", "r_ankle"]


def attribute_skeletons(keypoints: list[np.ndarray], masks: dict[str, np.ndarray],
                        min_conf: float = 0.3,
                        min_share: float = 0.55) -> dict[str, np.ndarray | None]:
    """Assign each detected skeleton to fighter A or B by mask membership.

    A skeleton belongs to whichever mask holds most of its confident keypoints.
    `min_share` refuses ambiguous cases rather than guessing: a skeleton split
    evenly across both masks is usually the estimator having stitched two people
    together, and writing that into the dataset as one athlete is worse than
    writing nothing.
    """
    out: dict[str, np.ndarray | None] = {"A": None, "B": None}
    best: dict[str, float] = {"A": -1.0, "B": -1.0}
    if not keypoints:
        return out

    for kp in keypoints:
        if kp is None or kp.shape[0] < 17:
            continue
        vis = kp[kp[:, 2] >= min_conf]
        if vis.shape[0] < 4:
            continue
        counts: dict[str, int] = {}
        for fid, m in masks.items():
            if m is None or not m.any():
                counts[fid] = 0
                continue
            h, w = m.shape
            ys = np.clip(vis[:, 1].astype(int), 0, h - 1)
            xs = np.clip(vis[:, 0].astype(int), 0, w - 1)
            counts[fid] = int(m[ys, xs].sum())
        total = sum(counts.values())
        if total == 0:
            continue
        fid = max(counts, key=counts.get)
        share = counts[fid] / total
        if share < min_share:
            continue                      # straddles both masks: refuse to guess
        score = share * vis.shape[0]
        if score > best[fid]:
            best[fid] = score
            out[fid] = kp
    return out


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    m = cv2.moments(mask.view(np.uint8) if mask.dtype == np.bool_ else mask.astype(np.uint8))
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _kp_mid(kp: np.ndarray | None, a: int, b: int, min_conf: float = 0.3):
    if kp is None:
        return None
    pts = [kp[i] for i in (a, b) if kp[i, 2] >= min_conf]
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))


def _occlusion_evidence(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Evidence for which athlete is in front, via silhouette completeness.

    The obvious measure, "how much of each outline touches the other body", is
    useless here: along an occlusion boundary the two outlines are the same line,
    so it is symmetric by construction and cannot distinguish front from behind.

    The asymmetry that does exist is completeness. The athlete in front keeps a
    whole silhouette. The one behind has a bite taken out of it, and that bite is
    filled by the body in front. So: take each mask's convex hull, subtract the
    mask to get its concavities, and measure how much of that missing area the
    other athlete occupies.

    Returns (a_occluded_by_b, b_occluded_by_a). Both are evidence, not a verdict.
    A body can be concave for ordinary anatomical reasons, which is exactly why
    this stays a feature instead of becoming a `top_athlete` column.
    """
    out = []
    for fg, other in ((a, b), (b, a)):
        u8 = fg.view(np.uint8) if fg.dtype == np.bool_ else fg.astype(np.uint8)
        cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            out.append(0.0)
            continue
        hull = cv2.convexHull(np.vstack(cnts))
        filled = np.zeros(fg.shape, np.uint8)
        cv2.fillConvexPoly(filled, hull, 1)
        concavity = filled.astype(bool) & ~fg.astype(bool)
        area = float(concavity.sum())
        if area < 1.0:
            out.append(0.0)
            continue
        out.append(float((concavity & other.astype(bool)).sum()) / area)
    return out[0], out[1]


@dataclass
class FrameFeatures:
    """One row of the dataset."""
    frame: int
    t_s: float
    shot_id: int = -1
    shot_kind: str = ""
    track_confidence: float = 0.0
    track_state: str = ""
    # per athlete
    kp: dict[str, np.ndarray | None] = field(default_factory=dict)
    mask_area: dict[str, float] = field(default_factory=dict)
    centroid: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    bbox: dict[str, tuple[int, int, int, int] | None] = field(default_factory=dict)
    kp_visible: dict[str, int] = field(default_factory=dict)
    attributed: dict[str, bool] = field(default_factory=dict)
    # pairwise
    centroid_dy: float = 0.0
    centroid_dist: float = 0.0
    hip_dist: float = 0.0
    area_ratio: float = 1.0
    mask_iou: float = 0.0
    contact_len: float = 0.0
    occl_a_by_b: float = 0.0
    occl_b_by_a: float = 0.0
    a_kp_in_b: int = 0
    b_kp_in_a: int = 0

    def to_row(self, w: int, h: int) -> dict:
        """Flatten to a Parquet row. Coordinates normalised by frame size so the
        table survives a change of resolution."""
        row: dict = {
            "frame": self.frame, "t_s": round(self.t_s, 3),
            "shot_id": self.shot_id, "shot_kind": self.shot_kind,
            "track_confidence": round(self.track_confidence, 4),
            "track_state": self.track_state,
            "centroid_dy": round(self.centroid_dy / h, 5),
            "centroid_dist": round(self.centroid_dist / max(w, h), 5),
            "hip_dist": round(self.hip_dist / max(w, h), 5),
            "area_ratio": round(self.area_ratio, 4),
            "mask_iou": round(self.mask_iou, 5),
            "contact_len": round(self.contact_len, 5),
            "occl_a_by_b": round(self.occl_a_by_b, 5),
            "occl_b_by_a": round(self.occl_b_by_a, 5),
            "a_kp_in_b": self.a_kp_in_b, "b_kp_in_a": self.b_kp_in_a,
        }
        for fid in ("A", "B"):
            row[f"{fid}_mask_area"] = round(self.mask_area.get(fid, 0.0) / (w * h), 6)
            row[f"{fid}_kp_visible"] = self.kp_visible.get(fid, 0)
            row[f"{fid}_attributed"] = bool(self.attributed.get(fid, False))
            c = self.centroid.get(fid)
            row[f"{fid}_cx"] = round(c[0] / w, 5) if c else None
            row[f"{fid}_cy"] = round(c[1] / h, 5) if c else None
            kp = self.kp.get(fid)
            for i, name in enumerate(KP_NAMES):
                if kp is None:
                    row[f"{fid}_{name}_x"] = None
                    row[f"{fid}_{name}_y"] = None
                    row[f"{fid}_{name}_c"] = 0.0
                else:
                    row[f"{fid}_{name}_x"] = round(float(kp[i, 0]) / w, 5)
                    row[f"{fid}_{name}_y"] = round(float(kp[i, 1]) / h, 5)
                    row[f"{fid}_{name}_c"] = round(float(kp[i, 2]), 4)
        return row


def extract(frame_idx: int, t_s: float, masks: dict[str, np.ndarray],
            skeletons: dict[str, np.ndarray | None],
            min_conf: float = 0.3) -> FrameFeatures:
    """Build one row from this frame's masks and attributed skeletons."""
    ff = FrameFeatures(frame=frame_idx, t_s=t_s)

    for fid in ("A", "B"):
        m = masks.get(fid)
        kp = skeletons.get(fid)
        ff.kp[fid] = kp
        ff.attributed[fid] = kp is not None
        ff.kp_visible[fid] = int((kp[:, 2] >= min_conf).sum()) if kp is not None else 0
        if m is None or not m.any():
            ff.mask_area[fid] = 0.0
            ff.centroid[fid] = None
            ff.bbox[fid] = None
            continue
        ff.mask_area[fid] = float(m.sum())
        ff.centroid[fid] = _centroid(m)
        x, y, bw, bh = cv2.boundingRect(m.view(np.uint8) if m.dtype == np.bool_
                                        else m.astype(np.uint8))
        ff.bbox[fid] = (x, y, x + bw, y + bh)

    a, b = masks.get("A"), masks.get("B")
    if a is not None and b is not None and a.any() and b.any():
        ca, cb = ff.centroid.get("A"), ff.centroid.get("B")
        if ca and cb:
            ff.centroid_dy = ca[1] - cb[1]        # image space; sign is meaningful
            ff.centroid_dist = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        inter = float((a & b).sum())
        union = float((a | b).sum())
        ff.mask_iou = inter / union if union else 0.0
        ff.area_ratio = (ff.mask_area["A"] / ff.mask_area["B"]
                         if ff.mask_area.get("B") else 1.0)
        k = np.ones((7, 7), np.uint8)
        au8 = a.view(np.uint8) if a.dtype == np.bool_ else a.astype(np.uint8)
        a_edge = cv2.morphologyEx(au8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        bu8 = b.view(np.uint8) if b.dtype == np.bool_ else b.astype(np.uint8)
        ff.contact_len = float((a_edge & cv2.dilate(bu8, k).astype(bool)).sum()) / max(
            float(a_edge.sum()), 1.0)
        ff.occl_a_by_b, ff.occl_b_by_a = _occlusion_evidence(a, b)

        ka, kb = skeletons.get("A"), skeletons.get("B")
        h, w = a.shape
        for src, dst_mask, attr in ((ka, b, "a_kp_in_b"), (kb, a, "b_kp_in_a")):
            if src is None:
                continue
            vis = src[src[:, 2] >= min_conf]
            if not vis.shape[0]:
                continue
            ys = np.clip(vis[:, 1].astype(int), 0, h - 1)
            xs = np.clip(vis[:, 0].astype(int), 0, w - 1)
            setattr(ff, attr, int(dst_mask[ys, xs].sum()))

    ha, hb = _kp_mid(skeletons.get("A"), L_HIP, R_HIP), _kp_mid(skeletons.get("B"), L_HIP, R_HIP)
    if ha and hb:
        ff.hip_dist = float(np.hypot(ha[0] - hb[0], ha[1] - hb[1]))
    return ff


def write_parquet(rows: list[dict], path, meta: dict | None = None) -> str:
    """Parquet if pyarrow is present, CSV otherwise. The run must not die at the
    final step because of a missing optional dependency."""
    import json
    from pathlib import Path
    path = Path(path)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(rows)
        if meta:
            table = table.replace_schema_metadata(
                {k: json.dumps(v) for k, v in meta.items()})
        pq.write_table(table, path, compression="zstd")
        return str(path)
    except ImportError:
        import csv
        alt = path.with_suffix(".csv")
        if rows:
            with alt.open("w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
        return str(alt)
