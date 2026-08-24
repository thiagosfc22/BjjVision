"""Deciding who is actually fighting: athlete vs referee vs crowd.

Three independent filters, deliberately chosen so their failure modes do not
overlap:

  1. MAT MEMBERSHIP  (where)  -- learned as a colour model, not a fixed polygon,
     because broadcast cameras pan and zoom and any hardcoded region drifts off.
  2. CONTACT         (what)   -- the two athletes are in near-continuous physical
     contact; the referee circles and only brushes past. This is the load-bearing
     signal, because it holds whether they are standing or on the ground.
  3. COLOUR OUTLIER  (who)    -- once both gi prototypes exist, anyone far from
     BOTH is not a competitor. Cheap, and it catches the referee instantly since
     referee uniforms are chosen to contrast with gis by design.

Posture is deliberately NOT trusted on its own: "referee stands, fighters are on
the ground" is true for most of a match and false for exactly the opening
exchange, which is when identities are first being locked in.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .appearance import ColorModel, FighterPrototype, hist_distance


@dataclass
class PersonObs:
    """One detected person in one frame."""
    track_id: int
    box: tuple[float, float, float, float]     # x1,y1,x2,y2
    mask: np.ndarray | None = None
    keypoints: np.ndarray | None = None        # (17,3) COCO xy+conf
    score: float = 0.0

    @property
    def foot_point(self) -> tuple[float, float]:
        """Ground contact point -- ankles when visible, else bottom-centre of box."""
        if self.keypoints is not None and self.keypoints.shape[0] >= 17:
            ank = self.keypoints[[15, 16]]
            vis = ank[ank[:, 2] > 0.3]
            if vis.shape[0]:
                return float(vis[:, 0].mean()), float(vis[:, 1].mean())
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2.0, y2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# --------------------------------------------------------------------------
# 1. Mat
# --------------------------------------------------------------------------
class MatModel:
    """Per-frame competition-area mask, learned from the video's own background.

    Learned as colour rather than geometry so it survives camera movement: the
    mat keeps its colour when the operator pans, a polygon does not.
    """

    def __init__(self, tol: float = 26.0):
        self.tol = tol
        self.center_lab: np.ndarray | None = None
        self._last_poly: np.ndarray | None = None

    def fit(self, frames: list[np.ndarray]) -> "MatModel":
        """Median-background the sampled frames, then take the dominant colour of
        the lower-central region -- where the mat is, in every broadcast framing."""
        if not frames:
            raise ValueError("MatModel.fit needs at least one frame")
        stack = np.stack([cv2.resize(f, (320, 180)) for f in frames])
        bg = np.median(stack, axis=0).astype(np.uint8)      # people wash out, mat stays
        lab = cv2.cvtColor(bg, cv2.COLOR_BGR2Lab)
        h, w = lab.shape[:2]
        roi = lab[int(0.45 * h):, int(0.15 * w):int(0.85 * w)].reshape(-1, 3).astype(np.float32)

        # k-means over the ROI; the mat is the largest cluster by construction
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(roi, 3, None, crit, 5, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=3)
        self.center_lab = centers[int(np.argmax(counts))]
        return self

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.center_lab is None:
            return np.ones(frame_bgr.shape[:2], dtype=bool)
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        dist = np.linalg.norm(lab - self.center_lab.reshape(1, 1, 3), axis=2)
        m = (dist < self.tol).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n <= 1:
            return np.zeros(frame_bgr.shape[:2], dtype=bool)
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = lbl == biggest
        # convex hull: athletes standing on the mat punch holes in the colour mask,
        # and we want those pixels to still count as "on the mat"
        cnts, _ = cv2.findContours(area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            hull = cv2.convexHull(np.vstack(cnts))
            self._last_poly = hull
            filled = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(filled, hull, 1)
            return filled.astype(bool)
        return area

    def contains(self, mat_mask: np.ndarray, person: PersonObs, pad: int = 18) -> bool:
        fx, fy = person.foot_point
        h, w = mat_mask.shape
        y0, y1 = max(0, int(fy) - pad), min(h, int(fy) + pad + 1)
        x0, x1 = max(0, int(fx) - pad), min(w, int(fx) + pad + 1)
        if y0 >= y1 or x0 >= x1:
            return False
        return bool(mat_mask[y0:y1, x0:x1].any())


# --------------------------------------------------------------------------
# 2. Contact
# --------------------------------------------------------------------------
def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class ContactTracker:
    """Sustained-contact score for every pair of tracks over a rolling window.

    Grappling means the athletes' silhouettes overlap or touch nearly every frame
    for the whole match. No other pair on screen -- fighter/referee,
    referee/cornerman, two spectators -- sustains that. Integrating over a window
    rather than scoring per frame is what makes it robust: the referee leaning in
    to check a choke spikes for a few frames, then decays.
    """

    def __init__(self, window_frames: int = 90, iou_min: float = 0.02):
        self.window = window_frames
        self.iou_min = iou_min
        self._hist: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=window_frames))
        self._seen: dict[int, int] = defaultdict(int)

    def update(self, persons: list[PersonObs]) -> None:
        for p in persons:
            self._seen[p.track_id] += 1
        for i, a in enumerate(persons):
            for b in persons[i + 1:]:
                key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                touching = _box_iou(a.box, b.box) >= self.iou_min
                if a.mask is not None and b.mask is not None and not touching:
                    # boxes can miss contact that the silhouettes have (a leg across)
                    da = cv2.dilate(a.mask.astype(np.uint8), np.ones((9, 9), np.uint8))
                    touching = bool((da.astype(bool) & b.mask.astype(bool)).any())
                self._hist[key].append(1.0 if touching else 0.0)

    def score(self, tid_a: int, tid_b: int) -> float:
        key = (min(tid_a, tid_b), max(tid_a, tid_b))
        h = self._hist.get(key)
        if not h:
            return 0.0
        return float(np.mean(h))

    def best_pair(self, candidates: list[int], min_support: int = 15) -> tuple[int, int] | None:
        best, best_s = None, -1.0
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                key = (min(a, b), max(a, b))
                if len(self._hist.get(key, ())) < min_support:
                    continue
                s = self.score(a, b)
                if s > best_s:
                    best, best_s = (a, b), s
        return best


# --------------------------------------------------------------------------
# 3. Role assignment
# --------------------------------------------------------------------------
@dataclass
class RoleDecision:
    fighters: tuple[int, int] | None = None
    referee: int | None = None
    crowd: set[int] = field(default_factory=set)
    confidence: float = 0.0
    reasons: dict[int, str] = field(default_factory=dict)


class RoleAssigner:
    def __init__(self, cfg: dict):
        r = cfg["roles"]
        self.min_area_frac = cfg["detect"]["min_box_area_frac"]
        self.crowd_y = r["crowd_y_reject_frac"]
        self.ref_margin = r["referee_color_margin"]
        self.require_mat = r["require_inside_mat"]
        self.contact = ContactTracker(
            window_frames=int(r["contact_window_s"] * cfg["video"]["target_fps"]),
            iou_min=r["contact_iou_min"])

    def update(self, persons: list[PersonObs]) -> None:
        self.contact.update(persons)

    def assign(self, frame_bgr: np.ndarray, persons: list[PersonObs],
               mat_mask: np.ndarray | None,
               protos: tuple[FighterPrototype, FighterPrototype] | None,
               color_of: dict[int, ColorModel] | None = None) -> RoleDecision:
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        dec = RoleDecision()

        # --- cheap geometric rejection first: crowd -------------------------
        # The mat test is applied LAST and is not a veto. Three filters exist so
        # their failure modes do not coincide; letting any one of them reject
        # unilaterally throws that away. Measured case: at a tighter camera framing
        # the mat covered 31% of the frame instead of 46%, its convex hull shrank,
        # and the largest body on screen was discarded for having a foot just
        # outside it -- leaving one candidate and no fight to track.
        on_mat: list[PersonObs] = []
        off_mat: list[PersonObs] = []
        for p in persons:
            if p.area / frame_area < self.min_area_frac:
                dec.crowd.add(p.track_id); dec.reasons[p.track_id] = "too small (stands)"
                continue
            _, fy = p.foot_point
            if fy / h < self.crowd_y:
                dec.crowd.add(p.track_id); dec.reasons[p.track_id] = "feet above mat horizon"
                continue
            if self.require_mat and mat_mask is not None and not MatModel().contains(mat_mask, p):
                off_mat.append(p)
                continue
            on_mat.append(p)

        # A frame with a match in it has two athletes in it. If the mat test left
        # fewer than two candidates it is the test that is wrong, not the frame,
        # so re-admit the largest bodies it discarded rather than tracking nothing.
        if len(on_mat) < 2 and off_mat:
            off_mat.sort(key=lambda p: p.area, reverse=True)
            for p in off_mat[: 2 - len(on_mat)]:
                on_mat.append(p)
                dec.reasons[p.track_id] = "re-admitted: mat test left too few candidates"

        for p in off_mat:
            if p not in on_mat:
                dec.crowd.add(p.track_id)
                dec.reasons.setdefault(p.track_id, "outside mat polygon")

        if len(on_mat) < 2:
            return dec

        # --- colour outliers: referee and anyone else in the frame ----------
        ref_candidates: set[int] = set()
        if protos and color_of and all(p.ready for p in protos):
            for p in on_mat:
                cm = color_of.get(p.track_id)
                if cm is None:
                    continue
                d = min(hist_distance(protos[0].model, cm), hist_distance(protos[1].model, cm))
                if d > self.ref_margin:
                    ref_candidates.add(p.track_id)
                    dec.reasons[p.track_id] = f"gi colour unlike both fighters (d={d:.2f})"

        # --- the decisive signal: sustained contact -------------------------
        pool = [p.track_id for p in on_mat if p.track_id not in ref_candidates]
        if len(pool) < 2:
            pool = [p.track_id for p in on_mat]
        pair = self.contact.best_pair(pool)
        if pair is None:                       # too early for contact history
            biggest = sorted(on_mat, key=lambda p: p.area, reverse=True)[:2]
            dec.fighters = (biggest[0].track_id, biggest[1].track_id)
            dec.confidence = 0.35
            for p in biggest:
                dec.reasons[p.track_id] = "largest on-mat person (contact history warming up)"
        else:
            dec.fighters = pair
            dec.confidence = min(1.0, 0.45 + 0.55 * self.contact.score(*pair))
            for tid in pair:
                dec.reasons[tid] = f"sustained contact {self.contact.score(*pair):.2f}"

        # referee = the on-mat non-fighter that is closest to the action
        leftovers = [p for p in on_mat if p.track_id not in (dec.fighters or ())]
        if leftovers:
            ref = max(leftovers, key=lambda p: max(
                self.contact.score(p.track_id, t) for t in dec.fighters) if dec.fighters else 0.0)
            dec.referee = ref.track_id
            dec.reasons.setdefault(ref.track_id, "on mat, not in sustained contact -> referee")
        return dec
