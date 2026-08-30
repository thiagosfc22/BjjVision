"""Mat delimitation and athlete seeding, rebuilt for footage the galvao-xande
path does not transfer to.

Why this exists alongside `roles.MatModel` rather than replacing it: on
dalpra-dorsey the global colour mat model is not merely weak, it is INVERTED at
some camera angles. Measured at t=630s against the learned Lab centre (tol=26),
the fraction of pixels within tolerance was 2.5% for the real mat, 0.6% for its
yellow border, and 57.9% for the IBJJF banner. The banner reads as mat twenty
times more strongly than the mat does. A single centre learned from a median over
the whole match cannot survive a change of angle and white balance.

The fix is scale, not formula: colour is a legitimate cue at SHOT scale and an
invalid one at match scale. Everything here is fitted per shot.

Three cues were measured on this footage and rejected before landing on the one
below. They are recorded because each is the obvious next idea:

  - union of the two dominant colour clusters (the change that helped
    galvao-xande) floods this frame to 0.959 of its area; the second cluster here
    is dark background at L=50, not the yellow mat border.
  - vanishing-point selection over Hough segments estimates the VP from banners,
    barrier rails and curtain edges, because the mat is the surface with the
    FEWEST edges in frame -- its panel seams are grey-on-blue-grey and Canny does
    not fire on them.
  - largest low-texture region alone climbs from the mat into the banners, the
    black curtain and the referee's black suit, all of which are exactly as flat
    as the mat; the failure is connectivity, not discrimination.

What survives is low-texture to FIND the mat, bounded by the frame's lower half
and by a per-shot colour model grown from that seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Burned-in broadcast graphics
# --------------------------------------------------------------------------
def static_graphics(frames: list[np.ndarray], std_max: float = 6.0) -> np.ndarray:
    """Pixels that never change across the whole match: scoreboard, station logo.

    They are flat and low-texture, so every mat cue below would otherwise happily
    adopt them. Pass frames sampled across the ENTIRE match, not one shot -- the
    overlay is the only thing constant across cuts.
    """
    if not frames:
        raise ValueError("static_graphics needs frames")
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    m = (stack.std(axis=0) < std_max).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)


# --------------------------------------------------------------------------
# The mat, per shot
# --------------------------------------------------------------------------
@dataclass
class MatField:
    """One shot's competition area: where it is, and every colour it takes."""

    hull: np.ndarray          # convex polygon -- use for "is this person on the mat"
    support: np.ndarray       # the actual grown region -- use to sample mat colour
    modes: list               # [(median_lab, mad_lab)] -- the mat is two-tone
    bg: np.ndarray            # temporal-median background of the shot

    def is_mat(self, lab: np.ndarray, k: float = 3.5) -> np.ndarray:
        """Multi-modal membership test.

        Single-mode was the bug behind the athlete seeding picking mat pixels: a
        two-tone mat leaves its non-dominant tone outside the model, that tone
        becomes 'foreground', and a blue mat band is indistinguishable from a blue
        gi. Learning the modes only INSIDE `support` is what makes this safe --
        every cluster found there is genuinely mat, unlike clustering the frame.
        """
        m = np.zeros(lab.shape[:2], dtype=bool)
        for centre, mad in self.modes:
            m |= (np.abs(lab - centre.reshape(1, 1, 3)) / (k * mad.reshape(1, 1, 3)) < 1).all(axis=2)
        return m


def fit_mat(frames: list[np.ndarray], graphics: np.ndarray | None = None,
            n_modes: int = 3, horizon_frac: float = 0.30) -> MatField | None:
    """Learn one shot's mat. Returns None when the shot has no usable mat."""
    if not frames:
        return None
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    grey = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = grey.shape
    if graphics is None:
        graphics = np.zeros((h, w), dtype=bool)

    gx = cv2.Scharr(grey, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(grey, cv2.CV_32F, 0, 1)
    energy = cv2.boxFilter(cv2.magnitude(gx, gy), -1, (25, 25))

    flat = (energy < np.percentile(energy, 50)).astype(np.uint8)
    flat[: int(horizon_frac * h)] = 0        # nothing that high is the floor
    flat[graphics] = 0
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))

    n, lbl, stats, cent = cv2.connectedComponentsWithStats(flat, 8)
    if n <= 1:
        return None
    # lowest-sitting large blob, not merely the largest: the banners are flat too
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA] if cent[i][1] > 0.50 * h else 0)
    seed = lbl == best
    if seed.sum() < 0.02 * h * w:
        return None

    lab = cv2.cvtColor(bg, cv2.COLOR_BGR2Lab).astype(np.float32)
    med = np.median(lab[seed], axis=0)
    mad = np.median(np.abs(lab[seed] - med), axis=0) + 2.0
    grown = ((np.abs(lab - med.reshape(1, 1, 3)) / (3.0 * mad.reshape(1, 1, 3)) < 1)
             .all(axis=2).astype(np.uint8))
    grown[: int(horizon_frac * h)] = 0
    grown[graphics] = 0
    grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    n2, lbl2, stats2, _ = cv2.connectedComponentsWithStats(grown, 8)
    if n2 <= 1:
        return None
    support = lbl2 == 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))

    cnts, _ = cv2.findContours(support.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros((h, w), np.uint8)
    if cnts:
        cv2.fillConvexPoly(hull, cv2.convexHull(np.vstack(cnts)), 1)

    px = lab[support].astype(np.float32)
    if len(px) > 60_000:
        px = px[np.random.default_rng(0).choice(len(px), 60_000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(0)          # kmeans++ seeding is random; a validation run that
                               # reports different areas each time is not evidence
    _, labels, _ = cv2.kmeans(px, n_modes, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    modes = []
    for k in range(n_modes):
        sel = px[labels.ravel() == k]
        if len(sel) < 0.05 * len(px):        # a mode this thin is noise, not a tone
            continue
        c = np.median(sel, axis=0)
        modes.append((c, np.median(np.abs(sel - c), axis=0) + 2.0))
    if not modes:
        return None

    # NOTE: a brightness filter lived here and was removed after being measured
    # wrong. It referenced the BRIGHTEST mode, so on shot 2 a yellow border tone
    # holding 6.0% of the support (L=213.3) set the scale and both blue mat tones
    # holding 94% (L=110.0 and L=119.9) were dropped as "too dark". The mat then
    # fell out of the model, blue mat pixels voted blue-gi, and the seed landed on
    # the frame corner at [1279,719] with a 111k px mask. Any future variant must
    # reference the DOMINANT tone by share, not the brightest one -- and even then
    # it does not save shot 30, whose intruding athlete tone IS the dominant mode
    # at 61.3%. That shot is carried by the multi-frame retry instead.

    return MatField(hull=hull.astype(bool), support=support, modes=modes, bg=bg)


# --------------------------------------------------------------------------
# Finding the athletes without a tracker
# --------------------------------------------------------------------------
# Deliberately no track ids anywhere below. Contact history keyed on track ids is
# what put two seated photographers on the podium at confidence 1.00: the
# athletes are by definition the pair that occludes itself most, occlusion is what
# makes a tracker drop ids, and they fragmented into 6 ids in 5 seconds while the
# static press row held 3 ids for the full window. Any metric keyed on ids rewards
# whoever is NOT grappling.

GI_BLUE = dict(b_max=-20.0, chroma_min=20.0, l_max=170.0)
GI_WHITE = dict(chroma_max=13.0, l_min=150.0)


def mat_confusable_with(mat: MatField, gi: str) -> bool:
    """Would any of this shot's mat tones pass that gi's own colour test?

    Excluding mat colour is mandatory when the mat is blue and so is the gi --
    without it the blue class seeds on the floor. But it is actively harmful when
    the mat is NOT confusable: on shot 30, an extreme close-up where the navy gi
    fills half the frame and barely moves, the athlete enters the temporal median,
    k-means learns him as a mat tone (share 61.3%, L=28.5, against L>=92.7 for
    every legitimate mat tone measured elsewhere), and `is_mat` then deletes 95.4%
    of him. That shot's mat sits at b*=+12 and cannot be mistaken for a blue gi at
    all, so the exclusion buys nothing and costs the athlete.
    """
    for centre, _ in mat.modes:
        L = float(centre[0])
        b = float(centre[2]) - 128.0
        chroma = float(np.hypot(centre[1] - 128.0, centre[2] - 128.0))
        if gi == "blue" and b < GI_BLUE["b_max"] and chroma > GI_BLUE["chroma_min"] and L < GI_BLUE["l_max"]:
            return True
        if gi == "white" and chroma < GI_WHITE["chroma_max"] and L > GI_WHITE["l_min"]:
            return True
    return False


def gi_votes(frame_bgr: np.ndarray, mat: MatField,
             graphics: np.ndarray | None = None,
             exclude_mat: bool | dict[str, bool] = True) -> dict[str, np.ndarray]:
    """Per-pixel 'this is that gi' votes, with mat pixels removed.

    Removing the mat is not optional. The mat is blue and so is one gi: without
    this the blue class seeds on the floor and SAM2 returns 27% of the frame.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.hypot(a, b)
    if isinstance(exclude_mat, bool):
        exclude_mat = {"blue": exclude_mat, "white": exclude_mat}
    not_mat = ~mat.is_mat(lab, k=3.0)
    base = np.ones(L.shape, dtype=bool)
    if graphics is not None:
        base &= ~graphics
    raw = {
        "blue": (b < GI_BLUE["b_max"]) & (chroma > GI_BLUE["chroma_min"]) & (L < GI_BLUE["l_max"]),
        "white": (chroma < GI_WHITE["chroma_max"]) & (L > GI_WHITE["l_min"]),
    }
    return {k: (v & base & (not_mat if exclude_mat.get(k, True) else True))
            for k, v in raw.items()}


def person_support(boxes, shape) -> np.ndarray:
    """Union of detected person boxes, as a mask.

    The independent cross-check that exposed how bad the colour-only seeding was
    is also the cue that fixes it. Colour answers WHICH athlete; a person detector
    answers WHETHER it is a person at all, and it knows nothing about gi tones, so
    the two signals fail in unrelated places. Banners, scoreboards and mat leaks
    are all rejected by construction here -- none of them is a person.

    Use it as a preference, never a veto: measured on this match, the detector
    finds nobody in an extreme close-up where a torso fills half the frame, and
    that is exactly where the colour seeding is at its most reliable.
    """
    m = np.zeros(shape[:2], dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m[max(0, int(y1)):int(y2) + 1, max(0, int(x1)):int(x2) + 1] = True
    return m


def seed_point(votes: np.ndarray, near_mat: np.ndarray | None = None,
               min_px: int = 1500,
               persons: np.ndarray | None = None) -> tuple[np.ndarray | None, int]:
    """One point, deep inside the largest blob of that gi's colour ON the mat.

    ONE point, not six. `segment.scale_candidates` can only reach SAM2's
    part/garment/person disambiguation with a single point -- the gate counts
    positives and negatives together, so six confident pixels plus six mutual
    negatives silently take whatever single mask comes back, and that mask is the
    jacket.

    `near_mat` is not optional in practice. Sponsor banners ring the mat and the
    IBJJF/ZEBRA boards are the same blue as the blue gi: measured on shot 28,
    banner L=63 b*=-21 against gi L=67 b*=-24. No colour gate separates those.
    Only position does, and without this test the banner wins the component for
    being far larger than a distant athlete.
    """
    m = cv2.morphologyEx(votes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)

    def pick(require_person: bool, min_on_person: float = 0.5):
        best, area = None, 0
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a <= area:
                continue
            comp = lbl == i
            if require_person:
                # Mostly inside a detected person, and that is the WHOLE test. Being
                # inside a person is a stronger claim than being near the mat
                # polygon, and the mat test actively rejects athletes: in a wide
                # shot the gi component is the jacket, 300px above the mat surface
                # with only bare feet touching it. Measured, the white gi was
                # 7,099 px at 100% inside a person on shot 2 and 25,627 px at 100%
                # on shot 17, and both were discarded for not reaching the mat --
                # after which the seed fell on the scoreboard and on the yellow
                # border. Demanding a real FRACTION, not any overlap, also drops
                # the "WINNER" caption on shot 36: 24,806 px at 0.3% inside.
                if (comp & persons).sum() < min_on_person * a:
                    continue
            elif near_mat is not None and not (comp & near_mat).any():
                continue                  # no detector to lean on: furniture test
            best, area = i, a
        return best, area

    best, area = (pick(True) if persons is not None and persons.any() else (None, 0))
    on_person = best is not None
    if best is None or area < min_px:      # nobody detected, or nobody wearing it
        best, area = pick(False)
        on_person = False
    if best is None or area < min_px:
        return None, area

    comp = lbl == best
    # Take the deepest point INSIDE the person, not inside the component. This
    # arena's mat panels are bright and near-neutral, so they satisfy the white-gi
    # test and merge with the white gi into one blob; the blob does touch a person
    # box, so it passes the test above, but its distance-transform maximum sits
    # out in the open mat. Measured: six of ten failing shots seeded the white gi
    # on bare mat and three seeded the blue gi on a sponsor banner, all while the
    # component itself was legitimately part-athlete.
    region = (comp & persons) if on_person else comp
    if not region.any():
        region = comp
    dist = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return np.array([[x, y]], dtype=np.float32), area


def dominant_blob(mask: np.ndarray) -> float:
    """Fraction of the mask living in its single largest connected component.

    An athlete is one body. Occlusion can cut him into a couple of pieces -- a
    head above the opponent, legs below -- so this is a fraction, not a demand for
    exactly one component.
    """
    m = mask.astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return 0.0
    total = float(m.sum())
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / max(1.0, total)


def choose_scale(masks: np.ndarray, votes_own: np.ndarray, votes_opp: np.ndarray,
                 max_contam: float = 0.25, min_px: int = 1500, max_frac: float = 0.20):
    """Pick the object scale that is the athlete, not his jacket and not the pair.

    Measured on dalpra-dorsey f13500: scale 0 was 67k px at contamination 0.451
    (it had swallowed both athletes), scale 1 was 18k at 0.002 (the garment),
    scale 2 was 27k at 0.002 (the person). Largest surviving candidate wins.

    `max_frac` is not a tuning knob, it is a physical bound, and contamination
    alone cannot replace it. A mask that has swallowed the MAT contains plenty of
    its own colour and none of the opponent's, so it scores contam ~ 0, passes,
    and wins for being largest. That is how 8 of 37 shots came back with a "white
    athlete" covering 24-50% of the frame. An athlete does not: over the validated
    galvao-xande run, median mask area stayed within 0.072-0.119 of frame.
    """
    frame_px = masks.shape[-2] * masks.shape[-1]
    cands = []
    for i, m in enumerate(masks.astype(bool)):
        px = int(m.sum())
        if px < min_px or px > max_frac * frame_px:
            continue
        own = int((m & votes_own).sum())
        opp = int((m & votes_opp).sum())
        contam = opp / max(1, own + opp)
        if contam <= max_contam:
            cands.append((px, i, m, contam))
    if not cands:
        return None, None, None
    px, i, m, contam = max(cands, key=lambda c: c[0])
    return m, i, contam
