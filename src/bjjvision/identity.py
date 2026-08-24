"""Identity state machine: continuous self-audit and recalibration.

A tracker that only ever propagates forward will, in a match with this much
occlusion, eventually swap the athletes and then stay confidently wrong for
minutes. The fix is to treat every frame as a *hypothesis to be audited* rather
than a result to be trusted.

Health is measured on five independent axes, and the interesting property is
that they fail at different times:

  purity      -- fraction of a mask's pixels voting for its own gi. Degrades
                 FIRST, while the mask still looks anatomically fine. Earliest warning.
  proto_dist  -- distance from the mask's colour signature to the prototype.
                 Catches slow drift that purity's hard argmax can mask.
  cross_iou   -- overlap between the two fighter masks. Fires on outright bleed.
  area_jump   -- frame-to-frame mask area ratio. Fires on collapse or explosion.
  sam2_score  -- the segmenter's own confidence. Honest but late, and it does not
                 know about identity at all -- only about "is this an object".

Escalation ladder, cheapest first:
  HEALTHY -> SOFT (repair the mask in-place from the colour posterior)
          -> HARD (re-prompt SAM2 from colour-confident points, reset memory)
          -> ESCALATED (hand the ambiguity to the LLM supervisor)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .appearance import (ColorModel, FighterPrototype, PixelClassifier,
                         build_color_model, lab_of)


class Health(Enum):
    HEALTHY = "healthy"
    SOFT = "soft_repair"
    HARD = "hard_reanchor"
    ESCALATED = "escalated_to_llm"
    LOST = "lost"


@dataclass
class FrameHealth:
    frame_idx: int
    purity: dict[str, float] = field(default_factory=dict)
    proto_dist: dict[str, float] = field(default_factory=dict)
    area_jump: dict[str, float] = field(default_factory=dict)
    sam2_score: dict[str, float] = field(default_factory=dict)
    cross_iou: float = 0.0
    state: Health = Health.HEALTHY
    triggers: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Single 0-1 confidence for the HUD. Purity dominates because it is the
        axis that actually tracks identity rather than segmentation quality."""
        if not self.purity:
            return 0.0
        pur = float(np.mean(list(self.purity.values())))
        dist = float(np.mean(list(self.proto_dist.values()))) if self.proto_dist else 0.5
        return float(np.clip(0.62 * pur + 0.26 * (1.0 - dist) + 0.12 * (1.0 - self.cross_iou), 0, 1))


@dataclass
class RecalEvent:
    frame_idx: int
    kind: Health
    triggers: list[str]
    resolved: bool = False
    note: str = ""
    llm_verdict: dict | None = None


class IdentityManager:
    """Owns the two prototypes, audits every frame, and drives recalibration."""

    def __init__(self, cfg: dict):
        rc = cfg["recalibrate"]
        ap = cfg["appearance"]
        self.cfg = cfg
        self.purity_min = rc["purity_min"]
        self.proto_dist_max = rc["proto_dist_max"]
        self.mask_iou_max = rc["mask_iou_max"]
        self.area_jump_max = rc["area_jump_max"]
        self.sam2_score_min = rc["sam2_score_min"]
        self.cooldown = rc["cooldown_frames"]
        self.max_consecutive = rc["max_consecutive"]
        self.proto_update_conf = ap["prototype_update_conf"]
        self.bins = tuple(ap["hist_bins"])
        self.band = tuple(ap["torso_band"]) if ap["torso_only"] else None
        self.min_px = ap["min_mask_pixels"]

        self.protos = {
            "A": FighterPrototype("A", capacity=ap["prototype_bank"]),
            "B": FighterPrototype("B", capacity=ap["prototype_bank"]),
        }
        self.classifier: PixelClassifier | None = None
        self.events: list[RecalEvent] = []
        self.history: list[FrameHealth] = []
        self._prev_area: dict[str, float] = {}
        self._last_recal = -10 ** 9
        self._consecutive = 0
        self.rng = np.random.default_rng(1234)

    # -- calibration --------------------------------------------------------
    def calibrate(self, samples: list[tuple[np.ndarray, dict[str, np.ndarray]]]) -> float:
        """Seed both prototypes from frames where the athletes are cleanly separated.

        Returns gi separability. Call this on the opening seconds (hand slap /
        stand-up) where masks are unambiguous -- everything downstream inherits
        the quality of this step, so it is worth spending frames on.
        """
        for frame, masks in samples:
            for fid, mask in masks.items():
                cm = build_color_model(frame, mask, self.bins, self.band, self.min_px)
                if cm is not None:
                    self.protos[fid].add(cm)
        self._rebuild_classifier()
        from .appearance import separability
        return separability(self.protos["A"], self.protos["B"])

    def _rebuild_classifier(self) -> None:
        if all(p.ready for p in self.protos.values()):
            self.classifier = PixelClassifier(self.protos["A"], self.protos["B"])

    @property
    def ready(self) -> bool:
        return self.classifier is not None

    # -- per-frame audit ----------------------------------------------------
    def audit(self, frame_idx: int, frame_bgr: np.ndarray,
              masks: dict[str, np.ndarray],
              sam2_scores: dict[str, float] | None = None) -> FrameHealth:
        fh = FrameHealth(frame_idx=frame_idx)
        if not self.ready:
            fh.state = Health.HEALTHY
            self.history.append(fh)
            return fh

        idx_of = {"A": 0, "B": 1}
        lab = lab_of(frame_bgr)          # once per frame, not once per lookup
        for fid, mask in masks.items():
            if mask is None or not mask.any():
                fh.purity[fid] = 0.0
                fh.proto_dist[fid] = 1.0
                fh.triggers.append(f"{fid}:empty_mask")
                continue

            fh.purity[fid] = self.classifier.purity(frame_bgr, mask, idx_of[fid], lab)
            cm = build_color_model(frame_bgr, mask, self.bins, self.band, self.min_px, lab)
            fh.proto_dist[fid] = self.protos[fid].distance(cm) if cm else 1.0

            area = float(mask.sum())
            prev = self._prev_area.get(fid)
            fh.area_jump[fid] = max(area / prev, prev / area) if prev and area else 1.0
            self._prev_area[fid] = area

            if sam2_scores:
                fh.sam2_score[fid] = sam2_scores.get(fid, 1.0)

            if fh.purity[fid] < self.purity_min:
                fh.triggers.append(f"{fid}:purity={fh.purity[fid]:.2f}")
            if fh.proto_dist[fid] > self.proto_dist_max:
                fh.triggers.append(f"{fid}:proto_dist={fh.proto_dist[fid]:.2f}")
            if fh.area_jump[fid] > self.area_jump_max:
                fh.triggers.append(f"{fid}:area_jump={fh.area_jump[fid]:.1f}x")
            if sam2_scores and fh.sam2_score.get(fid, 1.0) < self.sam2_score_min:
                fh.triggers.append(f"{fid}:sam2={fh.sam2_score[fid]:.2f}")

        ma, mb = masks.get("A"), masks.get("B")
        if ma is not None and mb is not None and ma.any() and mb.any():
            inter = float((ma & mb).sum())
            union = float((ma | mb).sum())
            fh.cross_iou = inter / union if union else 0.0
            if fh.cross_iou > self.mask_iou_max:
                fh.triggers.append(f"cross_iou={fh.cross_iou:.2f}")

        # both masks claiming the same athlete is the unambiguous swap signature
        if fh.purity and len(fh.purity) == 2:
            pa, pb = fh.purity.get("A", 0), fh.purity.get("B", 0)
            if pa < 0.5 and pb < 0.5:
                fh.triggers.append("both_masks_disown_their_id")

        fh.state = self._decide(frame_idx, fh)
        self.history.append(fh)
        return fh

    def _decide(self, frame_idx: int, fh: FrameHealth) -> Health:
        if not fh.triggers:
            self._consecutive = 0
            return Health.HEALTHY
        if frame_idx - self._last_recal < self.cooldown:
            return Health.SOFT           # inside cooldown, repair but do not re-anchor
        self._last_recal = frame_idx
        self._consecutive += 1
        if self._consecutive >= self.max_consecutive:
            return Health.ESCALATED
        severe = any(t.startswith(("cross_iou", "both_masks")) for t in fh.triggers) or \
                 any("empty_mask" in t for t in fh.triggers)
        return Health.HARD if severe else Health.SOFT

    # -- repair -------------------------------------------------------------
    def soft_repair(self, frame_bgr: np.ndarray,
                    masks: dict[str, np.ndarray],
                    lab: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Re-partition the union of both masks by colour posterior.

        No segmenter call: we already know the pixels belong to *someone*, we are
        only disputing which athlete. Runs in ~2ms, so it is affordable every frame.
        """
        if not self.ready:
            return masks
        ma = masks.get("A", np.zeros(frame_bgr.shape[:2], bool))
        mb = masks.get("B", np.zeros(frame_bgr.shape[:2], bool))
        union = ma.astype(bool) | mb.astype(bool)
        if not union.any():
            return masks
        a_new, b_new = self.classifier.split(frame_bgr, union, lab=lab)
        if not a_new.any() or not b_new.any():
            return masks                 # refuse to hand back a degenerate split
        return {"A": a_new, "B": b_new}

    def reanchor_prompts(self, frame_bgr: np.ndarray,
                         masks: dict[str, np.ndarray], k: int = 6,
                         lab: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Colour-confident point prompts for a SAM2 re-seed.

        Deliberately NOT box centres: in a tangle the centre of A's box often lands
        on B's body, which re-seeds the exact error we are trying to clear.
        """
        if not self.ready:
            return {}
        lab = lab_of(frame_bgr) if lab is None else lab
        out: dict[str, np.ndarray] = {}
        union = np.zeros(frame_bgr.shape[:2], bool)
        for m in masks.values():
            if m is not None:
                union |= m.astype(bool)
        a_px, b_px = self.classifier.split(frame_bgr, union, lab=lab)
        for fid, src, idx in (("A", a_px, 0), ("B", b_px, 1)):
            if src.any():
                out[fid] = self.classifier.sample_prompt_points(
                    frame_bgr, src, idx, k, self.rng, lab)
        return out

    # -- prototype maintenance ---------------------------------------------
    def maybe_update_prototypes(self, frame_bgr: np.ndarray,
                                masks: dict[str, np.ndarray], fh: FrameHealth) -> None:
        """Admit new exemplars only from frames we are confident about.

        This gate is the anti-drift mechanism. Updating on every frame would let a
        single bad frame poison the prototype, after which the *next* bad frame
        looks consistent and the error becomes self-reinforcing. The prototype has
        to be harder to move than the thing it is correcting.
        """
        if fh.state is not Health.HEALTHY or fh.score < self.proto_update_conf:
            return
        lab = lab_of(frame_bgr)
        for fid, mask in masks.items():
            if mask is None or not mask.any():
                continue
            if fh.purity.get(fid, 0) < 0.85:
                continue
            cm = build_color_model(frame_bgr, mask, self.bins, self.band, self.min_px, lab)
            if cm is not None:
                self.protos[fid].add(cm)
        self._rebuild_classifier()

    def apply_llm_verdict(self, verdict: dict, frame_bgr: np.ndarray,
                          masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Act on the supervisor's ruling: optionally swap, then re-seed prototypes."""
        out = masks
        if verdict.get("swap"):
            out = {"A": masks.get("B"), "B": masks.get("A")}
        if verdict.get("reset_prototypes"):
            for fid in ("A", "B"):
                m = out.get(fid)
                if m is None or not m.any():
                    continue
                cm = build_color_model(frame_bgr, m, self.bins, self.band, self.min_px)
                if cm is not None:
                    self.protos[fid] = FighterPrototype(
                        fid, label=self.protos[fid].label,
                        capacity=self.cfg["appearance"]["prototype_bank"])
                    for _ in range(3):
                        self.protos[fid].add(cm)
            self._rebuild_classifier()
        self._consecutive = 0
        return out

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict:
        if not self.history:
            return {}
        scores = np.array([h.score for h in self.history], dtype=np.float32)
        by_state: dict[str, int] = {}
        for h in self.history:
            by_state[h.state.value] = by_state.get(h.state.value, 0) + 1
        return {
            "frames": len(self.history),
            "mean_confidence": float(scores.mean()),
            "p10_confidence": float(np.percentile(scores, 10)),
            "frames_below_0.6": int((scores < 0.6).sum()),
            "state_counts": by_state,
            "recal_events": len(self.events),
            "escalations": sum(1 for e in self.events if e.kind is Health.ESCALATED),
            "prototype_sizes": {k: len(v.exemplars) for k, v in self.protos.items()},
        }
