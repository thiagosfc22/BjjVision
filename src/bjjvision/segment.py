"""SAM2 video segmentation with colour-guided re-anchoring.

Three decisions worth stating, because they are what make SAM2 usable on a
ten-minute grappling match rather than a five-second demo clip:

0. SCALE BEFORE IDENTITY. A point prompt is ambiguous by construction -- a click
   on a thigh can mean the fold, the trousers, or the athlete -- so SAM2 answers
   with three object scales and a quality score each. Decisions 1 and 2 below
   both prompt with more than one label, and that silently suppresses the
   three-way answer (see `scale_candidates`), leaving the model to pick alone.
   Measured on a full match, that is how fighter B's mask came to mean "the gi
   jacket": median area 0.0495 where re-asking at one point gives 0.0932, with
   the two masks' cross-IoU unchanged at 0.0000. So the seed now settles extent
   FIRST, from the object prior, and only then lets colour say whose it is.

1. CHUNKED PROPAGATION. `inference_state` retains per-frame output for every
   frame it has seen, so a single pass over a long match exhausts VRAM. We
   propagate in windows and re-seed at each boundary. The windows double as
   scheduled recalibration points -- the failure mode we care about is gradual,
   so periodic re-grounding is exactly right.

2. NEGATIVE PROMPTS ON THE OPPONENT. When re-anchoring, athlete A is prompted
   with positive points on A *and negative points on B in the same call*. Under
   heavy entanglement "this is A" is a weak constraint -- the boundary between two
   interlocked bodies is genuinely ambiguous from shape alone. Adding "and that
   is explicitly not A" resolves it in one step instead of several.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch

OBJ_IDS = {"A": 1, "B": 2}
ID_OF = {v: k for k, v in OBJ_IDS.items()}


class Sam2Segmenter:
    def __init__(self, cfg: dict, frames_dir: Path, device: str = "cuda"):
        from sam2.build_sam import build_sam2_video_predictor
        s = cfg["segment"]
        self.device = device
        self.frames_dir = frames_dir
        # SAM2 indexes the frame directory 0..N-1 regardless of filename, while the
        # rest of the pipeline speaks absolute video frame numbers. When only a
        # window was extracted (files starting at 00010177.jpg) those two disagree.
        # The offset is absorbed here so no caller has to think about it.
        jpgs = sorted(frames_dir.glob("*.jpg"))
        self.offset = int(jpgs[0].stem) if jpgs else 0
        self.predictor = build_sam2_video_predictor(
            s["sam2_cfg"], s["sam2_ckpt"], device=device)
        self._img_pred = None
        self.state = self.predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=s["offload_video_to_cpu"],
            offload_state_to_cpu=s["offload_state_to_cpu"])
        # bf16 autocast is a CUDA-only win here; on MPS it is unsupported and on
        # CPU it is slower than fp32, so both fall through to a no-op context.
        self._autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                          if device == "cuda" else contextlib.nullcontext())

    # -- prompting ----------------------------------------------------------
    def reset(self) -> None:
        self.predictor.reset_state(self.state)

    def prompt_boxes(self, frame_idx: int, boxes: dict[str, tuple[float, float, float, float]]) -> None:
        with self._autocast, torch.inference_mode():
            for fid, box in boxes.items():
                self.predictor.add_new_points_or_box(
                    inference_state=self.state, frame_idx=frame_idx - self.offset,
                    obj_id=OBJ_IDS[fid], box=np.array(box, dtype=np.float32))

    # -- object-scale disambiguation ----------------------------------------
    @property
    def image_predictor(self):
        """SAM2's single-image head, sharing this segmenter's weights.

        SAM2ImagePredictor takes any SAM2Base, and the video predictor is one,
        so wrapping it costs no second copy of an 898 MB checkpoint.
        """
        if self._img_pred is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self._img_pred = SAM2ImagePredictor(self.predictor)
        return self._img_pred

    def scale_candidates(self, frame_bgr: np.ndarray, point: np.ndarray):
        """The three object scales SAM2 sees at one point: part, garment, person.

        Only reachable with a SINGLE point. The gate in sam2_base._use_multimask
        is `multimask_min_pt_num <= num_pts <= multimask_max_pt_num`, which the
        checkpoint config sets to 0..1, and `num_pts` counts positives AND
        negatives together. So our six colour-confident points plus six mutual
        negatives (twelve labels), and equally a box (two labels, 2 and 3), both
        silently skip the disambiguation and take whatever single mask comes
        back. That is how a jacket ends up standing in for an athlete.
        """
        import cv2
        with self._autocast, torch.inference_mode():
            self.image_predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            masks, scores, _ = self.image_predictor.predict(
                point_coords=np.asarray(point, dtype=np.float32),
                point_labels=np.ones(len(point), dtype=np.int32),
                multimask_output=True)
        return masks.astype(bool), scores

    def prompt_masks(self, frame_idx: int, masks: dict[str, np.ndarray]) -> None:
        """Anchor the window on chosen masks rather than on re-derived points.

        Points make the segmenter re-solve the extent question every seed, and
        it re-solves it from colour-confident pixels, which live on the gi
        jacket. Handing back the mask we already chose keeps the extent decided
        by SAM2's object prior instead.
        """
        with self._autocast, torch.inference_mode():
            for fid, m in masks.items():
                if m is None or not m.any():
                    continue
                self.predictor.add_new_mask(
                    inference_state=self.state, frame_idx=frame_idx - self.offset,
                    obj_id=OBJ_IDS[fid], mask=np.ascontiguousarray(m, dtype=bool))

    def prompt_points(self, frame_idx: int, points: dict[str, np.ndarray],
                      mutual_negatives: bool = True) -> None:
        """Seed each athlete with their own confident pixels, and with the
        opponent's confident pixels marked as background."""
        with self._autocast, torch.inference_mode():
            for fid, pos in points.items():
                if pos is None or len(pos) == 0:
                    continue
                coords = [np.asarray(pos, dtype=np.float32)]
                labels = [np.ones(len(pos), dtype=np.int32)]
                if mutual_negatives:
                    for other, opp in points.items():
                        if other == fid or opp is None or len(opp) == 0:
                            continue
                        coords.append(np.asarray(opp, dtype=np.float32))
                        labels.append(np.zeros(len(opp), dtype=np.int32))
                self.predictor.add_new_points_or_box(
                    inference_state=self.state, frame_idx=frame_idx - self.offset,
                    obj_id=OBJ_IDS[fid],
                    points=np.concatenate(coords, axis=0),
                    labels=np.concatenate(labels, axis=0))

    # -- propagation --------------------------------------------------------
    def propagate(self, start_frame: int, max_frames: int):
        """Yield (frame_idx, {fid: bool mask}, {fid: score}) over one window."""
        with self._autocast, torch.inference_mode():
            for f_idx, obj_ids, logits in self.predictor.propagate_in_video(
                    self.state, start_frame_idx=start_frame - self.offset,
                    max_frame_num_to_track=max_frames):
                masks, scores = {}, {}
                for k, oid in enumerate(obj_ids):
                    fid = ID_OF.get(int(oid))
                    if fid is None:
                        continue
                    lg = logits[k]
                    masks[fid] = (lg > 0.0).cpu().numpy().squeeze().astype(bool)
                    # peak logit is a usable proxy for the segmenter's own certainty
                    scores[fid] = float(torch.sigmoid(lg.max()).cpu())
                yield int(f_idx) + self.offset, masks, scores


def masks_from_boxes_fallback(frame_shape, boxes: dict) -> dict[str, np.ndarray]:
    """Degenerate box masks -- only used if SAM2 returns nothing for a frame, so
    downstream code always has an array to reason about instead of a None."""
    h, w = frame_shape[:2]
    out = {}
    for fid, (x1, y1, x2, y2) in boxes.items():
        m = np.zeros((h, w), dtype=bool)
        m[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = True
        out[fid] = m
    return out
