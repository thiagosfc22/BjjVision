"""SAM2 video segmentation with colour-guided re-anchoring.

Two decisions worth stating, because they are what make SAM2 usable on a
ten-minute grappling match rather than a five-second demo clip:

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
        self.predictor = build_sam2_video_predictor(
            s["sam2_cfg"], s["sam2_ckpt"], device=device)
        self.state = self.predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=s["offload_video_to_cpu"],
            offload_state_to_cpu=s["offload_state_to_cpu"])
        self._autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                          if device == "cuda" else torch.autocast("cpu", enabled=False))

    # -- prompting ----------------------------------------------------------
    def reset(self) -> None:
        self.predictor.reset_state(self.state)

    def prompt_boxes(self, frame_idx: int, boxes: dict[str, tuple[float, float, float, float]]) -> None:
        with self._autocast, torch.inference_mode():
            for fid, box in boxes.items():
                self.predictor.add_new_points_or_box(
                    inference_state=self.state, frame_idx=frame_idx,
                    obj_id=OBJ_IDS[fid], box=np.array(box, dtype=np.float32))

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
                    inference_state=self.state, frame_idx=frame_idx,
                    obj_id=OBJ_IDS[fid],
                    points=np.concatenate(coords, axis=0),
                    labels=np.concatenate(labels, axis=0))

    # -- propagation --------------------------------------------------------
    def propagate(self, start_frame: int, max_frames: int):
        """Yield (frame_idx, {fid: bool mask}, {fid: score}) over one window."""
        with self._autocast, torch.inference_mode():
            for f_idx, obj_ids, logits in self.predictor.propagate_in_video(
                    self.state, start_frame_idx=start_frame,
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
                yield int(f_idx), masks, scores


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
