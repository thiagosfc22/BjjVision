"""Person detection + short-horizon tracking (YOLO-pose + ByteTrack).

Pose, not plain boxes: keypoints give the ankle positions that decide mat
membership, and the torso axis that the report uses to call top/bottom position.
Tracking here is only expected to hold across seconds -- long-horizon identity is
the colour anchor's job, not the tracker's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .roles import PersonObs


class PersonDetector:
    def __init__(self, cfg: dict, device: str = "cuda"):
        from ultralytics import YOLO
        d = cfg["detect"]
        self.model = YOLO(d["model"])
        self.conf = d["conf"]
        self.iou = d["iou"]
        self.max_persons = d["max_persons"]
        self.device = device

    def reset_tracker(self) -> None:
        """Drop accumulated track state.

        ByteTrack associates new detections against the tracks it already holds,
        which is right inside a shot and wrong across a discontinuity. Carrying
        state from the calibration window into a tracking window five minutes
        later collapsed 12 detections down to 2 at the first frame, because the
        rest matched nothing and were suppressed -- which then read as "fewer
        than two athletes on the mat" and silently dropped a whole window.
        Call this whenever the frame sequence jumps.
        """
        pred = getattr(self.model, "predictor", None)
        for t in (getattr(pred, "trackers", None) or []):
            if hasattr(t, "reset"):
                t.reset()

    def detect(self, frame_bgr: np.ndarray, persist: bool = True) -> list[PersonObs]:
        res = self.model.track(
            frame_bgr, persist=persist, conf=self.conf, iou=self.iou,
            classes=[0], tracker="bytetrack.yaml", device=self.device, verbose=False,
        )[0]
        if res.boxes is None or res.boxes.shape[0] == 0:
            return []

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None
               else np.arange(len(boxes)))
        kpts = res.keypoints.data.cpu().numpy() if res.keypoints is not None else [None] * len(boxes)

        out = [PersonObs(track_id=int(t), box=tuple(map(float, b)),
                         keypoints=k, score=float(s))
               for b, s, t, k in zip(boxes, scores, ids, kpts)]
        out.sort(key=lambda p: p.area, reverse=True)
        return out[:self.max_persons]


def detect_on_frames(detector: PersonDetector, frames_dir: Path,
                     indices: list[int]) -> dict[int, list[PersonObs]]:
    """Batch helper used by calibration and by LLM escalation sampling."""
    import cv2
    out: dict[int, list[PersonObs]] = {}
    for i in indices:
        p = frames_dir / f"{i:08d}.jpg"
        if not p.exists():
            continue
        frame = cv2.imread(str(p))
        if frame is not None:
            out[i] = detector.detect(frame, persist=False)
    return out
