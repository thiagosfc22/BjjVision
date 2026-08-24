"""On-screen report: masks, identity cards, live diagnostics, event timeline.

Composited beside the video rather than drawn on top of it. An overlaid HUD
occludes exactly the part of the frame you most want to see during a scramble --
which is also precisely when the diagnostics are most worth reading.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

PANEL_W = 400
TIMELINE_H = 96
MIN_PANEL_H = 660   # panel needs this much room; short video gets letterboxed
BG = (22, 22, 26)
FG = (238, 238, 242)
DIM = (140, 140, 150)
ACCENT = {"A": (60, 190, 255), "B": (255, 150, 60)}
OK_C, WARN_C, BAD_C = (110, 220, 130), (70, 200, 250), (80, 90, 245)
F = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, colour=FG, thick=1):
    cv2.putText(img, s, org, F, scale, colour, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, colour):
    cv2.rectangle(img, (x, y), (x + w, y + h), (55, 55, 62), -1)
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(frac, 0, 1))), y + h), colour, -1)


def _grade(v, good, warn):
    return OK_C if v >= good else (WARN_C if v >= warn else BAD_C)


@dataclass
class TimelineEvent:
    t_s: float
    kind: str              # "recal" | "escalate" | "swap" | "event"
    label: str = ""


@dataclass
class RenderState:
    """Everything the HUD needs for one frame."""
    t_s: float = 0.0
    duration_s: float = 0.0
    confidence: float = 0.0
    purity: dict = field(default_factory=dict)
    proto_dist: dict = field(default_factory=dict)
    cross_iou: float = 0.0
    state: str = "healthy"
    triggers: list = field(default_factory=list)
    position: str = ""
    dominant: str = ""
    commentary: str = ""
    labels: dict = field(default_factory=lambda: {"A": "Fighter A", "B": "Fighter B"})
    swatches: dict = field(default_factory=dict)
    referee_visible: bool = False
    n_crowd_rejected: int = 0
    recal_count: int = 0
    escalation_count: int = 0
    separability: float = 0.0
    fps_proc: float = 0.0


class ReportRenderer:
    def __init__(self, cfg: dict, video_w: int, video_h: int, duration_s: float):
        self.alpha = cfg["render"]["mask_alpha"]
        self.show_health = cfg["render"]["show_health"]
        self.show_timeline = cfg["render"]["timeline"]
        self.vw, self.vh = video_w, video_h
        self.duration = max(duration_s, 1e-3)
        # The panel has a fixed information budget. Rather than let it collide with
        # itself on a short source, give it a floor and letterbox the video to match.
        self.panel_h = max(video_h, MIN_PANEL_H)
        self.out_w = video_w + PANEL_W
        self.out_h = self.panel_h + (TIMELINE_H if self.show_timeline else 0)
        self.events: list[TimelineEvent] = []

    def add_event(self, ev: TimelineEvent) -> None:
        self.events.append(ev)

    # -- masks --------------------------------------------------------------
    def draw_masks(self, frame: np.ndarray, masks: dict[str, np.ndarray],
                   referee_mask: np.ndarray | None = None) -> np.ndarray:
        out = frame.copy()
        for fid in ("A", "B"):
            m = masks.get(fid)
            if m is None or not m.any():
                continue
            layer = np.zeros_like(out); layer[:] = ACCENT[fid]
            blend = cv2.addWeighted(out, 1 - self.alpha, layer, self.alpha, 0)
            out[m] = blend[m]
            edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8)).astype(bool)
            out[edge] = ACCENT[fid]
            ys, xs = np.nonzero(m)
            cx, top = int(xs.mean()), int(ys.min())
            cv2.rectangle(out, (cx - 16, top - 30), (cx + 16, top - 6), ACCENT[fid], -1)
            _text(out, fid, (cx - 9, top - 12), 0.7, (20, 20, 24), 2)
        if referee_mask is not None and referee_mask.any():
            edge = cv2.morphologyEx(referee_mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8)).astype(bool)
            out[edge] = DIM
            ys, xs = np.nonzero(referee_mask)
            _text(out, "REF", (int(xs.mean()) - 18, int(ys.min()) - 10), 0.55, DIM, 2)
        return out

    # -- side panel ---------------------------------------------------------
    def _panel(self, st: RenderState) -> np.ndarray:
        p = np.full((self.panel_h, PANEL_W, 3), BG, np.uint8)
        y = 34
        _text(p, "BjjVision", (18, y), 0.78, FG, 2); y += 20
        _text(p, "mask-anchored fighter re-ID", (18, y), 0.38, DIM); y += 26
        cv2.line(p, (18, y), (PANEL_W - 18, y), (55, 55, 62), 1); y += 26

        for fid in ("A", "B"):
            sw = st.swatches.get(fid, (120, 120, 120))
            cv2.rectangle(p, (18, y - 14), (46, y + 10), sw, -1)
            cv2.rectangle(p, (18, y - 14), (46, y + 10), (90, 90, 98), 1)
            _text(p, st.labels.get(fid, fid), (56, y + 2), 0.56, ACCENT[fid], 2)
            y += 28
            pur = st.purity.get(fid, 0.0)
            _text(p, "gi match", (24, y), 0.4, DIM)
            _bar(p, 110, y - 9, 200, 11, pur, _grade(pur, 0.78, 0.62))
            _text(p, f"{pur:.0%}", (320, y), 0.42, _grade(pur, 0.78, 0.62)); y += 22
            pd = 1.0 - st.proto_dist.get(fid, 1.0)
            _text(p, "prototype", (24, y), 0.4, DIM)
            _bar(p, 110, y - 9, 200, 11, pd, _grade(pd, 0.70, 0.55))
            _text(p, f"{pd:.0%}", (320, y), 0.42, _grade(pd, 0.70, 0.55)); y += 30

        cv2.line(p, (18, y - 8), (PANEL_W - 18, y - 8), (55, 55, 62), 1); y += 18
        _text(p, "TRACK CONFIDENCE", (18, y), 0.42, DIM, 1); y += 16
        _bar(p, 18, y - 4, PANEL_W - 100, 16, st.confidence, _grade(st.confidence, 0.75, 0.55))
        _text(p, f"{st.confidence:.0%}", (PANEL_W - 74, y + 9), 0.6,
              _grade(st.confidence, 0.75, 0.55), 2); y += 34

        state_c = {"healthy": OK_C, "soft_repair": WARN_C,
                   "hard_reanchor": WARN_C, "escalated_to_llm": BAD_C}.get(st.state, DIM)
        _text(p, f"state: {st.state}", (18, y), 0.46, state_c, 1); y += 20
        _text(p, "triggers:", (18, y), 0.38, DIM); y += 15
        if st.triggers:
            for t in st.triggers[:4]:
                _text(p, f"  {t}", (18, y), 0.36, BAD_C); y += 13
            if len(st.triggers) > 4:
                _text(p, f"  +{len(st.triggers) - 4} more", (18, y), 0.36, DIM); y += 13
        else:
            _text(p, "  none", (18, y), 0.36, OK_C); y += 13
        y += 10

        if self.show_health:
            cv2.line(p, (18, y - 6), (PANEL_W - 18, y - 6), (55, 55, 62), 1); y += 16
            _text(p, "DIAGNOSTICS", (18, y), 0.42, DIM, 1); y += 18
            rows = [
                ("mask overlap", f"{st.cross_iou:.3f}", _grade(1 - st.cross_iou, 0.9, 0.8)),
                ("gi separability", f"{st.separability:.2f}", _grade(st.separability, 0.5, 0.35)),
                ("recalibrations", str(st.recal_count), FG),
                ("llm escalations", str(st.escalation_count), FG),
                ("crowd rejected", str(st.n_crowd_rejected), DIM),
                ("referee on mat", "yes" if st.referee_visible else "no", DIM),
                ("throughput", f"{st.fps_proc:.1f} fps", DIM),
            ]
            for k, v, c in rows:
                _text(p, k, (24, y), 0.38, DIM)
                _text(p, v, (250, y), 0.40, c); y += 17

        y = max(y + 14, self.panel_h - 118)
        cv2.line(p, (18, y - 14), (PANEL_W - 18, y - 14), (55, 55, 62), 1)
        _text(p, "POSITION", (18, y + 4), 0.42, DIM, 1); y += 24
        _text(p, st.position or "-", (18, y), 0.58, FG, 2); y += 20
        if st.dominant in ("A", "B"):
            _text(p, f"advantage: {st.labels.get(st.dominant, st.dominant)}",
                  (18, y), 0.42, ACCENT[st.dominant]); y += 20
        for line in _wrap(st.commentary, 44)[:3]:
            _text(p, line, (18, y), 0.4, DIM); y += 15
        return p

    # -- timeline -----------------------------------------------------------
    def _timeline(self, st: RenderState) -> np.ndarray:
        t = np.full((TIMELINE_H, self.out_w, 3), (16, 16, 19), np.uint8)
        x0, x1 = 24, self.out_w - 24
        track_y = 46
        cv2.line(t, (x0, track_y), (x1, track_y), (55, 55, 62), 3)

        def x_of(ts: float) -> int:
            return int(x0 + (x1 - x0) * float(np.clip(ts / self.duration, 0, 1)))

        kind_c = {"recal": WARN_C, "escalate": BAD_C, "swap": (240, 120, 240), "event": OK_C}
        for ev in self.events:
            x = x_of(ev.t_s)
            c = kind_c.get(ev.kind, DIM)
            if ev.kind == "escalate":
                cv2.drawMarker(t, (x, track_y), c, cv2.MARKER_TRIANGLE_UP, 12, 2)
            else:
                cv2.circle(t, (x, track_y), 4, c, -1)

        px = x_of(st.t_s)
        cv2.line(t, (px, track_y - 16), (px, track_y + 16), FG, 2)
        _text(t, _hms(st.t_s), (max(x0, px - 26), track_y - 22), 0.42, FG, 1)
        _text(t, "0:00", (x0 - 4, track_y + 30), 0.36, DIM)
        _text(t, _hms(self.duration), (x1 - 34, track_y + 30), 0.36, DIM)

        lx = x0
        for label, c in (("recalibration", WARN_C), ("llm escalation", BAD_C),
                         ("id correction", (240, 120, 240)), ("match event", OK_C)):
            cv2.circle(t, (lx, TIMELINE_H - 14), 4, c, -1)
            _text(t, label, (lx + 10, TIMELINE_H - 10), 0.35, DIM)
            lx += 150
        return t

    # -- compose ------------------------------------------------------------
    def compose(self, frame: np.ndarray, masks: dict[str, np.ndarray],
                st: RenderState, referee_mask: np.ndarray | None = None) -> np.ndarray:
        vid = self.draw_masks(frame, masks, referee_mask)
        if vid.shape[:2] != (self.vh, self.vw):
            vid = cv2.resize(vid, (self.vw, self.vh))
        if self.panel_h > self.vh:
            pad = np.full((self.panel_h, self.vw, 3), (12, 12, 14), np.uint8)
            off = (self.panel_h - self.vh) // 2
            pad[off:off + self.vh] = vid
            vid = pad
        top = np.hstack([vid, self._panel(st)])
        return np.vstack([top, self._timeline(st)]) if self.show_timeline else top


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = (text or "").split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _hms(s: float) -> str:
    return f"{int(s) // 60}:{int(s) % 60:02d}"


class VideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.w.isOpened():
            raise RuntimeError(f"cannot open VideoWriter at {path}")

    def write(self, frame: np.ndarray) -> None:
        self.w.write(frame)

    def close(self) -> None:
        self.w.release()

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
