"""The supervisor: a VLM in the loop, used where pixels stop being decisive.

Three distinct jobs, deliberately kept separate because they have different
latency budgets and different failure costs:

  adjudicate() -- called on escalation only. Looks at annotated frames and rules
                  on identity. This is the one that must be right; everything
                  downstream inherits its verdict.
  narrate()    -- called on a slow cadence. Produces the human-readable position
                  and event commentary the on-screen report shows.
  tune()       -- called once per pass. Reads the run's own health telemetry and
                  proposes threshold changes for the next pass. This is what makes
                  the pipeline iterative rather than one-shot.

Cost discipline matters here: the pipeline is designed so the *geometry* handles
the common case and the model is spent only on genuine ambiguity. The stable
system prompt is cached, so escalations pay for the frames, not the instructions.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

MODEL = "claude-opus-5"

SYSTEM = """\
You are the identity supervisor for an automated Brazilian Jiu-Jitsu match \
analysis pipeline. A computer-vision stack tracks two competitors with SAM2 \
segmentation masks and a gi-colour identity model. You are consulted only when \
that stack cannot resolve an ambiguity on its own.

You will be shown annotated video frames. Each frame has the pipeline's current \
mask overlays drawn on it and labelled A and B, plus isolated cut-outs of each \
mask on a neutral background so you can judge colour without distraction.

What you are ruling on:

1. IDENTITY. The two competitors wear different-coloured gis; that is the ground \
truth for identity and it does not change during the match. Decide whether the \
pipeline's A/B labels currently match the reference gi colours you are given. If \
the labels are backwards, say so.

2. ROLE. Referees and cornerpeople appear on the mat. A referee is not a \
competitor. Competitors are in near-continuous physical contact with each other; \
a referee circles them, stays upright, and wears a uniform that contrasts with \
both gis. If a mask has captured the referee instead of a competitor, say so.

3. MASK QUALITY. Under heavy entanglement a mask often bleeds across both bodies. \
If a mask visibly contains parts of both competitors, report it as contaminated \
rather than assigning it to one athlete.

4. POSITION. Name the grappling position in standard terminology (standing, \
closed guard, open guard, half guard, side control, mount, back control, turtle, \
scramble) and say which competitor is in the dominant position.

Reasoning discipline: judge from the gi colour and the body geometry you can \
actually see. Say "unclear" when the frames genuinely do not settle the question \
-- a confident wrong verdict corrupts the prototype bank and costs far more than \
an honest abstention, because the pipeline will then treat the error as its new \
reference. Abstaining is cheap; being wrong is not.\
"""

ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "labels_correct": {"type": "boolean",
                           "description": "True if mask A really is fighter A."},
        "swap": {"type": "boolean",
                 "description": "True if the A/B labels must be exchanged."},
        "unclear": {"type": "boolean",
                    "description": "True if the frames do not settle identity."},
        "contaminated": {"type": "array", "items": {"type": "string", "enum": ["A", "B"]},
                         "description": "Masks that visibly contain both athletes."},
        "referee_in_mask": {"type": "array", "items": {"type": "string", "enum": ["A", "B"]},
                            "description": "Masks that captured the referee."},
        "reset_prototypes": {"type": "boolean",
                             "description": "True if the colour prototypes look stale."},
        "position": {"type": "string",
                     "description": "Grappling position in standard terminology."},
        "dominant": {"type": "string", "enum": ["A", "B", "neutral", "unclear"]},
        "confidence": {"type": "number", "description": "0.0-1.0"},
        "reasoning": {"type": "string", "description": "Two sentences maximum."},
    },
    "required": ["labels_correct", "swap", "unclear", "contaminated", "referee_in_mask",
                 "reset_prototypes", "position", "dominant", "confidence", "reasoning"],
    "additionalProperties": False,
}

NARRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "position": {"type": "string"},
        "dominant": {"type": "string", "enum": ["A", "B", "neutral", "unclear"]},
        "event": {"type": "string",
                  "description": "Notable action in this window, or empty string."},
        "commentary": {"type": "string", "description": "One short sentence for the overlay."},
    },
    "required": ["position", "dominant", "event", "commentary"],
    "additionalProperties": False,
}

TUNING_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "description": "What actually went wrong this pass."},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Dotted config path."},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["path", "from", "to", "rationale"],
                "additionalProperties": False,
            },
        },
        "rerun_recommended": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["diagnosis", "changes", "rerun_recommended", "confidence"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------
def _png_b64(img_bgr: np.ndarray, max_side: int = 900) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _img_block(img_bgr: np.ndarray) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(img_bgr)}}


def build_evidence(frame_bgr: np.ndarray, masks: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Annotated frame + isolated cut-outs.

    The cut-outs matter as much as the overlay: on the composite the model has to
    disentangle two overlapping tinted regions, while on a cut-out against neutral
    grey the gi colour is unmistakable. Showing both lets it check its own reading.
    """
    tint = {"A": (0, 200, 255), "B": (255, 120, 0)}
    over = frame_bgr.copy()
    for fid, m in masks.items():
        if m is None or not m.any():
            continue
        colour = np.zeros_like(frame_bgr); colour[:] = tint[fid]
        over[m] = cv2.addWeighted(over, 0.45, colour, 0.55, 0)[m]
        ys, xs = np.nonzero(m)
        cv2.putText(over, fid, (int(xs.mean()), int(ys.min()) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, tint[fid], 3, cv2.LINE_AA)

    out = [over]
    for fid in ("A", "B"):
        m = masks.get(fid)
        if m is None or not m.any():
            continue
        cut = np.full_like(frame_bgr, 128)
        cut[m] = frame_bgr[m]
        ys, xs = np.nonzero(m)
        pad = 24
        y0, y1 = max(0, ys.min() - pad), min(cut.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(cut.shape[1], xs.max() + pad)
        crop = cut[y0:y1, x0:x1]
        cv2.putText(crop, f"mask {fid}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 2, cv2.LINE_AA)
        out.append(crop)
    return out


@dataclass
class SupervisorStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    swaps_ordered: int = 0
    abstentions: int = 0
    errors: int = 0
    log: list[dict] = field(default_factory=list)

    @property
    def est_cost_usd(self) -> float:
        # claude-opus-5: $5 / 1M in, $25 / 1M out; cached reads ~0.1x input
        return ((self.input_tokens - self.cache_read) * 5.0
                + self.cache_read * 0.5 + self.output_tokens * 25.0) / 1e6


class LlmSupervisor:
    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.enabled = c["enabled"]
        self.model = c.get("model", MODEL)
        self.max_rpm = c["max_calls_per_minute"]
        self.stats = SupervisorStats()
        self._times: list[float] = []
        self.client = None
        if not self.enabled:
            return
        try:
            import anthropic
            self.client = anthropic.Anthropic()   # resolves key or `ant auth login` profile
        except Exception as exc:                  # noqa: BLE001
            print(f"[supervisor] disabled -- Anthropic client unavailable: {exc}")
            self.enabled = False

    def _throttle(self) -> None:
        now = time.time()
        self._times = [t for t in self._times if now - t < 60.0]
        if len(self._times) >= self.max_rpm:
            time.sleep(max(0.0, 60.0 - (now - self._times[0])))
        self._times.append(time.time())

    def _call(self, blocks: list[dict], schema: dict, effort: str = "high") -> dict | None:
        if not self.enabled or self.client is None:
            return None
        self._throttle()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[{"type": "text", "text": SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort,
                               "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": blocks}],
            )
            u = resp.usage
            self.stats.calls += 1
            self.stats.input_tokens += getattr(u, "input_tokens", 0)
            self.stats.output_tokens += getattr(u, "output_tokens", 0)
            self.stats.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
            text = next((b.text for b in resp.content if b.type == "text"), None)
            return json.loads(text) if text else None
        except Exception as exc:                  # noqa: BLE001
            self.stats.errors += 1
            print(f"[supervisor] call failed: {exc}")
            return None

    # -- job 1: adjudicate identity ----------------------------------------
    def adjudicate(self, frame_bgr: np.ndarray, masks: dict[str, np.ndarray],
                   frame_idx: int, t_s: float, triggers: list[str],
                   ref_swatches: dict[str, tuple[int, int, int]] | None = None) -> dict | None:
        imgs = build_evidence(frame_bgr, masks)
        ref = ""
        if ref_swatches:
            ref = ("\nReference gi colours locked in at calibration (BGR): "
                   + ", ".join(f"{k}={v}" for k, v in ref_swatches.items()))
        prompt = (
            f"Frame {frame_idx} (t={t_s:.1f}s). The pipeline flagged this frame.\n"
            f"Automatic health triggers that fired: {', '.join(triggers) or 'none'}.{ref}\n\n"
            "First image: current mask overlay on the frame (A gold, B blue).\n"
            "Following images: each mask isolated on neutral grey.\n\n"
            "Rule on identity, role, and mask quality."
        )
        blocks = [{"type": "text", "text": prompt}] + [_img_block(i) for i in imgs]
        v = self._call(blocks, ADJUDICATION_SCHEMA, effort="high")
        if v:
            if v.get("unclear"):
                self.stats.abstentions += 1
                v["swap"] = False               # never act on an abstention
                v["reset_prototypes"] = False
            if v.get("swap"):
                self.stats.swaps_ordered += 1
            v["frame_idx"] = frame_idx
            self.stats.log.append({"kind": "adjudicate", "frame": frame_idx, **v})
        return v

    # -- job 2: narrate ------------------------------------------------------
    def narrate(self, frame_bgr: np.ndarray, masks: dict[str, np.ndarray],
                t_s: float, last_position: str = "") -> dict | None:
        imgs = build_evidence(frame_bgr, masks)[:1]
        prompt = (f"t={t_s:.0f}s. Previous reported position: "
                  f"{last_position or 'unknown'}.\nDescribe the current position and any "
                  f"notable action. Keep the commentary to one short sentence suitable for "
                  f"an on-screen overlay.")
        blocks = [{"type": "text", "text": prompt}] + [_img_block(i) for i in imgs]
        v = self._call(blocks, NARRATION_SCHEMA, effort="low")   # cheap, runs often
        if v:
            v["t_s"] = t_s
            self.stats.log.append({"kind": "narrate", "t_s": t_s, **v})
        return v

    # -- job 3: tune between passes -----------------------------------------
    def tune(self, metrics: dict, cfg: dict, worst_frames: list[np.ndarray] | None = None) -> dict | None:
        """Read the run's telemetry and propose config changes for the next pass."""
        prompt = (
            "You are reviewing a completed pass of the pipeline and proposing "
            "threshold changes for the next one.\n\n"
            f"Run telemetry:\n{json.dumps(metrics, indent=2)}\n\n"
            f"Current configuration:\n{json.dumps(cfg, indent=2)}\n\n"
            "Diagnose the dominant failure mode and propose concrete changes as dotted "
            "config paths. Propose nothing if the run is healthy -- an unnecessary change "
            "costs a full re-run. Attached (if any) are the lowest-confidence frames."
        )
        blocks = [{"type": "text", "text": prompt}]
        for img in (worst_frames or [])[:6]:
            blocks.append(_img_block(img))
        v = self._call(blocks, TUNING_SCHEMA, effort="high")
        if v:
            self.stats.log.append({"kind": "tune", **v})
        return v
