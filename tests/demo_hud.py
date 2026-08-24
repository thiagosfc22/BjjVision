"""Render the on-screen report against the synthetic scene, with faults injected.

Purpose is to see the HUD under the conditions it exists for -- a swap, a merged
mask, an LLM escalation -- not on a clean clip where every panel reads green.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synth import make_frame

from bjjvision.identity import Health, IdentityManager
from bjjvision.render import RenderState, ReportRenderer, TimelineEvent, VideoWriter

CFG = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
CFG["llm"]["enabled"] = False
FPS, N = 25, 300

SCRIPT = {                       # frame -> injected fault
    "swap": range(120, 150),
    "merge": range(200, 224),
}
NARRATION = [
    (0,   "standing",     "neutral", "Grip fighting from the feet, collar and sleeve."),
    (60,  "closed guard", "B",       "Guard pulled; B closes the legs and controls posture."),
    (120, "scramble",     "unclear", "Sweep attempt turns into an open scramble."),
    (170, "side control", "A",       "A consolidates side control with a crossface."),
    (230, "mount",        "A",       "A steps over to mount and squares the hips."),
]


def main():
    mgr = IdentityManager(CFG)
    sep = mgr.calibrate([(make_frame(i * 0.3, 0.0)[0], make_frame(i * 0.3, 0.0)[1])
                         for i in range(10)])
    print(f"separability = {sep:.3f}")

    img0, _, _ = make_frame(0, 0)
    h, w = img0.shape[:2]
    rend = ReportRenderer(CFG, w, h, N / FPS)
    labels = {"A": "Galvao (white)", "B": "Opponent (navy)"}
    swatches = {k: v.model.swatch_bgr for k, v in mgr.protos.items()}
    out_raw = ROOT / "data" / "out" / "hud_demo.raw.mp4"
    out_raw.parent.mkdir(parents=True, exist_ok=True)

    narration = NARRATION[0]
    with VideoWriter(str(out_raw), FPS, (rend.out_w, rend.out_h)) as wr:
        for i in range(N):
            t = i / FPS
            overlap = float(np.clip(0.15 + 0.8 * np.sin(i / 46.0) ** 2, 0, 0.95))
            img, masks, mref = make_frame(t, overlap)

            if i in SCRIPT["swap"]:
                masks = {"A": masks["B"], "B": masks["A"]}
            if i in SCRIPT["merge"]:
                masks = {"A": masks["A"] | masks["B"], "B": np.zeros_like(masks["A"])}

            fh = mgr.audit(i, img, masks, {"A": 0.9, "B": 0.9})
            if fh.state in (Health.SOFT, Health.HARD, Health.ESCALATED):
                masks = mgr.soft_repair(img, masks)
                rend.add_event(TimelineEvent(t, "recal"))
                if fh.state is Health.ESCALATED:
                    rend.add_event(TimelineEvent(t, "escalate"))
            else:
                mgr.maybe_update_prototypes(img, masks, fh)

            for start, pos, dom, txt in NARRATION:
                if i >= start:
                    narration = (start, pos, dom, txt)

            st = RenderState(
                t_s=t, duration_s=N / FPS, confidence=fh.score,
                purity=fh.purity, proto_dist=fh.proto_dist, cross_iou=fh.cross_iou,
                state=fh.state.value, triggers=fh.triggers,
                labels=labels, swatches=swatches,
                position=narration[1], dominant=narration[2], commentary=narration[3],
                referee_visible=True, n_crowd_rejected=26,
                recal_count=len(mgr.events) + sum(
                    1 for e in rend.events if e.kind == "recal"),
                escalation_count=sum(1 for e in rend.events if e.kind == "escalate"),
                separability=sep, fps_proc=11.4)
            wr.write(rend.compose(img, masks, st, mref))

    final = ROOT / "data" / "out" / "hud_demo.mp4"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(out_raw), "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(final)], check=True)
    out_raw.unlink()

    # a still from the middle of the injected swap, for a quick look
    img, masks, mref = make_frame(120 / FPS, 0.85)
    masks = {"A": masks["B"], "B": masks["A"]}
    fh = mgr.audit(9999, img, masks, {"A": 0.9, "B": 0.9})
    st = RenderState(t_s=120 / FPS, duration_s=N / FPS, confidence=fh.score,
                     purity=fh.purity, proto_dist=fh.proto_dist, cross_iou=fh.cross_iou,
                     state=fh.state.value, triggers=fh.triggers, labels=labels,
                     swatches=swatches, position="scramble", dominant="unclear",
                     commentary="Sweep attempt turns into an open scramble.",
                     referee_visible=True, n_crowd_rejected=26, recal_count=14,
                     escalation_count=2, separability=sep, fps_proc=11.4)
    shot = ROOT / "data" / "out" / "hud_swap_detected.png"
    cv2.imwrite(str(shot), rend.compose(img, masks, st, mref))

    summary = mgr.summary()
    print(f"video  -> {final}  ({final.stat().st_size/1e6:.1f} MB)")
    print(f"still  -> {shot}")
    print(f"mean confidence {summary['mean_confidence']:.2f}, "
          f"states {summary['state_counts']}")


if __name__ == "__main__":
    main()
