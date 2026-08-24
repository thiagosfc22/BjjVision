"""Orchestrator: calibrate -> propagate in windows -> audit every frame -> report.

The windowed structure carries the core claim of the design. Each SAM2 window is
reset and re-seeded from the colour prototypes, so identity continuity never
depends on the segmenter's memory surviving the whole match -- SAM2 only has to
be right for eight seconds at a time. Everything longer than a window is the
colour anchor's responsibility, and colour does not drift when two athletes roll.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import features as feat
from .appearance import build_color_model
from .maskstore import MaskWriter
from .identity import Health, IdentityManager, RecalEvent
from .llm_supervisor import LlmSupervisor
from .render import RenderState, ReportRenderer, TimelineEvent, VideoWriter
from .roles import MatModel, PersonObs, RoleAssigner
from .shots import (Shot, build_shots, classify_shots, detect_cuts,
                    summarise, windows)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def deep_set(cfg: dict, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


class Pipeline:
    def __init__(self, cfg: dict, frames_dir: Path, out_dir: Path,
                 fps: float, device: str = "cuda"):
        self.cfg = cfg
        self.frames_dir = frames_dir
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.device = device
        self.frame_paths = sorted(frames_dir.glob("*.jpg"))
        self.n_frames = len(self.frame_paths)
        self.duration = self.n_frames / max(fps, 1e-6)

        self.identity = IdentityManager(cfg)
        self.supervisor = LlmSupervisor(cfg)
        self.roles = RoleAssigner(cfg)
        self.mat = MatModel()
        self.detector = None
        self.segmenter = None
        self.separability = 0.0
        self.crowd_rejected = 0
        self.last_narration: dict = {}
        self._last_narrate_t = -1e9
        self.shots: list[Shot] = []
        self.shot_summary: dict = {}
        self.passthrough_frames = 0

    def load_shots(self, video_path: Path, shots_json: Path | None = None) -> dict:
        """Segment the broadcast before tracking anything.

        Skipped at your peril: SAM2 propagating across a hard cut produces a
        confident mask of the wrong thing, and the colour audit then has to clean
        up an error that never needed to exist.
        """
        sc = self.cfg.get("shots", {})
        if shots_json and shots_json.exists():
            data = json.loads(shots_json.read_text())
            self.shots = [Shot(start=s["start"], end=s["end"], kind=s["kind"],
                               mat_frac_median=s.get("mat_frac", 0.0),
                               flat_frac_median=s.get("flat_frac", 0.0))
                          for s in data["shots"]]
            self.shot_summary = data.get("summary", {})
        else:
            cuts, diffs, _ = detect_cuts(video_path,
                                         z_threshold=sc.get("z_threshold", 150),
                                         min_shot_frames=sc.get("min_shot_frames", 10))
            self.shots = classify_shots(video_path, build_shots(len(diffs), cuts),
                                        detector=self.detector, mat_model=self.mat,
                                        flat_max=sc.get("flat_max", 0.55))
            self.shot_summary = summarise(self.shots, self.fps)
        return self.shot_summary

    def frame(self, i: int) -> np.ndarray | None:
        if not (0 <= i < self.n_frames):
            return None
        return cv2.imread(str(self.frame_paths[i]))

    # ------------------------------------------------------------------
    def calibrate(self, search_frames: int = 300, n_samples: int = 12) -> dict:
        """Learn the mat, pick the two athletes, and lock in both gi prototypes.

        Everything downstream inherits the quality of this step, so it samples
        widely and only keeps frames where the two masks are cleanly disjoint.
        """
        from .detect import PersonDetector
        from .segment import Sam2Segmenter

        self.detector = PersonDetector(self.cfg, self.device)
        self.segmenter = Sam2Segmenter(self.cfg, self.frames_dir, self.device)

        probe_idx = np.linspace(0, min(search_frames, self.n_frames) - 1,
                                self.cfg["roles"]["mat_estimate_frames"] // 2,
                                dtype=int).tolist()
        probe_frames = [f for f in (self.frame(i) for i in probe_idx) if f is not None]
        self.mat.fit(probe_frames)

        # find a frame where two on-mat people are clearly present and separated
        best = None
        for i in probe_idx:
            fr = self.frame(i)
            if fr is None:
                continue
            persons = self.detector.detect(fr, persist=True)
            self.roles.update(persons)
            dec = self.roles.assign(fr, persons, self.mat.mask(fr), None, None)
            if dec.fighters is None:
                continue
            pa, pb = [p for p in persons if p.track_id in dec.fighters][:2] if len(
                [p for p in persons if p.track_id in dec.fighters]) >= 2 else (None, None)
            if pa is None:
                continue
            from .roles import _box_iou
            sep = 1.0 - _box_iou(pa.box, pb.box)     # prefer them standing apart
            score = sep * (pa.area + pb.area)
            if best is None or score > best[0]:
                best = (score, i, pa, pb)

        if best is None:
            raise RuntimeError("calibration failed: never saw two people on the mat")

        _, seed_idx, pa, pb = best
        seed_frame = self.frame(seed_idx)
        self.segmenter.reset()
        self.segmenter.prompt_boxes(seed_idx, {"A": pa.box, "B": pb.box})

        samples: list[tuple[np.ndarray, dict]] = []
        for f_idx, masks, _ in self.segmenter.propagate(seed_idx, n_samples * 3):
            fr = self.frame(f_idx)
            if fr is None or len(masks) < 2:
                continue
            ma, mb = masks.get("A"), masks.get("B")
            if ma is None or mb is None or not ma.any() or not mb.any():
                continue
            inter = float((ma & mb).sum()) / max(float((ma | mb).sum()), 1.0)
            if inter > 0.05:                 # only clean, disjoint frames calibrate
                continue
            samples.append((fr, {"A": ma, "B": mb}))
            if len(samples) >= n_samples:
                break

        if len(samples) < 3:
            raise RuntimeError(f"calibration failed: only {len(samples)} clean frames")

        self.separability = self.identity.calibrate(samples)
        sw = {k: v.model.swatch_bgr for k, v in self.identity.protos.items()}
        report = {"seed_frame": int(seed_idx), "clean_samples": len(samples),
                  "separability": round(self.separability, 3), "gi_swatches_bgr": sw}

        if self.separability < 0.35:
            report["warning"] = (
                f"gi separability {self.separability:.2f} is low -- the two gis look "
                "alike to the colour model, so identity will lean on motion continuity "
                "and the LLM supervisor. Expect more escalations.")
        return report

    # ------------------------------------------------------------------
    def run(self, out_video: Path, max_frames: int | None = None,
            progress: bool = True, frame_range: tuple[int, int] | None = None) -> dict:
        cfg = self.cfg
        chunk = cfg["video"]["chunk_frames"]
        lo = frame_range[0] if frame_range else 0
        total = min(self.n_frames, frame_range[1] if frame_range else
                    (max_frames or self.n_frames))
        self.frame_lo, self.frame_hi = lo, total

        probe = self.frame(0)
        vh, vw = probe.shape[:2]
        renderer = ReportRenderer(cfg, vw, vh, self.duration)
        labels = {"A": self.identity.protos["A"].label or "Fighter A",
                  "B": self.identity.protos["B"].label or "Fighter B"}
        swatches = {k: v.model.swatch_bgr for k, v in self.identity.protos.items()}
        narrate_every = cfg["llm"]["adjudicate_every_s"]

        raw_out = out_video.with_suffix(".raw.mp4")
        writer = VideoWriter(str(raw_out), self.fps, (renderer.out_w, renderer.out_h))
        self.masks_out = MaskWriter(self.out_dir / "masks", (vh, vw))
        self.rows: list[dict] = []
        t_start = time.time()
        written = 0
        prev_masks: dict[str, np.ndarray] = {}

        # Windows are bounded by shots, never by an arbitrary frame count: SAM2's
        # memory assumes continuity, and a window that spans a cut propagates a
        # mask onto an unrelated camera angle while reporting high confidence.
        if self.shots:
            wins = [(a, min(b, total), f)
                    for a, b, f in windows(self.shots, chunk,
                                           tuple(cfg.get("shots", {}).get(
                                               "track_kinds", ["match", "mat"])))
                    if a < total and b > lo]
            wins = [(max(a, lo), b, f or a < lo) for a, b, f in wins]
        else:
            wins = [(a, min(a + chunk, total), a == lo) for a in range(lo, total, chunk)]

        try:
            out_cursor = lo
            for win_start, win_end, new_shot in wins:
                if win_end <= win_start:
                    continue
                # frames between windows are untracked (close-up, podium, graphics):
                # emit them so the output stays in sync with the source, and say so
                out_cursor = self._passthrough(writer, renderer, out_cursor, win_start,
                                               labels, swatches, st_base=dict(
                                                   duration_s=self.duration,
                                                   separability=self.separability))
                if new_shot:
                    prev_masks = {}          # a cut invalidates the previous masks

                # ---- re-seed this window from the colour prototypes ---------
                self.segmenter.reset()
                seeded = False
                if prev_masks:
                    fr = self.frame(win_start)
                    pts = self.identity.reanchor_prompts(fr, prev_masks,
                                                         cfg["segment"]["prompt_points_per_obj"])
                    if len(pts) == 2:
                        self.segmenter.prompt_points(win_start, pts, mutual_negatives=True)
                        seeded = True
                if not seeded:
                    fr = self.frame(win_start)
                    persons = self.detector.detect(fr, persist=True)
                    self.roles.update(persons)
                    dec = self.roles.assign(fr, persons, self.mat.mask(fr),
                                            (self.identity.protos["A"], self.identity.protos["B"]),
                                            self._colors_of(fr, persons))
                    if dec.fighters is None:
                        out_cursor = self._passthrough(writer, renderer, out_cursor, win_end,
                                                       labels, swatches, st_base=dict(
                                                           duration_s=self.duration,
                                                           separability=self.separability))
                        continue
                    sel = {p.track_id: p for p in persons if p.track_id in dec.fighters}
                    boxes = self._assign_boxes_by_color(fr, list(sel.values()))
                    self.segmenter.prompt_boxes(win_start, boxes)

                # ---- propagate, auditing every frame ------------------------
                cursor = win_start
                budget = win_end - win_start
                while budget > 0:
                    restart_at = None
                    for f_idx, masks, scores in self.segmenter.propagate(cursor, budget):
                        fr = self.frame(f_idx)
                        if fr is None:
                            continue

                        fh = self.identity.audit(f_idx, fr, masks, scores)

                        if fh.state in (Health.SOFT, Health.HARD, Health.ESCALATED):
                            masks = self.identity.soft_repair(fr, masks)
                            renderer.add_event(TimelineEvent(f_idx / self.fps, "recal"))
                            self.identity.events.append(
                                RecalEvent(f_idx, fh.state, list(fh.triggers)))

                        if fh.state is Health.ESCALATED and self.supervisor.enabled:
                            verdict = self.supervisor.adjudicate(
                                fr, masks, f_idx, f_idx / self.fps, fh.triggers, swatches)
                            renderer.add_event(TimelineEvent(f_idx / self.fps, "escalate"))
                            if verdict:
                                if verdict.get("swap"):
                                    renderer.add_event(TimelineEvent(f_idx / self.fps, "swap"))
                                masks = self.identity.apply_llm_verdict(verdict, fr, masks)
                                self.identity.events[-1].llm_verdict = verdict
                                self.identity.events[-1].resolved = not verdict.get("unclear")
                                self.last_narration = {
                                    "position": verdict.get("position", ""),
                                    "dominant": verdict.get("dominant", ""),
                                    "commentary": verdict.get("reasoning", "")}

                        if fh.state is Health.HARD:
                            pts = self.identity.reanchor_prompts(
                                fr, masks, cfg["segment"]["prompt_points_per_obj"])
                            if len(pts) == 2:
                                self.segmenter.prompt_points(f_idx, pts, mutual_negatives=True)
                                restart_at = f_idx           # re-propagate from here
                        else:
                            self.identity.maybe_update_prototypes(fr, masks, fh)

                        # Pose on EVERY frame, not only at window seeds. This is
                        # the deliverable: the mask is what makes an attributed
                        # skeleton possible at all, since a pose estimator alone
                        # cannot say whose limb is whose once the bodies interlock.
                        t_s = f_idx / self.fps
                        persons = self.detector.detect(fr, persist=True)
                        skeletons = feat.attribute_skeletons(
                            [p.keypoints for p in persons if p.keypoints is not None],
                            masks)
                        ff = feat.extract(f_idx, t_s, masks, skeletons)
                        ff.track_confidence = fh.score
                        ff.track_state = fh.state.value
                        ff.shot_kind = next((sh.kind for sh in self.shots
                                             if sh.start <= f_idx < sh.end), "")
                        self.rows.append(ff.to_row(vw, vh))
                        self.masks_out.add(f_idx, masks)

                        if (self.supervisor.enabled and cfg["llm"]["narrate"]
                                and t_s - self._last_narrate_t >= narrate_every):
                            n = self.supervisor.narrate(
                                fr, masks, t_s, self.last_narration.get("position", ""))
                            self._last_narrate_t = t_s
                            if n:
                                self.last_narration = n
                                if n.get("event"):
                                    renderer.add_event(TimelineEvent(t_s, "event", n["event"]))

                        st = RenderState(
                            t_s=t_s, duration_s=self.duration, confidence=fh.score,
                            purity=fh.purity, proto_dist=fh.proto_dist,
                            cross_iou=fh.cross_iou, state=fh.state.value,
                            triggers=fh.triggers, labels=labels, swatches=swatches,
                            position=self.last_narration.get("position", ""),
                            dominant=self.last_narration.get("dominant", ""),
                            commentary=self.last_narration.get("commentary", ""),
                            n_crowd_rejected=self.crowd_rejected,
                            recal_count=len(self.identity.events),
                            escalation_count=self.supervisor.stats.calls,
                            separability=self.separability,
                            fps_proc=written / max(time.time() - t_start, 1e-6))
                        writer.write(renderer.compose(fr, masks, st))
                        written += 1
                        prev_masks = masks

                        if progress and written % 60 == 0:
                            print(f"  {written}/{total} frames  "
                                  f"conf={fh.score:.2f}  state={fh.state.value}  "
                                  f"{st.fps_proc:.1f} fps", flush=True)
                        if restart_at is not None:
                            break

                    if restart_at is None:
                        break
                    budget -= (restart_at - cursor + 1)
                    cursor = restart_at

                out_cursor = max(out_cursor, win_end)
            out_cursor = self._passthrough(writer, renderer, out_cursor, total,
                                           labels, swatches, st_base=dict(
                                               duration_s=self.duration,
                                               separability=self.separability))
        finally:
            writer.close()
            self.masks_path = self.masks_out.close()

        self._encode_final(raw_out, out_video)
        return self._finalise(out_video, time.time() - t_start, written)

    # ------------------------------------------------------------------
    def _passthrough(self, writer, renderer, start: int, end: int,
                     labels: dict, swatches: dict, st_base: dict) -> int:
        """Emit untracked frames unchanged, labelled as untracked.

        Close-ups, podium shots and transition plates do not contain two
        trackable athletes. Running the tracker there would manufacture a second
        competitor out of a cameraman's shoulder; dropping the frames would
        desynchronise the output from the source. So they are passed through and
        the panel says plainly that nothing is being tracked.
        """
        kind_of = {}
        for sh in self.shots:
            for i in (sh.start, sh.end - 1):
                kind_of[i] = sh.kind
        for i in range(start, end):
            fr = self.frame(i)
            if fr is None:
                continue
            shot_kind = next((sh.kind for sh in self.shots if sh.start <= i < sh.end), "untracked")
            st = RenderState(t_s=i / self.fps, confidence=0.0, state="not tracking",
                             labels=labels, swatches=swatches,
                             position="-", commentary=f"no two-athlete view ({shot_kind})",
                             recal_count=len(self.identity.events),
                             escalation_count=self.supervisor.stats.calls, **st_base)
            writer.write(renderer.compose(fr, {}, st))
            self.passthrough_frames += 1
        return end

    def _colors_of(self, frame: np.ndarray, persons: list[PersonObs]) -> dict:
        ap = self.cfg["appearance"]
        out = {}
        for p in persons:
            m = p.mask
            if m is None:
                x1, y1, x2, y2 = (int(v) for v in p.box)
                m = np.zeros(frame.shape[:2], bool)
                m[max(0, y1):y2, max(0, x1):x2] = True
            cm = build_color_model(frame, m, tuple(ap["hist_bins"]),
                                   tuple(ap["torso_band"]) if ap["torso_only"] else None,
                                   ap["min_mask_pixels"])
            if cm:
                out[p.track_id] = cm
        return out

    def _assign_boxes_by_color(self, frame: np.ndarray, persons: list[PersonObs]) -> dict:
        """Bind the two detections to A/B by gi colour, never by detection order.

        Track ids are arbitrary and get reused; the prototype is what makes the
        binding stable across the whole match.
        """
        cols = self._colors_of(frame, persons)
        if len(persons) < 2 or not self.identity.ready:
            return {"A": persons[0].box, "B": persons[-1].box}
        p0, p1 = persons[0], persons[1]
        c0, c1 = cols.get(p0.track_id), cols.get(p1.track_id)
        if c0 is None or c1 is None:
            return {"A": p0.box, "B": p1.box}
        straight = (self.identity.protos["A"].distance(c0)
                    + self.identity.protos["B"].distance(c1))
        crossed = (self.identity.protos["A"].distance(c1)
                   + self.identity.protos["B"].distance(c0))
        return ({"A": p0.box, "B": p1.box} if straight <= crossed
                else {"A": p1.box, "B": p0.box})

    def _encode_final(self, raw: Path, final: Path) -> None:
        """mp4v from OpenCV is not broadly playable -- re-encode to H.264."""
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
             "-c:v", "libx264", "-preset", "medium",
             "-crf", str(self.cfg["render"]["out_crf"]),
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)],
            check=True)
        raw.unlink(missing_ok=True)

    def _finalise(self, out_video: Path, elapsed: float, written: int) -> dict:
        summary = self.identity.summary()
        metrics = {
            "video": str(out_video),
            "frames_written": written,
            "wall_clock_s": round(elapsed, 1),
            "throughput_fps": round(written / max(elapsed, 1e-6), 2),
            "gi_separability": round(self.separability, 3),
            "identity": summary,
            "supervisor": {
                "calls": self.supervisor.stats.calls,
                "swaps_ordered": self.supervisor.stats.swaps_ordered,
                "abstentions": self.supervisor.stats.abstentions,
                "errors": self.supervisor.stats.errors,
                "est_cost_usd": round(self.supervisor.stats.est_cost_usd, 4),
            },
            "recal_events": [
                {"frame": e.frame_idx, "t_s": round(e.frame_idx / self.fps, 2),
                 "kind": e.kind.value, "triggers": e.triggers, "resolved": e.resolved}
                for e in self.identity.events
            ][:400],
        }
        table = feat.write_parquet(
            self.rows, self.out_dir / "features.parquet",
            meta={"fps": self.fps, "frames": len(self.rows),
                  "gi_separability": self.separability,
                  "frame_range": [self.frame_lo, self.frame_hi]})
        attributed = sum(1 for r in self.rows
                         if r.get("A_attributed") and r.get("B_attributed"))
        metrics["features"] = {
            "table": table, "rows": len(self.rows),
            "both_athletes_attributed": attributed,
            "attribution_rate": round(attributed / max(len(self.rows), 1), 4),
            "masks": getattr(self, "masks_path", None),
        }
        (self.out_dir / "report.json").write_text(json.dumps(metrics, indent=2))
        (self.out_dir / "supervisor_log.json").write_text(
            json.dumps(self.supervisor.stats.log, indent=2))

        if self.cfg["llm"].get("tune_between_passes") and self.supervisor.enabled:
            worst = sorted(self.identity.history, key=lambda h: h.score)[:6]
            imgs = [f for f in (self.frame(h.frame_idx) for h in worst) if f is not None]
            tuning = self.supervisor.tune(
                {k: v for k, v in metrics.items() if k != "recal_events"},
                {k: self.cfg[k] for k in ("recalibrate", "appearance", "roles")}, imgs)
            if tuning:
                metrics["tuning_proposal"] = tuning
                (self.out_dir / "tuning_proposal.json").write_text(json.dumps(tuning, indent=2))
                (self.out_dir / "report.json").write_text(json.dumps(metrics, indent=2))
        return metrics
