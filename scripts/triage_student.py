"""The triage loop: run the student over new matches and sort every shot.

Usage:
    python scripts/triage_student.py buchecha-lo [more-slugs ...] [--llm]

For each slug this needs only what `bjj fetch` + `bjj scout` already produce:
data/interim/<slug>_norm.mp4 and <slug>_shots.json. No teacher, no GPU rental,
no extracted frames. Per trackable shot it measures the signals in
bjjvision.triage, writes an evidence grid, and (with --llm) asks the VLM
supervisor for a verdict. Output per slug:

    data/out/<slug>_triage/report.json      signals + flags + verdicts per shot
    data/out/<slug>_triage/shot_NNN.jpg     evidence grid, one per shot

The printed queue at the end is the point: which shots the student can already
label, which must be paid for on the teacher, which are not worth frames at
all. Verdicts follow the project rule -- numbers sort the queue, eyes decide.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjjvision.triage import (SIZE, THRESHOLDS, evidence_grid, load_student,  # noqa: E402
                              measure_shot)

TRACK_KINDS = ("match", "mat")


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def triage_slug(slug: str, model, dev, args, supervisor=None) -> dict:
    data = Path(args.data_dir)
    video = data / "interim" / f"{slug}_norm.mp4"
    shots_json = data / "interim" / f"{slug}_shots.json"
    if not video.exists() or not shots_json.exists():
        raise SystemExit(f"{slug}: precisa de {video.name} e {shots_json.name} "
                         "(rode `bjj fetch` e `bjj scout` antes)")
    out = Path(args.out_root) / f"{slug}_triage"
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads(shots_json.read_text())
    shots = meta["shots"]
    cap = cv2.VideoCapture(str(video))

    rows, t0 = [], time.time()
    for i, s in enumerate(shots):
        if s["kind"] not in TRACK_KINDS or (s["end"] - s["start"]) < args.run_len:
            rows.append({"shot_id": i, "start": s["start"], "end": s["end"],
                         "kind": s["kind"], "verdict": "skip",
                         "why": "kind" if s["kind"] not in TRACK_KINDS else "too_short"})
            continue
        sig, imgs, preds, ids = measure_shot(model, cap, dev, i, s["start"],
                                             s["end"], s["kind"],
                                             n_runs=args.runs, run_len=args.run_len)
        row = sig.to_dict()
        if not imgs:
            row.update(verdict="skip", why="decode_failed")
            rows.append(row)
            continue
        grid = evidence_grid(imgs, preds, ids, sig, n_tiles=args.grid_frames)
        gpath = out / f"shot_{i:03d}.jpg"
        cv2.imwrite(str(gpath), grid)
        row["evidence"] = gpath.name
        # stray_blobs alone means: core masks healthy, defect removable by
        # keeping the largest component -- a cleanup job, not a teacher job.
        row["verdict"] = ("ok" if not sig.flags
                          else "cleanup" if sig.flags == ["stray_blobs"]
                          else "review")

        if supervisor is not None and supervisor.enabled:
            v = supervisor.triage(grid, i, {k: row[k] for k in
                                            ("stability", "core_stability",
                                             "flip_rate", "empty_rate", "margin",
                                             "frag", "stray_frac", "dist", "area",
                                             "border")})
            if v:
                row["llm"] = v
                row["verdict"] = ("student_ok" if v["usable_for_training"]
                                  else "needs_teacher" if v["needs_teacher"]
                                  else "review")
        rows.append(row)
        print(f"  shot {i:3d} f{s['start']}-{s['end']:>6}  "
              f"stab {sig.stability:.2f} marg {sig.margin:.2f} "
              f"-> {row['verdict']:<13} {','.join(sig.flags)}", flush=True)
    cap.release()

    report = {
        "slug": slug, "ckpt": str(args.ckpt), "size": list(SIZE),
        "runs": args.runs, "run_len": args.run_len,
        "thresholds": THRESHOLDS, "elapsed_s": round(time.time() - t0, 1),
        "shots": rows,
    }
    if supervisor is not None:
        report["llm_stats"] = {"calls": supervisor.stats.calls,
                               "est_cost_usd": round(supervisor.stats.est_cost_usd, 3)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def summarise(report: dict) -> None:
    rows = report["shots"]
    frames = lambda r: r["end"] - r["start"]                       # noqa: E731
    total = sum(frames(r) for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r["verdict"], []).append(r)
    print(f"\n=== {report['slug']}: fila de trabalho "
          f"({report['elapsed_s']:.0f}s de triagem) ===")
    for v in ("ok", "student_ok", "cleanup", "review", "needs_teacher", "skip"):
        if v not in by:
            continue
        f = sum(frames(r) for r in by[v])
        ids = ",".join(str(r["shot_id"]) for r in by[v])
        print(f"  {v:<13} {len(by[v]):3d} shots  {f:6d} frames ({f/total:4.0%})  [{ids}]")
    flags = {}
    for r in rows:
        for fl in r.get("flags") or []:
            flags[fl] = flags.get(fl, 0) + 1
    if flags:
        print("  flags: " + "  ".join(f"{k}x{v}" for k, v in
                                      sorted(flags.items(), key=lambda kv: -kv[1])))
    print(f"  evidencias em data/out/{report['slug']}_triage/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--ckpt", default="data/out/student_ckpt_v2/student.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-root", default="data/out")
    ap.add_argument("--runs", type=int, default=3,
                    help="janelas de frames consecutivos por shot")
    ap.add_argument("--run-len", type=int, default=24)
    ap.add_argument("--grid-frames", type=int, default=6)
    ap.add_argument("--llm", action="store_true",
                    help="pede veredito ao supervisor VLM por shot")
    ap.add_argument("--model", default=None, help="override do modelo do supervisor")
    a = ap.parse_args()

    dev = device_auto()
    model = load_student(a.ckpt, dev)
    print(f"device {dev} | student {a.ckpt}")

    supervisor = None
    if a.llm:
        from bjjvision.llm_supervisor import MODEL, LlmSupervisor
        supervisor = LlmSupervisor({"llm": {"enabled": True,
                                            "model": a.model or MODEL,
                                            "max_calls_per_minute": 12}})
        if not supervisor.enabled:
            print("[triage] supervisor indisponivel; seguindo so com sinais")

    for slug in a.slugs:
        print(f"\n--- triando {slug} ---")
        report = triage_slug(slug, model, dev, a, supervisor)
        summarise(report)


if __name__ == "__main__":
    main()
