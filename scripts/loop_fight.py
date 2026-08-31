"""The front door of the fight loop: one command takes a match from URL to queue.

    python scripts/loop_fight.py <slug> --url <youtube-url> [--start S --end E]
    python scripts/loop_fight.py <slug>              # video already fetched

Chains fetch -> scout -> triage and records the result in config/fights.yaml.
Every step is idempotent: whatever already exists on disk is skipped, so
re-running after a failure (or after a threshold change, with --force-triage)
costs only the missing part. Nothing here rents a GPU; the whole chain runs on
the laptop, and the expensive decision -- which shots go to the teacher -- is
exactly what the triage output exists to inform.

--commit stages fights.yaml and commits the measured queue as one fact, in
keeping with the repo's one-commit-per-measurement practice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "fights.yaml"
sys.path.insert(0, str(ROOT / "src"))


def run(cmd: list[str], why: str) -> None:
    print(f"\n=== {why}\n    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def duration_s(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def ensure_norm_complete(slug: str, clipped: bool) -> None:
    """Fail loudly on the silent-truncation failure mode, then heal it.

    gracie-bastos came out of a batch run with HALF the match missing: ffmpeg
    exited 0, produced 248s from a 488s source, and every downstream number
    would have described a video that does not exist. The same command re-run
    by hand produced the full file, so the cause is transient -- which makes
    detection, not diagnosis, the fix. A norm shorter than 95% of its raw is
    re-normalised once; stale scout/triage outputs of the truncated video are
    deleted because they describe it.
    """
    if clipped:
        return                       # a requested clip is SUPPOSED to be shorter
    meta_path = ROOT / "data" / "interim" / f"{slug}.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    raw = Path(meta.get("raw_path") or "")
    norm = ROOT / "data" / "interim" / f"{slug}_norm.mp4"
    if not raw.exists() or not norm.exists():
        return
    raw_d, norm_d = duration_s(raw), duration_s(norm)
    if raw_d <= 0 or norm_d >= 0.95 * raw_d:
        return
    print(f"!! norm truncado: {norm_d:.0f}s de {raw_d:.0f}s -- renormalizando")
    from bjjvision.ingest import MatchSource, normalise
    src = normalise(MatchSource(**meta), ROOT / "data" / "interim")
    src.save(meta_path)
    (ROOT / "data" / "interim" / f"{slug}_shots.json").unlink(missing_ok=True)
    import shutil
    shutil.rmtree(ROOT / "data" / "out" / f"{slug}_triage", ignore_errors=True)
    norm_d = duration_s(norm)
    if norm_d < 0.95 * duration_s(raw):
        raise SystemExit(f"{slug}: norm segue truncado ({norm_d:.0f}s) apos "
                         "renormalizar -- investigar antes de continuar")


def triage_summary(slug: str) -> dict:
    report = json.loads((ROOT / "data" / "out" / f"{slug}_triage" / "report.json").read_text())
    rows = report["shots"]
    total = sum(r["end"] - r["start"] for r in rows) or 1
    frames = {}
    for r in rows:
        frames[r["verdict"]] = frames.get(r["verdict"], 0) + (r["end"] - r["start"])
    usable = frames.get("ok", 0) + frames.get("cleanup", 0) + frames.get("student_ok", 0)
    return {
        "date": dt.date.today().isoformat(),
        "usable_pct": round(100 * usable / total, 1),
        "frames_pct": {k: round(100 * v / total, 1) for k, v in sorted(frames.items())},
        "review_shots": [r["shot_id"] for r in rows
                         if r["verdict"] in ("review", "needs_teacher")],
    }


def update_registry(slug: str, summary: dict) -> None:
    """Set the fight's status to triaged, preserving the doctrine header.

    yaml.safe_dump would strip the held-out-per-fight comment block, which is
    the whole reason the file exists -- so everything above `fights:` is kept
    verbatim and only the list below it is re-rendered.
    """
    text = REGISTRY.read_text()
    head = text.split("\nfights:")[0]
    fights = yaml.safe_load(text)["fights"]
    entry = next((f for f in fights if f["slug"] == slug), None)
    if entry is None:
        entry = {"slug": slug, "role": "TBD"}
        fights.append(entry)
    entry["status"] = "triaged"
    entry["triage"] = summary
    body = yaml.safe_dump({"fights": fights}, sort_keys=False, allow_unicode=True)
    REGISTRY.write_text(head + "\n" + body)
    print(f"registro atualizado: {slug} -> triaged ({summary['usable_pct']}% aproveitavel)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--url", default=None, help="YouTube URL; dispensavel se ja baixou")
    ap.add_argument("--start", type=float, default=None, help="corte inicial em segundos")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--cookies-from", default=None,
                    help="navegador, se o YouTube pedir login")
    ap.add_argument("--force-triage", action="store_true",
                    help="re-tria mesmo com report existente (thresholds mudaram)")
    ap.add_argument("--commit", action="store_true",
                    help="commita fights.yaml com a fila medida")
    a = ap.parse_args()

    interim = ROOT / "data" / "interim"
    norm = interim / f"{a.slug}_norm.mp4"
    shots = interim / f"{a.slug}_shots.json"
    report = ROOT / "data" / "out" / f"{a.slug}_triage" / "report.json"

    if not norm.exists():
        if not a.url:
            raise SystemExit(f"{norm.name} nao existe e --url nao foi passado")
        cmd = [str(ROOT / "bjj"), "fetch", a.url, "--name", a.slug]
        if a.start is not None and a.end is not None:
            cmd += ["--start", str(a.start), "--end", str(a.end)]
        if a.cookies_from:
            cmd += ["--cookies-from", a.cookies_from]
        run(cmd, f"fetch {a.slug}")
    else:
        print(f"fetch: {norm.name} ja existe, pulando")

    ensure_norm_complete(a.slug, clipped=a.start is not None and a.end is not None)

    if not shots.exists():
        run([str(ROOT / "bjj"), "scout", a.slug], f"scout {a.slug}")
    else:
        print(f"scout: {shots.name} ja existe, pulando")

    if a.force_triage or not report.exists():
        run([sys.executable, str(ROOT / "scripts" / "triage_student.py"), a.slug],
            f"triagem {a.slug}")
    else:
        print(f"triagem: {report} ja existe, pulando (--force-triage para refazer)")

    summary = triage_summary(a.slug)
    update_registry(a.slug, summary)

    if a.commit:
        pct = summary["frames_pct"]
        msg = (f"triagem {a.slug}: {summary['usable_pct']}% aproveitavel "
               f"({', '.join(f'{k} {v}%' for k, v in pct.items())}); "
               f"review shots {summary['review_shots'] or 'nenhum'}\n\n"
               "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
        subprocess.run(["git", "add", str(REGISTRY)], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        print("fato commitado.")


if __name__ == "__main__":
    main()
