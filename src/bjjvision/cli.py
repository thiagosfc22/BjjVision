"""Command line surface. Heavy imports stay lazy so `fetch` and `doctor` run on
the laptop without torch, CUDA, or SAM2 installed."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="BjjVision - fighter ID/re-ID for BJJ video")
console = Console()

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = ROOT / "config" / "default.yaml"


@app.command()
def doctor():
    """Check which half of the pipeline this machine can run."""
    import shutil
    t = Table(title="BjjVision environment", show_header=True)
    t.add_column("component"); t.add_column("status"); t.add_column("detail")

    for b in ("yt-dlp", "ffmpeg", "ffprobe", "rsync", "ssh"):
        p = shutil.which(b)
        t.add_row(b, "[green]ok" if p else "[red]missing", p or "-")
    try:
        import torch
        cuda = torch.cuda.is_available()
        t.add_row("torch", "[green]ok", torch.__version__)
        t.add_row("cuda", "[green]ok" if cuda else "[yellow]cpu only",
                  torch.cuda.get_device_name(0) if cuda else "-")
    except ImportError:
        t.add_row("torch", "[yellow]absent", "local ingest host - expected")
    for mod in ("sam2", "ultralytics", "anthropic"):
        try:
            __import__(mod); t.add_row(mod, "[green]ok", "")
        except ImportError:
            t.add_row(mod, "[yellow]absent", "gpu host only")
    t.add_row("ANTHROPIC_API_KEY", "[green]set" if os.getenv("ANTHROPIC_API_KEY")
              else "[yellow]unset", "supervisor needs a key or `ant auth login`")
    console.print(t)


@app.command()
def fetch(url: str, name: str = typer.Option(None, "--name", "-n"),
          fps: int = typer.Option(30), width: int = typer.Option(1280),
          start: float = typer.Option(None, help="clip start seconds"),
          end: float = typer.Option(None, help="clip end seconds"),
          cookies_from: str = typer.Option(None, "--cookies-from",
                                           help="browser name, if YouTube asks for sign-in"),
          data_dir: Path = typer.Option(ROOT / "data")):
    """Download one match and normalise it. Runs locally -- YouTube blocks datacenter IPs."""
    from .ingest import download, normalise

    console.print(f"[cyan]downloading[/] {url}")
    src = download(url, data_dir / "raw", name, max_height=1080,
                   cookies_from_browser=cookies_from)
    console.print(f"  [green]{src.title}[/]  {src.duration_s:.0f}s  "
                  f"{src.width}x{src.height} @ {src.fps}fps")
    clip = (start, end) if start is not None and end is not None else None
    src = normalise(src, data_dir / "interim", fps, width, clip)
    meta = data_dir / "interim" / f"{src.slug}.json"
    src.save(meta)
    console.print(f"  [green]normalised[/] -> {src.norm_path}")
    console.print(f"  metadata -> {meta}")


@app.command()
def frames(slug: str, data_dir: Path = typer.Option(ROOT / "data"),
           window: str = typer.Option(None, "--frames",
                                      help="extract only START:END (saves disk)")):
    """Explode a normalised match into the JPEG directory SAM2 expects.

    Pass --frames to extract only the window you intend to process. A full match
    is ~3.4 GB of JPEG; a 60-second window is ~340 MB.
    """
    import json as _json
    from .ingest import extract_frames
    video = data_dir / "interim" / f"{slug}_norm.mp4"
    if not video.exists():
        raise typer.BadParameter(f"{video} not found - run `fetch` first")
    meta = data_dir / "interim" / f"{slug}.json"
    fps = _json.loads(meta.read_text())["fps"] if meta.exists() else 30.0

    rng = None
    if window:
        try:
            a, b = window.split(":")
            rng = (int(a), int(b))
        except ValueError:
            raise typer.BadParameter("--frames expects START:END, e.g. 5400:7200")

    out = data_dir / "interim" / f"{slug}_frames"
    n = extract_frames(video, out, frame_range=rng, fps=fps)
    size = sum(f.stat().st_size for f in out.glob("*.jpg")) / 1e9
    console.print(f"[green]{n}[/] frames ({size:.2f} GB) -> {out}")
    if rng:
        console.print(f"  window {rng[0]}-{rng[1]}; filenames keep original indices")


@app.command()
def scout(slug: str, config: Path = typer.Option(DEFAULT_CFG),
          data_dir: Path = typer.Option(ROOT / "data"),
          z: float = typer.Option(None, help="override cut sensitivity")):
    """Map the broadcast structure on CPU, before renting any GPU.

    Broadcast BJJ is cut footage, not a locked-off camera. Knowing how often it
    cuts, and how much of the runtime is close-ups, podium and transition plates,
    decides the propagation window size and how many frames are worth paying to
    process at all.
    """
    import json as _json
    import cv2
    from .pipeline import load_config
    from .roles import MatModel
    from .shots import build_shots, classify_shots, detect_cuts, summarise, windows

    cfg = load_config(config)
    video = data_dir / "interim" / f"{slug}_norm.mp4"
    if not video.exists():
        raise typer.BadParameter(f"{video} not found - run `fetch` first")

    console.print(f"[cyan]scanning[/] {video.name}")
    sh_cfg = cfg.get("shots", {})
    cuts, diffs, fps = detect_cuts(video, z_threshold=z or sh_cfg.get("z_threshold", 150),
                                   min_shot_frames=sh_cfg.get("min_shot_frames", 10))
    n = len(diffs)
    shots = build_shots(n, cuts)

    cap = cv2.VideoCapture(str(video))
    probe = []
    for i in np.linspace(0, n - 1, cfg["roles"]["mat_estimate_frames"], dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            probe.append(fr)
    cap.release()
    mat = MatModel().fit(probe)

    shots = classify_shots(video, shots, detector=None, mat_model=mat,
                           flat_max=sh_cfg.get("flat_max", 0.55))
    summary = summarise(shots, fps)
    wins = windows(shots, cfg["video"]["chunk_frames"],
                   kinds=tuple(sh_cfg.get("track_kinds", ["match", "mat"])))
    trackable = sum(e - s for s, e, _ in wins)

    t = Table(title=f"{slug}: broadcast structure", show_header=True)
    t.add_column("kind"); t.add_column("shots", justify="right")
    t.add_column("seconds", justify="right"); t.add_column("share", justify="right")
    for k, v in sorted(summary["by_kind"].items(), key=lambda kv: -kv[1]["frames"]):
        t.add_row(k, str(v["shots"]), f"{v['seconds']:.0f}", f"{v['frames']/n:.0%}")
    console.print(t)
    console.print(f"  cuts: [bold]{len(cuts)}[/]   median shot: "
                  f"[bold]{summary['median_shot_s']:.1f}s[/]   "
                  f"shortest: {summary['shortest_shot_s']:.1f}s   "
                  f"longest: {summary['longest_shot_s']:.1f}s")
    console.print(f"  propagation windows: [bold]{len(wins)}[/]  covering "
                  f"[bold]{trackable}[/] frames ({trackable/n:.0%} of runtime, "
                  f"{trackable/fps/60:.1f} min)")

    out = data_dir / "interim" / f"{slug}_shots.json"
    out.write_text(_json.dumps({
        "fps": fps, "n_frames": n, "cuts": cuts, "summary": summary,
        "shots": [{"start": s.start, "end": s.end, "kind": s.kind,
                   "mat_frac": round(s.mat_frac_median, 3),
                   "flat_frac": round(s.flat_frac_median, 3)} for s in shots],
    }, indent=2))
    console.print(f"  -> {out}")


@app.command()
def run(slug: str, config: Path = typer.Option(DEFAULT_CFG),
        data_dir: Path = typer.Option(ROOT / "data"),
        device: str = typer.Option("cuda"),
        max_frames: int = typer.Option(None, help="cap for smoke tests"),
        frames: str = typer.Option(None, "--frames",
                                   help="process only START:END, e.g. 5400:7200"),
        no_llm: bool = typer.Option(False, "--no-llm")):
    """Full pipeline. GPU host."""
    from .pipeline import Pipeline, load_config

    cfg = load_config(config)
    if no_llm:
        cfg["llm"]["enabled"] = False

    meta_path = data_dir / "interim" / f"{slug}.json"
    fps = json.loads(meta_path.read_text())["fps"] if meta_path.exists() else cfg["video"]["target_fps"]
    frames_dir = data_dir / "interim" / f"{slug}_frames"
    if not frames_dir.exists():
        raise typer.BadParameter(f"{frames_dir} not found - run `frames` first")

    out_dir = data_dir / "out" / slug
    pipe = Pipeline(cfg, frames_dir, out_dir, fps, device)

    console.rule("[bold]broadcast structure")
    shots_json = data_dir / "interim" / f"{slug}_shots.json"
    video = data_dir / "interim" / f"{slug}_norm.mp4"
    if not shots_json.exists():
        console.print("[yellow]no scout output; detecting cuts now "
                      "(run `bjj scout` beforehand to see this on CPU first)")
    console.print_json(data=pipe.load_shots(video, shots_json if shots_json.exists() else None))

    console.rule("[bold]calibration")
    cal = pipe.calibrate()
    console.print_json(data=cal)
    if "warning" in cal:
        console.print(f"[yellow]{cal['warning']}")

    console.rule("[bold]tracking")
    rng = None
    if frames:
        try:
            a, b = frames.split(":")
            rng = (int(a), int(b))
        except ValueError:
            raise typer.BadParameter("--frames expects START:END, e.g. 5400:7200")
        console.print(f"[cyan]window[/] frames {rng[0]}-{rng[1]} "
                      f"({(rng[1]-rng[0])/fps:.0f}s)")
    out_video = out_dir / f"{slug}_analysis.mp4"
    metrics = pipe.run(out_video, max_frames=max_frames, frame_range=rng)

    console.rule("[bold]summary")
    console.print_json(data={k: v for k, v in metrics.items() if k != "recal_events"})
    console.print(f"\n[green]video[/] {out_video}")


@app.command("sync-up")
def sync_up(slug: str, host: str, remote_dir: str = "~/BjjVision",
            data_dir: Path = typer.Option(ROOT / "data")):
    """Push code + normalised video to the GPU host."""
    import subprocess
    console.print(f"[cyan]-> {host}:{remote_dir}")
    subprocess.run(["rsync", "-avz", "--progress",
                    "--exclude", ".venv", "--exclude", "data", "--exclude", ".git",
                    f"{ROOT}/", f"{host}:{remote_dir}/"], check=True)
    subprocess.run(["ssh", host, f"mkdir -p {remote_dir}/data/interim"], check=True)
    for pat in (f"{slug}_norm.mp4", f"{slug}.json"):
        p = data_dir / "interim" / pat
        if p.exists():
            subprocess.run(["rsync", "-avz", "--progress", str(p),
                            f"{host}:{remote_dir}/data/interim/"], check=True)
    console.print("[green]done")


@app.command("sync-down")
def sync_down(slug: str, host: str, remote_dir: str = "~/BjjVision",
              data_dir: Path = typer.Option(ROOT / "data")):
    """Pull the finished video and reports back to this machine."""
    import subprocess
    dest = data_dir / "out" / slug
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-avz", "--progress",
                    f"{host}:{remote_dir}/data/out/{slug}/", f"{dest}/"], check=True)
    console.print(f"[green]-> {dest}")
    for f in sorted(dest.iterdir()):
        console.print(f"  {f.name}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    app()
