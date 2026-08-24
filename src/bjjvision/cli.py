"""Command line surface. Heavy imports stay lazy so `fetch` and `doctor` run on
the laptop without torch, CUDA, or SAM2 installed."""
from __future__ import annotations

import json
import os
from pathlib import Path

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
def frames(slug: str, data_dir: Path = typer.Option(ROOT / "data")):
    """Explode a normalised match into the JPEG directory SAM2 expects."""
    from .ingest import extract_frames
    video = data_dir / "interim" / f"{slug}_norm.mp4"
    if not video.exists():
        raise typer.BadParameter(f"{video} not found - run `fetch` first")
    out = data_dir / "interim" / f"{slug}_frames"
    n = extract_frames(video, out)
    console.print(f"[green]{n}[/] frames -> {out}")


@app.command()
def run(slug: str, config: Path = typer.Option(DEFAULT_CFG),
        data_dir: Path = typer.Option(ROOT / "data"),
        device: str = typer.Option("cuda"),
        max_frames: int = typer.Option(None, help="cap for smoke tests"),
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

    console.rule("[bold]calibration")
    cal = pipe.calibrate()
    console.print_json(data=cal)
    if "warning" in cal:
        console.print(f"[yellow]{cal['warning']}")

    console.rule("[bold]tracking")
    out_video = out_dir / f"{slug}_analysis.mp4"
    metrics = pipe.run(out_video, max_frames=max_frames)

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
