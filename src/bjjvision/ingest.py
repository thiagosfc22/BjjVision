"""Ingestion: pull match video, normalise it, and expose frames to the pipeline.

Runs LOCALLY (macOS). YouTube blocks datacenter IPs, so the GPU host never
downloads anything -- it receives an already-normalised file over rsync.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


class IngestError(RuntimeError):
    pass


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise IngestError(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return proc


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise IngestError(f"{binary} not found on PATH")
    return path


def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:max_len].strip("-") or "match"


@dataclass
class MatchSource:
    """Everything we know about one downloaded match."""
    slug: str
    url: str
    title: str
    duration_s: float
    raw_path: str
    norm_path: str | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    n_frames: int | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))


def probe(path: Path) -> dict:
    """ffprobe -> {fps, width, height, duration, n_frames}."""
    _require("ffprobe")
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]).stdout
    data = json.loads(out)
    st = (data.get("streams") or [{}])[0]
    num, _, den = (st.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) else 0.0
    duration = float(st.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    return {
        "fps": round(fps, 4),
        "width": int(st.get("width") or 0),
        "height": int(st.get("height") or 0),
        "duration": duration,
        "n_frames": int(nb) if nb and str(nb).isdigit() else int(fps * duration),
    }


def download(url: str, raw_dir: Path, name_hint: str | None = None,
             max_height: int = 1080, cookies_from_browser: str | None = None) -> MatchSource:
    """Fetch one match with yt-dlp. Prefers mp4/h264 so ffmpeg stays cheap."""
    _require("yt-dlp")
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta_cmd = ["yt-dlp", "--no-warnings", "--skip-download", "--dump-single-json"]
    if cookies_from_browser:
        meta_cmd += ["--cookies-from-browser", cookies_from_browser]
    meta = json.loads(_run(meta_cmd + [url]).stdout)

    title = meta.get("title") or "match"
    slug = slugify(name_hint or title)
    target = raw_dir / f"{slug}.%(ext)s"

    fmt = (f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
           f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best")
    dl_cmd = ["yt-dlp", "--no-warnings", "-f", fmt, "--merge-output-format", "mp4",
              "-o", str(target)]
    if cookies_from_browser:
        dl_cmd += ["--cookies-from-browser", cookies_from_browser]
    _run(dl_cmd + [url])

    found = sorted(raw_dir.glob(f"{slug}.*"))
    found = [p for p in found if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
    if not found:
        raise IngestError(f"download produced no video file for slug={slug}")
    raw = found[0]

    info = probe(raw)
    return MatchSource(
        slug=slug, url=url, title=title,
        duration_s=info["duration"], raw_path=str(raw),
        fps=info["fps"], width=info["width"], height=info["height"],
        n_frames=info["n_frames"],
    )


def normalise(src: MatchSource, out_dir: Path, target_fps: int = 30,
              max_width: int = 1280, clip: tuple[float, float] | None = None) -> MatchSource:
    """Constant frame rate, bounded width, no audio.

    CFR matters: SAM2's memory attention assumes uniform temporal spacing, and a
    variable-frame-rate source silently desynchronises frame indices from timestamps.
    """
    _require("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.slug}_norm.mp4"

    vf = f"fps={target_fps},scale='min({max_width},iw)':-2:flags=lanczos"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if clip:
        cmd += ["-ss", f"{clip[0]:.3f}", "-t", f"{clip[1] - clip[0]:.3f}"]
    cmd += ["-i", src.raw_path, "-vf", vf, "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    _run(cmd)

    info = probe(out)
    src.norm_path = str(out)
    src.fps, src.width, src.height = info["fps"], info["width"], info["height"]
    src.n_frames = info["n_frames"]
    return src


def extract_frames(video: Path, frames_dir: Path, quality: int = 2) -> int:
    """SAM2's video predictor wants a directory of zero-padded JPEGs."""
    _require("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
          "-q:v", str(quality), "-start_number", "0",
          str(frames_dir / "%08d.jpg")])
    return len(list(frames_dir.glob("*.jpg")))
