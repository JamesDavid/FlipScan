"""Thin ffmpeg/ffprobe subprocess wrappers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


class FFmpegNotFound(RuntimeError):
    pass


def _find(tool: str) -> str:
    """Locate ffmpeg/ffprobe: env override, PATH, then the winget links dir."""
    env = os.environ.get(f"FLIPSCAN_{tool.upper()}")
    if env:
        return env
    found = shutil.which(tool)
    if found:
        return found
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        candidate = Path(localapp) / "Microsoft" / "WinGet" / "Links" / f"{tool}.exe"
        if candidate.exists():
            return str(candidate)
    raise FFmpegNotFound(
        f"{tool} not found. Install ffmpeg (winget install Gyan.FFmpeg / apt install ffmpeg) "
        f"or set FLIPSCAN_{tool.upper()} to its path."
    )


def probe_video(path: Path) -> dict:
    """Stream-level metadata. iPhone slow-mo containers lie about playback fps;
    the video *stream* rate is the real capture rate, so read it from the stream."""
    cmd = [
        _find("ffprobe"),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,duration",
        "-of", "json",
        str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]

    def _rate(s: str) -> float:
        try:
            return float(Fraction(s))
        except (ValueError, ZeroDivisionError):
            return 0.0

    r = _rate(stream.get("r_frame_rate", "0/1"))
    avg = _rate(stream.get("avg_frame_rate", "0/1"))
    return {
        "fps_actual": round(max(r, avg), 3),
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration": float(stream["duration"]) if stream.get("duration") else None,
    }


def extract_frames(video: Path, out_dir: Path, jpeg_quality: int = 2) -> int:
    """Dump every frame of `video` as JPEG into out_dir/f%06d.jpg. Returns frame count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _find("ffmpeg"),
        "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-qscale:v", str(jpeg_quality),
        "-fps_mode", "passthrough",  # every stream frame, no dup/drop
        str(out_dir / "f%06d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return sum(1 for _ in out_dir.glob("f*.jpg"))
