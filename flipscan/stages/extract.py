"""Stage 2: extract — dump all frames from each source video."""

from __future__ import annotations

from ..ffmpeg import extract_frames
from ..workspace import Workspace


def run(ws: Workspace, cfg: dict, log=print) -> None:
    quality = cfg["extract"]["jpeg_quality"]
    totals = {}
    for video in ws.manifest["videos"]:
        vid = video["id"]
        out_dir = ws.frames_dir(vid)
        expected = video.get("nb_frames")
        existing = sum(1 for _ in out_dir.glob("f*.jpg")) if out_dir.exists() else 0
        if expected and existing >= expected:
            log(f"  {vid}: {existing} frames already extracted, skipping")
            totals[vid] = existing
            continue
        log(f"  {vid}: extracting frames from {video['path']} ...")
        count = extract_frames(ws.root / video["path"], out_dir, jpeg_quality=quality)
        video["frames_extracted"] = count
        totals[vid] = count
        log(f"  {vid}: {count} frames")
    ws.stage_done("extract", frames=totals)
