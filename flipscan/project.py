"""Shared project operations used by both the CLI and the GUI."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

from .ffmpeg import probe_video
from .workspace import STAGES, Workspace

STAGE_MODULES = {
    "extract": "flipscan.stages.extract",
    "score": "flipscan.stages.score",
    "cluster": "flipscan.stages.cluster",
    "select": "flipscan.stages.select",
    "preprocess": "flipscan.stages.preprocess",
    "transcribe": "flipscan.stages.transcribe",
    "figures": "flipscan.stages.figures",
    "assemble": "flipscan.stages.assemble",
}


def create_project(directory: Path, videos: list[dict[str, Any]],
                   title: str | None = None, expected_pages: int | None = None,
                   log: Callable[[str], None] = print) -> Workspace:
    """Create a workspace from video specs: [{path, pages, direction}, ...]."""
    ws = Workspace.create(directory, videos=[], title=title,
                          expected_pages=expected_pages)
    entries = []
    for i, spec in enumerate(videos):
        vid = f"v{i}"
        src = Path(spec["path"])
        pages = spec.get("pages", "all")
        direction = spec.get("direction", "forward")
        log(f"{vid}: importing {src} (pages={pages}, direction={direction})")
        rel = ws.import_video(src, vid)
        meta = probe_video(ws.root / rel)
        log(f"{vid}: {meta['fps_actual']} fps, {meta.get('nb_frames') or '?'} frames")
        entries.append({
            "id": vid, "path": str(rel).replace("\\", "/"), "source": str(src),
            "pages": pages, "direction": direction, **meta,
        })
    ws.manifest["videos"] = entries
    ws.save()
    return ws


def run_pipeline(ws: Workspace, cfg: dict, only_stage: str | None = None,
                 force: bool = False, log: Callable[[str], None] = print) -> None:
    """Execute pipeline stages in order, resuming where it left off."""
    stages = [only_stage] if only_stage else STAGES
    for stage in stages:
        try:
            mod = importlib.import_module(STAGE_MODULES[stage])
        except ModuleNotFoundError:
            log(f"[{stage}] not implemented yet, skipping")
            continue
        if not only_stage and not force and ws.stage_status(stage) == "done":
            log(f"[{stage}] done, skipping")
            continue
        log(f"[{stage}] running")
        mod.run(ws, cfg, log=log)
        log(f"[{stage}] ok")
