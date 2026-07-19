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
        direction = spec.get("direction", "forward")
        log(f"{vid}: importing {src}")
        rel = ws.import_video(src, vid)
        meta = probe_video(ws.root / rel)
        log(f"{vid}: {meta['fps_actual']} fps, {meta.get('nb_frames') or '?'} frames")
        entries.append({
            "id": vid, "path": str(rel).replace("\\", "/"), "source": str(src),
            "direction": direction, **meta,
        })
    ws.manifest["videos"] = entries
    ws.save()
    return ws


def add_video(ws: Workspace, src: Path, direction: str = "forward",
              log: Callable[[str], None] = print) -> dict:
    """Add another capture video to an existing project. Pages it shares with
    earlier videos merge (best capture wins); new pages slot into the order.
    Already-transcribed pages whose best frame is unchanged are not re-transcribed."""
    vid = f"v{len(ws.manifest['videos'])}"
    src = Path(src)
    log(f"{vid}: importing {src}")
    rel = ws.import_video(src, vid)
    meta = probe_video(ws.root / rel)
    log(f"{vid}: {meta['fps_actual']} fps, {meta.get('nb_frames') or '?'} frames")
    entry = {
        "id": vid, "path": str(rel).replace("\\", "/"), "source": str(src),
        "direction": direction, **meta,
    }
    ws.manifest["videos"].append(entry)
    ws.stage_reset("extract")  # re-run; per-video skips keep it incremental
    ws.save()
    return entry


def next_page_id(ws: Workspace) -> str:
    nums = [int(p["id"][1:]) for p in ws.manifest["pages"]
            if p["id"].startswith("p") and p["id"][1:].isdigit()]
    return f"p{(max(nums) + 1 if nums else 0):04d}"


def add_page_from_photo(ws: Workspace, cfg: dict, image: Path,
                        position: str = "end", role: str | None = None,
                        log: Callable[[str], None] = print) -> dict:
    """Insert a new page from a photo (cover, inside-cover text, missed page).

    position: "start" | "end" | integer index into the page order.
    role: "cover" marks it as the EPUB cover image (excluded from the body text).
    """
    import shutil

    patches = ws.root / "patches"
    patches.mkdir(exist_ok=True)
    page_id = next_page_id(ws)
    dest = patches / f"{page_id}{Path(image).suffix.lower() or '.jpg'}"
    shutil.copy2(image, dest)

    page = {
        "id": page_id,
        "cluster_frames": [],
        "canonical": None,
        "scores": {},
        "status": "patched",
        "printed_number": None,
        "patched_source": f"patches/{dest.name}",
        "md": None,
    }
    if role:
        page["role"] = role
    if position in ("start", "end"):
        page["pinned"] = position  # survives re-clustering at this end

    pages = ws.manifest["pages"]
    if position == "start":
        idx = 0
    elif position == "end":
        idx = len(pages)
    else:
        idx = max(0, min(len(pages), int(position)))
    pages.insert(idx, page)

    from .stages.preprocess import preprocess_page
    from .stages.transcribe import run as transcribe_run

    log(f"{page_id}: preprocessing photo ({role or 'page'} at position {idx})")
    preprocess_page(ws, page, cfg)
    ws.save()
    if role == "cover":
        # covers are used as an image; no need to burn transcription on them
        page["md"] = None
        ws.save()
    else:
        log(f"{page_id}: transcribing")
        transcribe_run(ws, cfg, log=log)
    ws.stage_reset("figures")
    return page


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
