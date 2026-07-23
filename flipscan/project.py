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
                   book: dict[str, Any] | None = None,
                   log: Callable[[str], None] = print) -> Workspace:
    """Create a workspace from video specs: [{path, pages, direction}, ...].
    `book` carries optional metadata (author, isbn, publisher, year)."""
    ws = Workspace.create(directory, videos=[], title=title,
                          expected_pages=expected_pages, book=book)
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
              rotate: int = 0, log: Callable[[str], None] = print) -> dict:
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
        "direction": direction, "rotate": rotate, **meta,
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
                        transcribe: bool = True,
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
    elif transcribe:
        log(f"{page_id}: transcribing")
        transcribe_run(ws, cfg, log=log)
    else:
        ws.save()
    # deferred pages must actually get transcribed on the next run
    ws.stage_reset("figures" if transcribe or role == "cover" else "transcribe")
    return page


def set_video_rotation(ws: Workspace, vid: str, rotate: int,
                       log: Callable[[str], None] = print) -> None:
    """Change a video's orientation and invalidate everything derived from it."""
    video = next(v for v in ws.manifest["videos"] if v["id"] == vid)
    if video.get("rotate", None) == rotate:
        video["rotate"] = rotate
        ws.save()
        return
    video["rotate"] = rotate
    from .stages.transcribe import load_cache, save_cache
    cache = {k: v for k, v in load_cache(ws).items() if not k.startswith(vid + "_")}
    save_cache(ws, cache)
    for p in ws.manifest["pages"]:
        if (p.get("canonical") or "").startswith(vid + "_"):
            p["md"] = None
            for key in ("confidence", "flags", "transcribe_error", "printed_number"):
                p.pop(key, None)
    ws.save()
    log(f"{vid}: orientation set to {rotate} degrees")


def add_pages_from_pdf(ws: Workspace, cfg: dict, pdf: Path,
                       log: Callable[[str], None] = print) -> int:
    """Start (or extend) a book from a PDF: every PDF page is rendered to an
    image and becomes a photo-sourced page, used exactly as-is — the video
    stages are skipped entirely. PDF page order is the book order."""
    import cv2
    import numpy as np
    import pypdfium2 as pdfium

    from .stages.preprocess import preprocess_page

    doc = pdfium.PdfDocument(str(pdf))
    patches = ws.root / "patches"
    patches.mkdir(exist_ok=True)
    pages = ws.manifest["pages"]
    try:
        n = len(doc)
        for i in range(n):
            pg = doc[i]
            w, h = pg.get_size()
            scale = 2200.0 / max(w, h)     # ~200 DPI for a trade book
            arr = pg.render(scale=scale).to_numpy()
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            page_id = next_page_id(ws)
            dest = patches / f"{page_id}.png"
            cv2.imwrite(str(dest), arr)
            page = {
                "id": page_id,
                "cluster_frames": [],
                "canonical": None,
                "scores": {},
                "status": "patched",
                "printed_number": None,
                "patched_source": f"patches/{dest.name}",
                "source": "pdf",
                "md": None,
            }
            pages.append(page)
            preprocess_page(ws, page, cfg)
            if (i + 1) % 20 == 0:
                ws.save()
                log(f"  {i + 1}/{n} PDF pages rendered")
    finally:
        doc.close()   # release the file handle (Windows can't unlink it open)
    ws.stage_reset("transcribe")
    ws.save()
    log(f"{n} pages imported from {pdf.name} — run the pipeline to transcribe")
    return n


def crop_page_photo(ws: Workspace, cfg: dict, page: dict,
                    quad_norm: list, log: Callable[[str], None] = print) -> None:
    """Crop a photo-sourced page (cover, back cover, any patched page) with
    the same corner tool figures use: the quad is perspective-corrected and
    REPLACES the page's source pixels, so everything derived — EPUB cover,
    facsimile, thumbnails — uses exactly what the user framed."""
    import cv2
    import numpy as np

    from .imaging import order_quad
    from .stages.preprocess import correct_page, preprocess_page

    src = ws.root / (page.get("patched_source") or "")
    if not src.exists():
        raise FileNotFoundError(f"{page['id']} has no photo source to crop")
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"{page['id']}: photo unreadable")
    quad = order_quad(np.clip(np.array(quad_norm, dtype=np.float64), 0, 1))
    crop = correct_page(img, quad)
    if crop.shape[0] < 50 or crop.shape[1] < 50:
        raise ValueError("crop too small")
    cv2.imwrite(str(src), crop)
    # old regions/figures/transcription described the uncropped image
    for key in ("regions", "figures"):
        page.pop(key, None)
    if page.get("role") != "cover":
        page["md"] = None
        for key in ("confidence", "flags", "transcribe_error"):
            page.pop(key, None)
        ws.stage_reset("transcribe")
    preprocess_page(ws, page, cfg)
    ws.stage_reset("assemble")
    ws.save()
    log(f"{page['id']}: page photo cropped")


def rotate_patch(ws: Workspace, cfg: dict, page: dict, degrees: int = 180,
                 log: Callable[[str], None] = print) -> None:
    """Rotate a photo-sourced page's pixels in place and re-derive its
    processed images. The old transcription described the wrong orientation,
    so it's cleared and deferred to the next run."""
    import cv2

    src = ws.root / (page.get("patched_source") or "")
    if not src.exists():
        raise FileNotFoundError(f"{page['id']} has no patch photo")
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"{page['id']}: patch unreadable")
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(degrees % 360)
    if rot is None:
        raise ValueError(f"unsupported rotation {degrees}")
    cv2.imwrite(str(src), cv2.rotate(img, rot))
    page["md"] = None
    for key in ("confidence", "flags", "transcribe_error", "regions", "figures"):
        page.pop(key, None)
    from .stages.preprocess import preprocess_page
    preprocess_page(ws, page, cfg)
    ws.stage_reset("transcribe")
    ws.save()
    log(f"{page['id']}: rotated {degrees}° — re-transcribes next run")


def fix_photo_orientation(ws: Workspace, cfg: dict, page: dict,
                          log: Callable[[str], None] = print) -> bool:
    """Vision-check a freshly captured photo page (same check videos get);
    auto-rotate 180° when it was shot upside down. Returns True if rotated."""
    if not page.get("llm_image"):
        return False
    check_cfg = cfg
    if cfg["provider"]["name"] == "hybrid":
        check_cfg = {**cfg, "provider": {**cfg["provider"], "name": "ollama"}}
    from .backends import get_backend
    verdict = get_backend(check_cfg).check_orientation(
        ws.root / page["llm_image"])
    if verdict:
        rotate_patch(ws, cfg, page, 180, log=log)
        return True
    return False


def set_page_deleted(ws: Workspace, page_id: str, deleted: bool = True) -> dict:
    """Soft-delete an erroneous page (mid-turn junk, desk shots, misfires).

    Deletion is remembered by capture identity in manifest.deleted_captures,
    so re-clustering after adding a video doesn't resurrect the page."""
    from .stages.transcribe import cache_key

    page = ws.page(page_id)
    if page is None:
        raise KeyError(page_id)
    key = cache_key(page)
    dl = ws.manifest.setdefault("deleted_captures", [])
    if deleted:
        if key and key not in dl:
            dl.append(key)
        page["status"] = "deleted"
    else:
        if key in dl:
            dl.remove(key)
        page["status"] = "ok"
    ws.stage_reset("assemble")  # the book text must rebuild without/with it
    ws.save()
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
