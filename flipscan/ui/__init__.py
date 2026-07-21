"""flipscan ui — local web GUI. Thin client over the same stage functions the CLI uses."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..config import load_config, save_global_config
from ..project import create_project, run_pipeline
from ..workspace import STAGES, Workspace

# workspace files the browser may fetch, by top-level directory
SERVABLE = {"frames", "work", "figures", "review", "out", "pages", "videos", "patches"}


class VideoSpec(BaseModel):
    path: str
    direction: str = "forward"


class NewProject(BaseModel):
    name: str
    videos: list[VideoSpec]
    title: str | None = None
    expected_pages: int | None = None


class MarkdownEdit(BaseModel):
    markdown: str


class BookEdit(BaseModel):
    title: str | None = None
    author: str | None = None
    expected_pages: int | None = None


class PageEdit(BaseModel):
    printed_number: int | None = None
    needs_reshoot: bool | None = None
    unduplicate: bool | None = None
    mark_duplicate: bool | None = None
    ignore_suspect: bool | None = None
    section: str | None = None   # "" clears; this page then opens a chapter


class ReaderFlag(BaseModel):
    snippet: str
    note: str | None = None


class CropEdit(BaseModel):
    bbox_norm: list[float] | None = None
    quad_norm: list[list[float]] | None = None  # 4 corners; skewed quads get
    #                                             perspective-corrected


class KeepBest(BaseModel):
    items: list[dict]  # [{page_id, fig_idx}, ...] — duplicates of one figure


class Settings(BaseModel):
    provider: str = "ollama"
    ollama_url: str = ""
    ollama_model: str = ""
    anthropic_model: str = ""
    anthropic_api_key: str = ""


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def create_app(root: Path) -> FastAPI:
    root = Path(root).resolve()
    app = FastAPI(title="FlipScan")
    runs: dict[str, dict] = {}  # project name -> {queue, thread}

    def ws_for(name: str) -> Workspace:
        target = (root / name).resolve()
        if not str(target).startswith(str(root)) or not (target / "manifest.json").exists():
            raise HTTPException(404, f"no project {name!r}")
        return Workspace.open(target)

    # ---------------- projects

    @app.get("/api/projects")
    def list_projects():
        out = []
        for p in sorted(root.iterdir()) if root.exists() else []:
            if p.is_dir() and (p / "manifest.json").exists():
                ws = Workspace.open(p)
                pages = ws.manifest["pages"]
                out.append({
                    "name": p.name,
                    "title": ws.manifest["book"].get("title"),
                    "pages": len(pages),
                    "suspects": sum(1 for x in pages if x["status"] == "suspect"),
                    "stages": {s: ws.stage_status(s) for s in STAGES},
                    "running": p.name in runs and runs[p.name]["thread"].is_alive(),
                })
        return out

    @app.post("/api/projects")
    def new_project(spec: NewProject):
        target = root / spec.name
        if (target / "manifest.json").exists():
            raise HTTPException(409, "project already exists")
        for v in spec.videos:
            if not Path(v.path).exists():
                raise HTTPException(400, f"video not found: {v.path}")
        create_project(target, [v.model_dump() for v in spec.videos],
                       title=spec.title, expected_pages=spec.expected_pages)
        return {"ok": True, "name": spec.name}

    def printed_toc(ws: Workspace) -> list[dict]:
        """Chapter list read from the book's own contents page (front matter)."""
        from ..stages.assemble import parse_printed_toc
        texts = []
        for p in ws.manifest["pages"]:
            n = p.get("printed_number")
            if ((n is None or n < 5) and p.get("md") and not p.get("role")
                    and len(texts) < 40):
                f = ws.root / p["md"]
                if f.exists():
                    texts.append(f.read_text(encoding="utf-8"))
        return [{"title": t, "start": s} for t, s in parse_printed_toc(texts)]

    @app.get("/api/projects/{name}")
    def project_detail(name: str):
        from ..review import page_reasons
        ws = ws_for(name)
        m = ws.manifest
        return {
            "name": name,
            "book": m["book"],
            "toc": printed_toc(ws),
            "videos": m["videos"],
            "stages": {s: ws.stage_status(s) for s in STAGES},
            "pages": [{**p, "reasons": page_reasons(p)} for p in m["pages"]],
            "running": name in runs and runs[name]["thread"].is_alive(),
            "outputs": [f.name for f in ws.dir("out").glob("*") if f.is_file()],
            "contact_sheet": (ws.work_file("contact_sheet.jpg")).exists(),
        }

    @app.patch("/api/projects/{name}")
    def edit_project(name: str, edit: BookEdit):
        """Update book metadata: title, author, expected page count."""
        ws = ws_for(name)
        fields = edit.model_dump(exclude_unset=True)
        for k in ("title", "author", "expected_pages"):
            if k in fields:
                v = fields[k]
                ws.manifest["book"][k] = (v.strip() or None) if isinstance(v, str) else v
        ws.save()
        return {"ok": True, "book": ws.manifest["book"]}

    # ---------------- pipeline

    @app.post("/api/projects/{name}/run")
    def run_project(name: str, provider: str | None = None, force: bool = False):
        ws = ws_for(name)
        if name in runs and runs[name]["thread"].is_alive():
            raise HTTPException(409, "already running")
        cfg = load_config(ws.root)
        if provider:
            cfg["provider"]["name"] = provider

        # fan-out log: every SSE stream gets its own queue, and ALL of them
        # receive the final None terminator. A single shared queue meant only
        # one stream ever saw the end-of-run marker — the others kept a
        # threadpool thread parked forever (which eventually starved every
        # sync endpoint in the app: the "server froze at end of run" bug).
        run = {"subs": [], "history": deque(maxlen=1000),
               "lock": threading.Lock(), "done": False}

        def put(msg):
            with run["lock"]:
                run["history"].append(msg)
                if msg is None:
                    run["done"] = True
                for sub in run["subs"]:
                    sub.put(msg)

        def work():
            try:
                run_pipeline(ws, cfg, force=force, log=lambda m: put(str(m)))
                put("[pipeline] finished")
            except Exception as e:  # surface errors into the log stream
                put(f"[pipeline] ERROR: {e}")
            finally:
                put(None)

        t = threading.Thread(target=work, daemon=True)
        run["thread"] = t
        runs[name] = run
        t.start()
        return {"ok": True}

    rate_samples: dict[str, list] = {}  # project -> [(t, stage, done)] for ETA

    @app.get("/api/projects/{name}/progress")
    def progress(name: str):
        import time as _time

        ws = ws_for(name)
        stages_status = {s: ws.stage_status(s) for s in STAGES}
        current = next((s for s in STAGES if stages_status[s] != "done"), None)
        pages = ws.manifest["pages"]
        detail = None
        if current == "transcribe":
            eligible = [p for p in pages if p.get("role") != "cover"]
            detail = {"done": sum(1 for p in eligible if p.get("md")),
                      "total": len(eligible), "unit": "pages transcribed"}
        elif current == "preprocess":
            detail = {"done": sum(1 for p in pages if p.get("color")),
                      "total": len(pages), "unit": "pages corrected"}
        elif current in ("extract", "score"):
            vids = ws.manifest["videos"]
            done = sum(1 for v in vids
                       if (current == "extract" and v.get("frames_extracted"))
                       or (current == "score"
                           and ws.work_file(f"scores_{v['id']}.json").exists()))
            detail = {"done": done, "total": len(vids), "unit": "videos"}

        eta = None
        if detail and detail["total"]:
            samples = rate_samples.setdefault(name, [])
            now = _time.time()
            samples.append((now, current, detail["done"]))
            samples[:] = [s for s in samples if s[1] == current and now - s[0] < 900]
            if len(samples) >= 2 and samples[-1][2] > samples[0][2]:
                rate = (samples[-1][2] - samples[0][2]) / (samples[-1][0] - samples[0][0])
                if rate > 0:
                    eta = int((detail["total"] - detail["done"]) / rate)
        return {
            "running": name in runs and runs[name]["thread"].is_alive(),
            "stages": stages_status,
            "current": current,
            "detail": detail,
            "eta_seconds": eta,
        }

    @app.get("/api/projects/{name}/events")
    async def events(name: str):
        if name not in runs:
            raise HTTPException(404, "no active run")
        run = runs[name]
        sub: queue.Queue = queue.Queue()
        with run["lock"]:
            for msg in run["history"]:  # replay what this stream missed
                sub.put(msg)
            if not run["done"]:
                run["subs"].append(sub)

        async def stream():
            # non-blocking poll — an SSE reader must never park a threadpool
            # thread; it may stay open (quietly) for hours
            idle = 0.0
            try:
                while True:
                    try:
                        msg = sub.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.4)
                        idle += 0.4
                        if idle >= 20:
                            idle = 0.0
                            yield ": keepalive\n\n"
                        continue
                    idle = 0.0
                    if msg is None:
                        yield "event: done\ndata: done\n\n"
                        break
                    yield f"data: {json.dumps(msg)}\n\n"
            finally:
                with run["lock"]:
                    if sub in run["subs"]:
                        run["subs"].remove(sub)

        return StreamingResponse(stream(), media_type="text/event-stream")

    # ---------------- settings (global: applies to every project)

    @app.get("/api/settings")
    def get_settings():
        p = load_config()["provider"]
        return {
            "provider": p["name"], "ollama_url": p["ollama_url"],
            "ollama_model": p["ollama_model"],
            "anthropic_model": p["anthropic_model"],
            "anthropic_api_key_set": bool(p.get("anthropic_api_key")
                                          or os.environ.get("ANTHROPIC_API_KEY")
                                          or os.environ.get("FLIPSCAN_ANTHROPIC_API_KEY")),
        }

    @app.put("/api/settings")
    def put_settings(s: Settings):
        current = load_config()["provider"]
        save_global_config({"provider": {
            "name": s.provider,
            "ollama_url": s.ollama_url,
            "ollama_model": s.ollama_model,
            "anthropic_model": s.anthropic_model,
            # keep the stored key unless a new one was typed
            "anthropic_api_key": s.anthropic_api_key or current.get("anthropic_api_key", ""),
        }})
        return {"ok": True}

    @app.get("/api/settings/ollama-models")
    def ollama_models(url: str):
        import httpx
        try:
            r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=6.0)
            r.raise_for_status()
            return {"ok": True,
                    "models": [m["name"] for m in r.json().get("models", [])]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------------- uploads (videos/photos from the browser, incl. phones)

    @app.post("/api/upload")
    async def upload(file: UploadFile):
        uploads = root / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        safe = Path(file.filename or "upload.bin").name
        dest = uploads / safe
        i = 1
        while dest.exists():
            dest = uploads / f"{Path(safe).stem}-{i}{Path(safe).suffix}"
            i += 1
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 22):  # 4 MB chunks — videos are big
                f.write(chunk)
        return {"ok": True, "path": str(dest)}

    @app.post("/api/projects/{name}/videos")
    async def add_video_endpoint(name: str, video: UploadFile):
        ws = ws_for(name)
        if name in runs and runs[name]["thread"].is_alive():
            raise HTTPException(409, "pipeline is running — wait for it to finish")
        uploads = root / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / Path(video.filename or "video.mov").name
        with open(dest, "wb") as f:
            while chunk := await video.read(1 << 22):
                f.write(chunk)
        from ..project import add_video
        entry = await asyncio.to_thread(add_video, ws, dest, log=lambda m: None)
        return {"ok": True, "id": entry["id"]}

    @app.put("/api/projects/{name}/videos/{vid}")
    def set_video_rotation(name: str, vid: str, rotate: int):
        """Mark a video as shot upside down (rotate=180) or normal (rotate=0).
        Re-derives everything from preprocess onward for that video's pages."""
        ws = ws_for(name)
        if not any(v["id"] == vid for v in ws.manifest["videos"]):
            raise HTTPException(404, f"no video {vid!r}")
        from ..project import set_video_rotation
        set_video_rotation(ws, vid, rotate, log=lambda m: None)
        ws.stage_reset("preprocess")
        ws.save()
        return {"ok": True, "rotate": rotate}

    @app.post("/api/projects/{name}/pages/add")
    async def add_page(name: str, photo: UploadFile, position: str = "end",
                       cover: bool = False):
        ws = ws_for(name)
        cfg = load_config(ws.root)
        uploads = ws.root / "uploads"
        uploads.mkdir(exist_ok=True)
        tmp = uploads / Path(photo.filename or "page.jpg").name
        tmp.write_bytes(await photo.read())
        from ..project import add_page_from_photo
        page = await asyncio.to_thread(
            add_page_from_photo, ws, cfg, tmp, position=position,
            role="cover" if cover else None, transcribe=False,
            log=lambda m: None)
        tmp.unlink(missing_ok=True)
        return {"ok": True, "id": page["id"],
                "transcription": "deferred — run the pipeline"}

    # ---------------- pages / patch / review

    @app.put("/api/projects/{name}/pages/{page_id}/md")
    def save_md(name: str, page_id: str, edit: MarkdownEdit):
        ws = ws_for(name)
        page = ws.page(page_id)
        if page is None:
            raise HTTPException(404, "no such page")
        md_path = ws.dir("pages") / f"{page_id}.md"
        md_path.write_text(edit.markdown, encoding="utf-8")
        page["md"] = f"pages/{page_id}.md"
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True}

    @app.post("/api/projects/{name}/pages/{page_id}/retranscribe")
    async def retranscribe_page(name: str, page_id: str):
        """Retry OCR for one page right now (e.g. after a model hiccup) —
        heavy work off the event loop; other requests keep flowing."""
        ws = ws_for(name)
        page = ws.page(page_id)
        if page is None:
            raise HTTPException(404, "no such page")
        if not page.get("llm_image"):
            raise HTTPException(400, "page has no processed image yet — run the pipeline first")
        cfg = load_config(ws.root)
        if cfg["provider"]["name"] == "hybrid":  # one page: local is plenty
            cfg = {**cfg, "provider": {**cfg["provider"], "name": "ollama"}}
        from ..backends import get_backend
        from ..stages.transcribe import _cache_page, _write_result
        backend = get_backend(cfg)

        def work():
            src = ws.root / page["llm_image"]
            r = backend.transcribe([(page_id, src)], log=lambda m: None)[page_id]
            if "error" in r:
                # repetition loops are usually triggered by the neighboring
                # page's curled text at the frame edge — retry with each
                # vertical edge shaved off (p0093 taught us this)
                import cv2
                img = cv2.imread(str(src))
                if img is not None:
                    w = img.shape[1]
                    for sl in (slice(0, int(w * 0.86)), slice(int(w * 0.14), w)):
                        tmp = ws.work_file(f"_retry_{page_id}.jpg")
                        cv2.imwrite(str(tmp), img[:, sl],
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                        r2 = backend.transcribe([(page_id, tmp)],
                                                log=lambda m: None)[page_id]
                        tmp.unlink(missing_ok=True)
                        if "error" not in r2:
                            r = r2
                            break
            _write_result(ws, page, r, backend.name)
            _cache_page(ws, page)
            return r

        r = await asyncio.to_thread(work)
        if "error" in r:
            ws.save()
            raise HTTPException(502, f"failed again: {r['error']}")
        _reconcile(ws)  # the retry may have brought a page number with it
        ws.stage_reset("figures")
        ws.save()
        return {"ok": True}

    @app.post("/api/projects/{name}/pages/{page_id}/patch")
    async def patch_page(name: str, page_id: str, photo: UploadFile):
        """Replace a page's capture. Transcription is DEFERRED to the next
        pipeline run — running a model synchronously inside a request froze
        the whole server (async endpoints run on the event loop)."""
        ws = ws_for(name)
        cfg = load_config(ws.root)
        page = ws.page(page_id)
        if page is None:
            raise HTTPException(404, "no such page")
        patches = ws.root / "patches"
        patches.mkdir(exist_ok=True)
        suffix = Path(photo.filename or "photo.jpg").suffix or ".jpg"
        dest = patches / f"{page_id}{suffix}"
        dest.write_bytes(await photo.read())

        page["patched_source"] = f"patches/{dest.name}"
        page["status"] = "patched"
        page["md"] = None
        for key in ("confidence", "flags", "transcribe_error"):
            page.pop(key, None)

        from ..stages.preprocess import preprocess_page
        await asyncio.to_thread(preprocess_page, ws, page, cfg)
        ws.stage_reset("transcribe")  # the deferred page must transcribe next run
        ws.save()
        return {"ok": True, "transcription": "deferred — run the pipeline"}

    def _reconcile(ws):
        from ..stages.transcribe import reconcile
        reconcile(ws, ws.manifest["pages"], log=lambda m: None)
        ws.stage_reset("assemble")
        ws.save()

    @app.patch("/api/projects/{name}/pages/{page_id}")
    def edit_page(name: str, page_id: str, edit: PageEdit):
        """Manual corrections: set/clear the printed page number, mark a page
        for re-acquisition, or rescue a false duplicate."""
        ws = ws_for(name)
        page = ws.page(page_id)
        if page is None:
            raise HTTPException(404, "no such page")
        fields = edit.model_fields_set
        if "printed_number" in fields:
            # the cache keeps the pristine model-read value; manual numbers
            # live in the manifest only (number_manual survives reconcile)
            page["printed_number"] = edit.printed_number
            page["number_manual"] = True
            page.pop("number_inferred", None)
            page.pop("number_conflict", None)
        if edit.needs_reshoot is not None:
            page["needs_reshoot"] = edit.needs_reshoot
        if edit.unduplicate:
            page["status"] = "ok"
            page["dedupe_exempt"] = True
            page.pop("manual_duplicate", None)
        if edit.mark_duplicate:
            page["status"] = "duplicate"
            page["manual_duplicate"] = True  # auto-dedupe keeps hands off
        if edit.ignore_suspect:
            from ..review import page_reasons
            # remember what the complaint was — the page shows "acceptable"
            # with the original issue noted, not a clean bill of health
            page["ignored_reasons"] = page_reasons(page)
            page["suspect_ignored"] = True   # the user vouches for this page
        if "section" in fields:
            s = (edit.section or "").strip()
            if s:
                page["section"] = s
            else:
                page.pop("section", None)
            ws.stage_reset("assemble")   # the book structure changed
            if page["status"] == "suspect":
                page["status"] = "ok"
        _reconcile(ws)
        return {"ok": True, "page": ws.page(page_id)}

    def _figure(ws, page_id: str, fig_idx: int):
        """Resolve (page, figure path, its region). fig_idx indexes
        page['figures']; the region is recovered from the filename letter."""
        page = ws.page(page_id)
        figs = (page or {}).get("figures") or []
        if page is None or fig_idx >= len(figs):
            raise HTTPException(404, "no such figure")
        rel = figs[fig_idx]
        letter = Path(rel).stem.rsplit("_", 1)[-1]
        ridx = ord(letter[0]) - ord("a") if letter and letter[0].isalpha() else fig_idx
        regions = page.get("regions") or []
        region = regions[ridx] if 0 <= ridx < len(regions) else None
        return page, rel, region

    def _compute_crop(ws, page, edit: CropEdit):
        """Shared crop math: bbox crops directly; a 4-corner quad is
        perspective-warped back to a rectangle. Returns (crop, bbox, quad)."""
        import cv2
        import numpy as np

        from ..imaging import order_quad
        from ..stages.preprocess import correct_page

        if not page.get("color"):
            raise HTTPException(400, "page has no corrected image")
        color = cv2.imread(str(ws.root / page["color"]))
        if color is None:
            raise HTTPException(500, "page image unreadable")
        h, w = color.shape[:2]

        if edit.quad_norm and len(edit.quad_norm) == 4:
            quad = order_quad(np.clip(np.array(edit.quad_norm, dtype=np.float64), 0, 1))
            crop = correct_page(color, quad)
            xs, ys = quad[:, 0], quad[:, 1]
            bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            stored_quad = quad.tolist()
        elif edit.bbox_norm and len(edit.bbox_norm) == 4:
            x0, y0, x1, y1 = edit.bbox_norm
            x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
            y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
            px = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
            crop = color[px[1]:px[3], px[0]:px[2]]
            bbox = [x0, y0, x1, y1]
            stored_quad = None
        else:
            raise HTTPException(400, "need bbox_norm or quad_norm")
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            raise HTTPException(400, "crop too small")
        return crop, bbox, stored_quad

    @app.post("/api/projects/{name}/pages/{page_id}/figures/{fig_idx}/crop")
    def recrop_figure(name: str, page_id: str, fig_idx: int, edit: CropEdit):
        """Re-crop an existing figure."""
        import cv2

        ws = ws_for(name)
        page, rel, region = _figure(ws, page_id, fig_idx)
        crop, bbox, stored_quad = _compute_crop(ws, page, edit)
        cv2.imwrite(str(ws.root / rel), crop)
        if region is not None:
            from ..stages.figures import file_ref
            region["bbox_norm"] = bbox
            if stored_quad:
                region["quad_norm"] = stored_quad
            region["user_crop"] = True
            region["color_ref"] = file_ref(ws.root / page["color"])
            region.pop("stale_crop", None)
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True, "figure": rel}

    @app.post("/api/projects/{name}/pages/{page_id}/regions/{ridx}/crop")
    def crop_region(name: str, page_id: str, ridx: int, edit: CropEdit):
        """Create the figure for a region that has none (its [[region-N]]
        marker sits in the markdown with nothing to show) — or re-crop it."""
        import re as _re

        import cv2

        ws = ws_for(name)
        page = ws.page(page_id)
        if page is None:
            raise HTTPException(404, "no such page")
        crop, bbox, stored_quad = _compute_crop(ws, page, edit)

        rel = f"figures/{page_id}_{chr(97 + ridx % 26)}.png"
        cv2.imwrite(str(ws.root / rel), crop)

        regions = page.setdefault("regions", [])
        while len(regions) <= ridx:
            regions.append({"type": "figure", "bbox_norm": [0, 0, 1, 1], "caption": ""})
        region = regions[ridx]
        from ..stages.figures import file_ref
        region["bbox_norm"] = bbox
        if stored_quad:
            region["quad_norm"] = stored_quad
        region["user_crop"] = True
        region["color_ref"] = file_ref(ws.root / page["color"])
        region.pop("stale_crop", None)
        region.pop("deleted", None)

        figs = page.setdefault("figures", [])
        if rel not in figs:
            figs.append(rel)

        if page.get("md") and (ws.root / page["md"]).exists():
            md_path = ws.root / page["md"]
            md = md_path.read_text(encoding="utf-8")
            img_md = f"![{region.get('caption') or ''}]({rel})"
            placeholder = f"[[region-{ridx}]]"
            if placeholder in md:
                md = md.replace(placeholder, img_md)
            elif rel not in md:
                md = md.rstrip() + f"\n\n{img_md}\n"
            md_path.write_text(md, encoding="utf-8")

        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True, "figure": rel}

    def _delete_figure(ws, page, rel):
        import re as _re

        letter = Path(rel).stem.rsplit("_", 1)[-1]
        ridx = ord(letter[0]) - ord("a") if letter and letter[0].isalpha() else -1
        regions = page.get("regions") or []
        if 0 <= ridx < len(regions):
            regions[ridx]["deleted"] = True  # tombstone keeps indexes stable
        page["figures"] = [f for f in (page.get("figures") or []) if f != rel]
        (ws.root / rel).unlink(missing_ok=True)
        if page.get("md") and (ws.root / page["md"]).exists():
            md_path = ws.root / page["md"]
            md = _re.sub(r"!\[[^\]]*\]\(" + _re.escape(rel) + r"\)\n?", "",
                         md_path.read_text(encoding="utf-8"))
            md_path.write_text(md, encoding="utf-8")

    @app.delete("/api/projects/{name}/pages/{page_id}/figures/{fig_idx}")
    def delete_figure(name: str, page_id: str, fig_idx: int):
        """Remove a figure: file, markdown link, and region tombstone."""
        ws = ws_for(name)
        page, rel, _region = _figure(ws, page_id, fig_idx)
        _delete_figure(ws, page, rel)
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True}

    @app.post("/api/projects/{name}/pages/{page_id}/magic-crop")
    def magic_crop(name: str, page_id: str, edit: CropEdit):
        """Edge-detection crop suggestion for the crop modal: returns a bbox
        for the user to review and adjust before saving."""
        import cv2
        import numpy as np

        from ..imaging import refine_figure_bbox

        ws = ws_for(name)
        page = ws.page(page_id)
        if page is None or not page.get("color"):
            raise HTTPException(404, "no page image")
        color = cv2.imread(str(ws.root / page["color"]))
        if color is None:
            raise HTTPException(500, "page image unreadable")
        if edit.quad_norm:
            q = np.array(edit.quad_norm, dtype=np.float64)
            prior = [float(q[:, 0].min()), float(q[:, 1].min()),
                     float(q[:, 0].max()), float(q[:, 1].max())]
        elif edit.bbox_norm:
            prior = edit.bbox_norm
        else:
            prior = [0.05, 0.05, 0.95, 0.95]
        from ..imaging import detect_figures
        # candidate list: the prior-anchored refiner first (when it fires),
        # then prior-free full-page detections — the modal cycles through them
        cands = []
        box = refine_figure_bbox(color, prior)
        if box is not None:
            cands.append(box)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        for c in detect_figures(gray):
            if all(_iou(c, b) < 0.6 for b in cands):
                cands.append(c)
        if not cands:
            return {"ok": False, "detail": "no confident detection here — "
                                           "draw the box manually"}
        # most-relevant first: overlap with the user's current box wins
        cands.sort(key=lambda c: -_iou(c, prior))
        return {"ok": True, "bbox_norm": cands[0], "candidates": cands}

    @app.post("/api/projects/{name}/pages/{page_id}/figures/{fig_idx}/reshoot")
    def figure_reshoot(name: str, page_id: str, fig_idx: int, flag: bool = True):
        """Mark a figure for re-acquisition: the page text is fine, but the
        figure deserves a dedicated higher-res photo."""
        ws = ws_for(name)
        _page, _rel, region = _figure(ws, page_id, fig_idx)
        if region is None:
            raise HTTPException(404, "figure has no region")
        if flag:
            region["needs_reshoot"] = True
        else:
            region.pop("needs_reshoot", None)
        ws.save()
        return {"ok": True}

    @app.post("/api/projects/{name}/figures/auto-refine")
    def auto_refine_figures(name: str):
        """Edge-detection pass over every figure the user hasn't hand-cropped:
        snap the model's approximate bbox to the actual photo and re-crop."""
        import cv2

        from ..imaging import refine_figure_bbox

        ws = ws_for(name)
        refined = 0
        for page in ws.manifest["pages"]:
            if page.get("status") in ("duplicate", "deleted") or not page.get("color"):
                continue
            regions = page.get("regions") or []
            color = None
            for ri, region in enumerate(regions):
                if region.get("user_crop") or region.get("deleted"):
                    continue
                if color is None:
                    color = cv2.imread(str(ws.root / page["color"]))
                    if color is None:
                        break
                box = refine_figure_bbox(color, region["bbox_norm"])
                if box is None:
                    continue
                h, w = color.shape[:2]
                px = (int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))
                if px[2] - px[0] < 20 or px[3] - px[1] < 20:
                    continue
                rel = f"figures/{page['id']}_{chr(97 + ri % 26)}.png"
                cv2.imwrite(str(ws.root / rel), color[px[1]:px[3], px[0]:px[2]])
                region["bbox_norm"] = box
                region["auto_refined"] = True
                if rel not in (page.get("figures") or []):
                    page.setdefault("figures", []).append(rel)
                refined += 1
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True, "refined": refined}

    @app.post("/api/projects/{name}/figures/keep-best")
    def keep_best_figure(name: str, spec: KeepBest):
        """Given duplicate captures of the same figure, keep the sharpest and
        delete the rest."""
        import cv2

        ws = ws_for(name)
        scored = []
        for it in spec.items:
            page, rel, _r = _figure(ws, it["page_id"], int(it["fig_idx"]))
            img = cv2.imread(str(ws.root / rel), cv2.IMREAD_GRAYSCALE)
            sharp = float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None else -1
            scored.append((sharp, page, rel))
        if len(scored) < 2:
            raise HTTPException(400, "need at least two figures to compare")
        scored.sort(key=lambda t: t[0], reverse=True)
        for _sharp, page, rel in scored[1:]:
            _delete_figure(ws, page, rel)
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True, "kept": scored[0][2],
                "deleted": [rel for _s, _p, rel in scored[1:]]}

    @app.post("/api/projects/{name}/pages/{page_id}/figures/{fig_idx}/upload")
    async def upload_figure(name: str, page_id: str, fig_idx: int, photo: UploadFile):
        """Replace a figure image with an uploaded photo (e.g. re-shot close-up)."""
        import cv2
        import numpy as np

        ws = ws_for(name)
        page, rel, region = _figure(ws, page_id, fig_idx)
        data = await photo.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "not a readable image")
        cv2.imwrite(str(ws.root / rel), img)  # keep filename: markdown link stays valid
        if region is not None:
            region["user_crop"] = True  # never let the figures stage overwrite it
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True}

    @app.delete("/api/projects/{name}/pages/{page_id}")
    def delete_page(name: str, page_id: str):
        ws = ws_for(name)
        from ..project import set_page_deleted
        try:
            set_page_deleted(ws, page_id, True)
        except KeyError:
            raise HTTPException(404, "no such page")
        return {"ok": True}

    @app.post("/api/projects/{name}/pages/{page_id}/restore")
    def restore_page(name: str, page_id: str):
        ws = ws_for(name)
        from ..project import set_page_deleted
        try:
            set_page_deleted(ws, page_id, False)
        except KeyError:
            raise HTTPException(404, "no such page")
        return {"ok": True}

    @app.get("/api/projects/{name}/capture-queue")
    def capture_queue(name: str):
        """Everything worth photographing, in page order: pages never captured
        (from printed-number gaps) and badly-captured pages (reshoot list)."""
        from ..review import reshoot_list
        ws = ws_for(name)
        items = []
        for n in ws.manifest.get("missing_pages", []):
            items.append({"kind": "missing", "number": n,
                          "label": f"page {n}", "reasons": ["never captured"]})
        for it in reshoot_list(ws):
            # only pages the user can actually find in the physical book —
            # an internal p#### id means nothing at the bookshelf
            if it.get("printed_number") is None or it["printed_number"] < 1:
                continue
            items.append({"kind": "reshoot", "page_id": it["id"],
                          "number": it["printed_number"],
                          "label": f"page {it['printed_number']}",
                          "reasons": it["reasons"]})
        # figures flagged for a dedicated close-up shot
        for p in ws.manifest["pages"]:
            if p.get("status") in ("duplicate", "deleted") or p.get("role"):
                continue
            for ri, r in enumerate(p.get("regions") or []):
                if r.get("needs_reshoot") and p.get("printed_number"):
                    figs = p.get("figures") or []
                    expected = f"figures/{p['id']}_{chr(97 + ri % 26)}.png"
                    items.append({
                        "kind": "figure", "page_id": p["id"], "fig_idx": ri,
                        "number": p["printed_number"],
                        "label": f"figure on page {p['printed_number']}",
                        "reasons": [f"close-up of the "
                                    f"{r.get('caption') or 'figure'}"],
                        "preview": expected if expected in figs else None,
                    })
        items.sort(key=lambda i: i["number"])
        return {"items": items}

    @app.post("/api/projects/{name}/capture")
    async def capture(name: str, photo: UploadFile, kind: str,
                      number: int | None = None, page_id: str | None = None,
                      fig_idx: int | None = None):
        """One wizard shot. Transcription is deferred so the user can bang
        through the queue; the next pipeline run transcribes everything new."""
        ws = ws_for(name)
        cfg = load_config(ws.root)
        uploads = ws.root / "uploads"
        uploads.mkdir(exist_ok=True)
        tmp = uploads / f"capture_{Path(photo.filename or 'shot.jpg').name}"
        tmp.write_bytes(await photo.read())

        from ..project import add_page_from_photo
        if kind == "missing":
            page = await asyncio.to_thread(
                add_page_from_photo, ws, cfg, tmp, position="end",
                transcribe=False, log=lambda m: None)
            page.pop("pinned", None)  # order by page number, not by "end"
            if number is not None:
                page["printed_number"] = number
                page["number_manual"] = True
            _reconcile(ws)
            ws.stage_reset("transcribe")
            ws.save()
            result = page["id"]
        elif kind == "figure" and page_id is not None and fig_idx is not None:
            # fig_idx is the REGION index here (queue items address regions)
            import cv2
            import numpy as np
            page = ws.page(page_id)
            if page is None:
                raise HTTPException(404, "no such page")
            img = cv2.imdecode(np.frombuffer(tmp.read_bytes(), np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(400, "not a readable image")
            rel = f"figures/{page_id}_{chr(97 + fig_idx % 26)}.png"
            cv2.imwrite(str(ws.root / rel), img)
            if rel not in (page.get("figures") or []):
                page.setdefault("figures", []).append(rel)
            regions = page.get("regions") or []
            if fig_idx < len(regions):
                regions[fig_idx]["user_crop"] = True
                regions[fig_idx].pop("needs_reshoot", None)
            ws.stage_reset("assemble")
            ws.save()
            result = rel
        elif kind == "reshoot" and page_id:
            page = ws.page(page_id)
            if page is None:
                raise HTTPException(404, "no such page")
            patches = ws.root / "patches"
            patches.mkdir(exist_ok=True)
            dest = patches / f"{page_id}{tmp.suffix or '.jpg'}"
            dest.write_bytes(tmp.read_bytes())
            page["patched_source"] = f"patches/{dest.name}"
            page["status"] = "patched"
            page["md"] = None
            page.pop("needs_reshoot", None)
            # the wizard told the user "photograph page N" and they did —
            # that's a human confirmation of the number, even if the printed
            # folio isn't visible on the page (full-page plates often have
            # none). Without this the next reconcile drops the number.
            if number is not None:
                page["printed_number"] = number
                page["number_manual"] = True
            for key in ("confidence", "flags", "transcribe_error"):
                page.pop(key, None)
            from ..stages.preprocess import preprocess_page
            await asyncio.to_thread(preprocess_page, ws, page, cfg)
            ws.stage_reset("transcribe")
            ws.save()
            result = page_id
        else:
            raise HTTPException(400, "bad capture kind")
        tmp.unlink(missing_ok=True)
        return {"ok": True, "page": result}

    @app.post("/api/projects/{name}/reader-flag")
    def reader_flag(name: str, flag: ReaderFlag):
        """Flag an issue from inside the EPUB reader: match the passage the
        reader is showing back to its source page and mark that page for
        re-acquisition."""
        from ..review import find_page_by_text
        ws = ws_for(name)
        p = find_page_by_text(ws, flag.snippet)
        if p is None:
            raise HTTPException(404, "couldn't match this passage to a page — "
                                     "flag it from the Pages tab instead")
        p["needs_reshoot"] = True
        if flag.note:
            p["flag_note"] = flag.note.strip()
        ws.save()
        return {"ok": True, "page": p["id"],
                "printed_number": p.get("printed_number")}

    # ---------------- proofread: a distinct, non-destructive layer

    @app.get("/api/projects/{name}/proof")
    def proof_list(name: str):
        from ..proofread import proof_status
        ws = ws_for(name)
        try:
            return {"chapters": proof_status(ws)}
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/projects/{name}/proof/{idx}")
    def proof_detail(name: str, idx: int):
        from ..proofread import load_proof
        ws = ws_for(name)
        d = load_proof(ws, idx)
        if d is None:
            raise HTTPException(404, "chapter not proofread yet")
        return d

    @app.post("/api/projects/{name}/proof/{idx}/run")
    async def proof_run(name: str, idx: int):
        """Proofread one chapter (lint + LLM edit list). Heavy work off the
        event loop; the frontend runs chapters sequentially for 'proof all'."""
        from ..proofread import proofread_chapter
        ws = ws_for(name)
        cfg = load_config(ws.root)
        try:
            d = await asyncio.to_thread(proofread_chapter, ws, cfg, idx)
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(502, f"proofread failed: {e}")
        return d

    @app.post("/api/projects/{name}/proof/{idx}/finding/{fi}")
    def proof_finding(name: str, idx: int, fi: int, enabled: bool):
        """Per-finding accept/reject: rebuilds the proofed chapter copy."""
        from ..proofread import toggle_finding
        ws = ws_for(name)
        try:
            return toggle_finding(ws, idx, fi, enabled)
        except (FileNotFoundError, IndexError) as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/projects/{name}/proof/{idx}/review")
    def proof_review(name: str, idx: int, accept: bool):
        from ..proofread import load_proof, save_proof
        ws = ws_for(name)
        d = load_proof(ws, idx)
        if d is None:
            raise HTTPException(404, "chapter not proofread yet")
        d["status"] = "accepted" if accept else "rejected"
        save_proof(ws, idx, d)
        return {"ok": True, "status": d["status"]}

    @app.get("/api/projects/{name}/reshoot")
    def reshoot(name: str):
        from ..review import reshoot_list
        from ..stages.transcribe import format_ranges
        ws = ws_for(name)
        return {
            "items": reshoot_list(ws),
            "missing": ws.manifest.get("missing_pages", []),
            "missing_ranges": format_ranges(ws.manifest.get("missing_pages", [])),
        }

    # ---------------- build + files

    @app.post("/api/projects/{name}/build")
    def build(name: str, format: str = "epub", device: str = "none"):
        ws = ws_for(name)
        ext = "epub" if format == "epub" else "pdf"
        suffix = "-facsimile" if format == "pdf-facsimile" else ""
        if device != "none":
            suffix += f"-{device}"
        out = ws.dir("out") / f"{ws.root.name}{suffix}.{ext}"
        try:
            if format == "epub":
                from ..build_epub import build_epub
                build_epub(ws, out, author=ws.manifest["book"].get("author"),
                           device=device, log=lambda m: None)
            elif format == "pdf-facsimile":
                from ..build_pdf import build_pdf_facsimile
                build_pdf_facsimile(ws, out, device=device, log=lambda m: None)
            elif format == "pdf":
                from ..build_pdf import build_pdf_reflowed
                build_pdf_reflowed(ws, out, log=lambda m: None)
            else:
                raise HTTPException(400, f"unknown format {format!r}")
        except (RuntimeError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "file": out.name}

    @app.get("/api/projects/{name}/thumb/{path:path}")
    def get_thumb(name: str, path: str, w: int = 760, request: Request = None):
        """Downscaled JPEG of a workspace image. The page list was serving
        full-resolution multi-MB PNGs — hundreds of those OOM-crash phone
        browsers. Thumbs are cached by source mtime and revalidated cheaply."""
        import cv2

        ws = ws_for(name)
        src = (ws.root / path).resolve()
        if not str(src).startswith(str(ws.root.resolve())) or not src.is_file():
            raise HTTPException(404, "not found")
        if path.split("/")[0] not in SERVABLE:
            raise HTTPException(403, "not servable")
        w = max(64, min(2000, w))
        st = src.stat()
        etag = f'"{st.st_mtime_ns}-{st.st_size}-{w}"'
        if request is not None and request.headers.get("if-none-match") == etag:
            return Response(status_code=304)
        tdir = ws.work_file("thumbs")
        tdir.mkdir(exist_ok=True)
        import hashlib
        key = hashlib.sha1(f"{path}|{w}|{st.st_mtime_ns}".encode()).hexdigest()[:20]
        out = tdir / f"{key}.jpg"
        if not out.exists():
            img = cv2.imread(str(src))
            if img is None:
                raise HTTPException(500, "image unreadable")
            h0, w0 = img.shape[:2]
            if w0 > w:
                img = cv2.resize(img, (w, int(h0 * w / w0)),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return FileResponse(out, media_type="image/jpeg",
                            headers={"Cache-Control": "no-cache", "ETag": etag})

    @app.get("/api/projects/{name}/file/{path:path}")
    def get_file(name: str, path: str):
        ws = ws_for(name)
        target = (ws.root / path).resolve()
        if not str(target).startswith(str(ws.root.resolve())) or not target.is_file():
            raise HTTPException(404, "not found")
        top = path.split("/")[0]
        if top not in SERVABLE:
            raise HTTPException(403, "not servable")
        # extracted frames are immutable; everything else (corrected pages,
        # figures, patches, outputs) gets rewritten in place — always revalidate
        headers = None if top == "frames" else {"Cache-Control": "no-cache"}
        return FileResponse(target, headers=headers)

    # ---------------- frontend

    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        # always revalidate the app shell — stale cached UI on phones is worse
        # than the tiny refetch
        return FileResponse(static_dir / "index.html",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/static/{path:path}")
    def static_file(path: str):
        target = (static_dir / path).resolve()
        if not str(target).startswith(str(static_dir.resolve())) or not target.is_file():
            raise HTTPException(404, "not found")
        cache = ("max-age=86400" if path.startswith("vendor/")  # libs are pinned
                 else "no-cache")
        return FileResponse(target, headers={"Cache-Control": cache})

    return app


def serve(root: Path, host: str = "127.0.0.1", port: int = 8321) -> None:
    import uvicorn
    uvicorn.run(create_app(root), host=host, port=port, log_level="warning")
