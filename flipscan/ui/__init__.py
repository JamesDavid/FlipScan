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
    name: str | None = None          # optional — a unique slug is generated
    videos: list[VideoSpec] = []
    title: str | None = None
    author: str | None = None
    expected_pages: int | None = None
    isbn: str | None = None
    publisher: str | None = None
    year: str | None = None


class MarkdownEdit(BaseModel):
    markdown: str


class BookEdit(BaseModel):
    title: str | None = None
    author: str | None = None
    expected_pages: int | None = None
    isbn: str | None = None
    publisher: str | None = None
    year: str | None = None


def _slugify(s: str) -> str:
    import re
    s = re.sub(r"[^\w\s-]", "", (s or "").strip().lower())
    return re.sub(r"[\s_-]+", "-", s).strip("-")[:60] or "book"


def _unique_slug(root: Path, title: str | None) -> str:
    """A readable, collision-free folder name from the title (never shown to
    the user, who only ever sees the title)."""
    import uuid
    base = _slugify(title)
    if not (root / base).exists():
        return base
    for _ in range(50):
        cand = f"{base}-{uuid.uuid4().hex[:4]}"
        if not (root / cand).exists():
            return cand
    return f"{base}-{uuid.uuid4().hex}"


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


class FindingEdit(BaseModel):
    replacement: str | None = None   # user-authored fix ("" deletes the quote)


class Cleanup(BaseModel):
    # NOTE: request-body models MUST live at module scope — defined inside
    # create_app, future-annotations make FastAPI treat them as query params
    categories: list[str] = []
    videos: list[str] = []           # video ids whose SOURCE file to delete


class CaptionEdit(BaseModel):
    caption: str = ""


class SwapCaptions(BaseModel):
    a: int
    b: int


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
    anthropic_enabled: bool = True


def _iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def create_app(root: Path) -> FastAPI:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)   # projects folder made on demand
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

    @app.get("/api/lookup")
    def book_lookup(q: str):
        """Look a book up by ISBN, title, or author and return candidates to
        auto-fill the new-project form. Google Books first (keyless, broad),
        Open Library as fallback."""
        import re

        import httpx
        q = (q or "").strip()
        if len(q) < 3:
            return {"results": []}
        digits = re.sub(r"[^0-9Xx]", "", q)
        is_isbn = len(digits) in (10, 13) and digits[:-1].isdigit()
        results: list[dict] = []
        try:
            gq = f"isbn:{digits}" if is_isbn else q
            r = httpx.get("https://www.googleapis.com/books/v1/volumes",
                          params={"q": gq, "maxResults": 5}, timeout=8.0)
            for it in (r.json().get("items") or [])[:5]:
                vi = it.get("volumeInfo") or {}
                ids = {x.get("type"): x.get("identifier")
                       for x in vi.get("industryIdentifiers") or []}
                results.append({
                    "title": vi.get("title"),
                    "author": ", ".join(vi.get("authors") or []) or None,
                    "year": (vi.get("publishedDate") or "")[:4] or None,
                    "pages": vi.get("pageCount") or None,
                    "publisher": vi.get("publisher"),
                    "isbn": ids.get("ISBN_13") or ids.get("ISBN_10"),
                    "cover_url": (vi.get("imageLinks") or {}).get("thumbnail"),
                })
        except Exception:
            pass
        if not results:
            try:
                key = "isbn" if is_isbn else "q"
                r = httpx.get("https://openlibrary.org/search.json",
                              params={key: digits if is_isbn else q,
                                      "limit": 5, "fields": "title,author_name,"
                                      "first_publish_year,number_of_pages_median,"
                                      "isbn,cover_i"}, timeout=8.0)
                for d in (r.json().get("docs") or [])[:5]:
                    cover = d.get("cover_i")
                    results.append({
                        "title": d.get("title"),
                        "author": ", ".join(d.get("author_name") or []) or None,
                        "year": str(d.get("first_publish_year") or "") or None,
                        "pages": d.get("number_of_pages_median"),
                        "publisher": None,
                        "isbn": (d.get("isbn") or [None])[0],
                        "cover_url": (f"https://covers.openlibrary.org/b/id/"
                                      f"{cover}-M.jpg" if cover else None),
                    })
            except Exception:
                pass
        return {"results": [r for r in results if r.get("title")]}

    @app.post("/api/projects")
    def new_project(spec: NewProject):
        name = spec.name or _unique_slug(root, spec.title)
        target = root / name
        if (target / "manifest.json").exists():
            raise HTTPException(409, "project already exists")
        for v in spec.videos:
            if not Path(v.path).exists():
                raise HTTPException(400, f"video not found: {v.path}")
        book_meta = {"author": spec.author, "isbn": spec.isbn,
                     "publisher": spec.publisher, "year": spec.year}
        create_project(target, [v.model_dump() for v in spec.videos],
                       title=spec.title, expected_pages=spec.expected_pages,
                       book=book_meta)
        return {"ok": True, "name": name}

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

    def _build_signature(ws: Workspace) -> str:
        """Fingerprint of everything a built output depends on: the assembled
        book text, every referenced figure file, proof states, and whether
        assemble is even up to date. Any page/figure/proof change after a
        build makes the output stale."""
        import hashlib
        import re as _re
        h = hashlib.sha1()
        h.update(ws.stage_status("assemble").encode())
        book = ws.work_file("book.md")
        if book.exists():
            text = book.read_text(encoding="utf-8")
            h.update(text.encode())
            for rel in sorted(set(_re.findall(r"\]\((figures/[^)]+)\)", text))):
                f = ws.root / rel
                h.update(rel.encode())
                if f.exists():
                    h.update(str(f.stat().st_mtime_ns).encode())
        pdir = ws.work_file("proof")
        if pdir.exists():
            for f in sorted(pdir.glob("proof_*.json")):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                h.update(f"{f.name}|{d.get('status')}|{d.get('base_hash')}|"
                         f"{len(d.get('proofed_md') or '')}".encode())
        cover = next((p for p in ws.manifest["pages"]
                      if p.get("role") == "cover"), None)
        if cover and cover.get("color"):
            f = ws.root / cover["color"]
            if f.exists():
                h.update(str(f.stat().st_mtime_ns).encode())
        return h.hexdigest()[:16]

    @app.get("/api/projects/{name}")
    def project_detail(name: str):
        from ..review import page_reasons
        ws = ws_for(name)
        m = ws.manifest
        sig = _build_signature(ws)
        built = m.get("outputs_built", {})
        # offer to auto-fill book details from an ISBN found in the OCR — only
        # when the book has no ISBN yet and the user hasn't dismissed it.
        # Scan once and cache (ISBNs don't change).
        detected_isbn = None
        if not m["book"].get("isbn") and not m.get("isbn_dismissed"):
            if "isbn_detected" not in m:
                from ..project import find_isbn
                m["isbn_detected"] = find_isbn(ws) or ""
                ws.save()
            detected_isbn = m.get("isbn_detected") or None
        # per-file versions: thumbnail URLs embed these so a re-crop or
        # re-patch changes the URL and can never be served from memory cache
        file_v: dict[str, int] = {}
        for p in m["pages"]:
            for rel in ([p.get("color"), p.get("patched_source")]
                        + (p.get("figures") or [])):
                if rel and rel not in file_v:
                    try:
                        file_v[rel] = (ws.root / rel).stat().st_mtime_ns \
                            // 1_000_000
                    except OSError:
                        pass
        return {
            "file_versions": file_v,
            "name": name,
            "book": m["book"],
            "detected_isbn": detected_isbn,
            "toc": printed_toc(ws),
            "videos": m["videos"],
            "stages": {s: ws.stage_status(s) for s in STAGES},
            "pages": [{**p, "reasons": page_reasons(p)} for p in m["pages"]],
            "running": name in runs and runs[name]["thread"].is_alive(),
            "outputs": [{"name": f.name, "stale": built.get(f.name) != sig}
                        for f in ws.dir("out").glob("*") if f.is_file()],
            "contact_sheet": (ws.work_file("contact_sheet.jpg")).exists(),
        }

    @app.patch("/api/projects/{name}")
    def edit_project(name: str, edit: BookEdit):
        """Update book metadata: title, author, expected page count."""
        ws = ws_for(name)
        fields = edit.model_dump(exclude_unset=True)
        for k in ("title", "author", "expected_pages", "isbn", "publisher", "year"):
            if k in fields:
                v = fields[k]
                ws.manifest["book"][k] = (v.strip() or None) if isinstance(v, str) else v
        ws.save()
        return {"ok": True, "book": ws.manifest["book"]}

    @app.post("/api/projects/{name}/dismiss-isbn")
    def dismiss_isbn(name: str):
        ws = ws_for(name)
        ws.manifest["isbn_dismissed"] = True
        ws.save()
        return {"ok": True}

    @app.delete("/api/projects/{name}")
    def delete_project(name: str, confirm: str = ""):
        """Delete a whole project, videos/frames/pages and all. Irreversible —
        the caller must echo the exact project name in `confirm`."""
        import shutil
        ws = ws_for(name)   # 404s if the project doesn't exist
        if confirm != name:
            raise HTTPException(400, "type the exact project name to confirm")
        if name in runs and runs[name]["thread"].is_alive():
            raise HTTPException(409, "pipeline is running — stop it first")
        runs.pop(name, None)
        shutil.rmtree(ws.root, ignore_errors=True)
        return {"ok": True}

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
            "anthropic_enabled": bool(p.get("anthropic_enabled", True)),
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
            "anthropic_enabled": s.anthropic_enabled,
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

    @app.post("/api/projects/{name}/pdf")
    async def add_pdf(name: str, pdf: UploadFile):
        """Import a whole book from a PDF — the alternative to filming it."""
        ws = ws_for(name)
        if name in runs and runs[name]["thread"].is_alive():
            raise HTTPException(409, "pipeline is running — wait for it to finish")
        cfg = load_config(ws.root)
        uploads = ws.root / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / Path(pdf.filename or "book.pdf").name
        with open(dest, "wb") as f:
            while chunk := await pdf.read(1 << 22):
                f.write(chunk)
        from ..project import add_pages_from_pdf
        try:
            n = await asyncio.to_thread(add_pages_from_pdf, ws, cfg, dest,
                                        lambda m: None)
        except Exception as e:
            raise HTTPException(400, f"PDF import failed: {e}")
        finally:
            try:
                dest.unlink(missing_ok=True)   # never let cleanup mask success
            except OSError:
                pass
        return {"ok": True, "pages": n,
                "next": "run the pipeline to transcribe them"}

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
        # regions/figures described the OLD image — the new read defines them
        for key in ("confidence", "flags", "transcribe_error",
                    "regions", "figures"):
            page.pop(key, None)

        from ..project import fix_photo_orientation
        from ..stages.preprocess import preprocess_page
        await asyncio.to_thread(preprocess_page, ws, page, cfg)
        rotated = await asyncio.to_thread(
            fix_photo_orientation, ws, cfg, page, lambda m: None)
        ws.stage_reset("transcribe")  # the deferred page must transcribe next run
        ws.save()
        return {"ok": True, "rotated": rotated,
                "transcription": "deferred — run the pipeline"}

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
        section_changed = "section" in fields
        if section_changed:
            s = (edit.section or "").strip()
            if s:
                page["section"] = s
            else:
                page.pop("section", None)
            ws.stage_reset("assemble")   # the book structure changed
            if page["status"] == "suspect":
                page["status"] = "ok"
        _reconcile(ws)
        if section_changed:
            # regenerate book.md now so the new chapter shows in proof/output
            # immediately (assemble is cheap — no model calls)
            try:
                from ..stages.assemble import run as assemble_run
                assemble_run(ws, load_config(ws.root), log=lambda m: None)
            except Exception:
                pass
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

    def _compute_crop(ws, page, edit: CropEdit, source_rel=None):
        """Shared crop math: bbox crops directly; a 4-corner quad is
        perspective-warped back to a rectangle. Returns (crop, bbox, quad).
        Crops from `source_rel` when given (a standalone close-up figure),
        otherwise the page's corrected full image."""
        import cv2
        import numpy as np

        from ..imaging import order_quad
        from ..stages.preprocess import correct_page

        src = source_rel or page.get("color")
        if not src:
            raise HTTPException(400, "page has no corrected image")
        color = cv2.imread(str(ws.root / src))
        if color is None:
            raise HTTPException(500, "image unreadable")
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

    def _set_caption(ws, page, rel, region, caption):
        """Set a figure's caption on its region AND in the page markdown's
        image alt text (the alt text is what becomes the EPUB figcaption)."""
        import re as _re
        caption = (caption or "").strip()
        if region is not None:
            region["caption"] = caption
            region["caption_manual"] = True
        if page.get("md") and (ws.root / page["md"]).exists():
            mdp = ws.root / page["md"]
            md = mdp.read_text(encoding="utf-8")
            pat = _re.compile(r"!\[[^\]]*\]\(" + _re.escape(rel) + r"\)")
            repl = f"![{caption}]({rel})"
            if pat.search(md):
                md = pat.sub(lambda _m: repl, md)
                mdp.write_text(md, encoding="utf-8")

    @app.post("/api/projects/{name}/pages/{page_id}/figures/{fig_idx}/caption")
    def set_figure_caption(name: str, page_id: str, fig_idx: int,
                           edit: CaptionEdit):
        """Assign/correct one figure's caption."""
        ws = ws_for(name)
        page, rel, region = _figure(ws, page_id, fig_idx)
        _set_caption(ws, page, rel, region, edit.caption)
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True}

    @app.post("/api/projects/{name}/pages/{page_id}/figures/swap-captions")
    def swap_figure_captions(name: str, page_id: str, spec: SwapCaptions):
        """Swap the captions of two figures on the same page (fixes the
        model attaching the right caption to the wrong image)."""
        ws = ws_for(name)
        pa, rela, rega = _figure(ws, page_id, spec.a)
        _pb, relb, regb = _figure(ws, page_id, spec.b)
        ca = (rega or {}).get("caption", "")
        cb = (regb or {}).get("caption", "")
        _set_caption(ws, pa, rela, rega, cb)
        _set_caption(ws, pa, relb, regb, ca)
        ws.stage_reset("assemble")
        ws.save()
        return {"ok": True}

    _phash_cache: dict[str, tuple[float, int]] = {}

    @app.get("/api/projects/{name}/figures/duplicate-clusters")
    def figure_duplicate_clusters(name: str):
        """Which same-page figures are ACTUALLY the same image (near-identical
        perceptual hash) vs merely on the same page. Only real clusters get
        the destructive keep-sharpest option."""
        import cv2

        from ..imaging import hamming, phash64

        ws = ws_for(name)

        def ph(rel):
            f = ws.root / rel
            try:
                mt = f.stat().st_mtime
            except OSError:
                return None
            hit = _phash_cache.get(rel)
            if hit and hit[0] == mt:
                return hit[1]
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            h = phash64(cv2.resize(img, (32, 32)))
            _phash_cache[rel] = (mt, h)
            return h

        by_num: dict = {}
        for p in ws.manifest["pages"]:
            if p.get("status") in ("duplicate", "deleted"):
                continue
            n = p.get("printed_number")
            if n is None:
                continue
            for i, rel in enumerate(p.get("figures") or []):
                by_num.setdefault(n, []).append(
                    {"page_id": p["id"], "fig_idx": i, "rel": rel,
                     "hash": ph(rel)})

        clusters = []
        for figs in by_num.values():
            if len(figs) < 2:
                continue
            used = set()
            for i in range(len(figs)):
                if i in used or figs[i]["hash"] is None:
                    continue
                group = [i]
                for j in range(i + 1, len(figs)):
                    if (j not in used and figs[j]["hash"] is not None
                            and hamming(figs[i]["hash"], figs[j]["hash"]) <= 10):
                        group.append(j)
                        used.add(j)
                if len(group) >= 2:
                    used.update(group)
                    clusters.append([{"page_id": figs[k]["page_id"],
                                      "fig_idx": figs[k]["fig_idx"]}
                                     for k in group])
        return {"clusters": clusters}

    @app.post("/api/projects/{name}/pages/{page_id}/figures/{fig_idx}/crop")
    def recrop_figure(name: str, page_id: str, fig_idx: int, edit: CropEdit):
        """Re-crop a figure. A re-acquired close-up (own_image) is its OWN
        photo, not a region of the page — so trim the close-up itself instead
        of cropping the full page (which would destroy the close-up)."""
        import cv2

        ws = ws_for(name)
        page, rel, region = _figure(ws, page_id, fig_idx)
        standalone = bool(region and region.get("own_image"))
        # crop the close-up in place via a temp (can't read+write same file mid-op)
        src_rel = None
        if standalone and (ws.root / rel).exists():
            tmp = ws.work_file(f"_crop_{page_id}_{fig_idx}.png")
            import shutil
            shutil.copyfile(ws.root / rel, tmp)
            src_rel = tmp.relative_to(ws.root).as_posix()
        crop, bbox, stored_quad = _compute_crop(ws, page, edit, source_rel=src_rel)
        cv2.imwrite(str(ws.root / rel), crop)
        if src_rel:
            (ws.root / src_rel).unlink(missing_ok=True)
        if region is not None:
            from ..stages.figures import file_ref
            region["user_crop"] = True
            region.pop("stale_crop", None)
            if standalone:
                region["own_image"] = True   # still a standalone close-up
            else:
                region["bbox_norm"] = bbox
                if stored_quad:
                    region["quad_norm"] = stored_quad
                region["color_ref"] = file_ref(ws.root / page["color"])
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

    @app.post("/api/projects/{name}/figures/ai-refine")
    async def ai_refine_figures(name: str):
        """Claude-vision pass over every figure whose crop is NOT a current
        manual one: Claude proposes the corners from the page image, the CV
        refiner snaps them tight, and the crop is regenerated. A manual crop
        made on the CURRENT page image is the most recent human input and is
        never touched; a manual crop orphaned by a later re-patch (stale) IS
        eligible — that's exactly how the bad crops happened."""
        import cv2

        from ..imaging import refine_figure_bbox
        from ..stages.figures import file_ref

        ws = ws_for(name)
        cfg = load_config(ws.root)
        if not cfg["provider"].get("anthropic_api_key"):
            raise HTTPException(400, "add an Anthropic API key in settings first")
        from ..backends import anthropic_enabled
        if not anthropic_enabled(cfg):
            raise HTTPException(400, "the Anthropic API is disabled in settings "
                                     "— enable it to run AI refine")
        from ..backends.anthropic_backend import claude_figure_boxes

        def iou(a, b):
            x0, y0 = max(a[0], b[0]), max(a[1], b[1])
            x1, y1 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            ua = ((a[2] - a[0]) * (a[3] - a[1])
                  + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            return inter / ua if ua > 0 else 0.0

        def work():
            refined, skipped_manual, failed = 0, 0, 0
            for page in ws.manifest["pages"]:
                if (page.get("status") in ("duplicate", "deleted")
                        or not page.get("color")):
                    continue
                regions = page.get("regions") or []
                todo = []
                cur_ref = None
                for ri, region in enumerate(regions):
                    if region.get("deleted"):
                        continue
                    if region.get("user_crop"):
                        # a manual crop is NEVER touched by AI — even one that
                        # looks stale. (Older manual crops predate color_ref;
                        # assuming those were orphaned destroyed real work.)
                        skipped_manual += 1
                        continue
                    todo.append((ri, region))
                if not todo:
                    continue
                color = cv2.imread(str(ws.root / page["color"]))
                if color is None:
                    continue
                h, w = color.shape[:2]
                scale = 1400.0 / max(h, w)
                small = (cv2.resize(color, (int(w * scale), int(h * scale)))
                         if scale < 1 else color)
                ok, buf = cv2.imencode(".jpg", small,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                try:
                    boxes = claude_figure_boxes(cfg, buf.tobytes())
                except Exception:
                    failed += len(todo)
                    continue
                used: set[int] = set()
                ref = file_ref(ws.root / page["color"])
                for ri, region in todo:
                    best_i, best = None, 0.05
                    for bi, b in enumerate(boxes):
                        if bi in used:
                            continue
                        v = iou(b["bbox"], region.get("bbox_norm") or [0, 0, 1, 1])
                        if v > best:
                            best_i, best = bi, v
                    if best_i is None:   # no overlap info — largest unused box
                        free = [bi for bi in range(len(boxes)) if bi not in used]
                        if not free:
                            failed += 1
                            continue
                        best_i = max(free, key=lambda bi: (
                            (boxes[bi]["bbox"][2] - boxes[bi]["bbox"][0])
                            * (boxes[bi]["bbox"][3] - boxes[bi]["bbox"][1])))
                    used.add(best_i)
                    box = boxes[best_i]["bbox"]
                    # pass 2: zoom into the proposed area and ask again — a
                    # tight box on a zoomed crop is far more accurate than
                    # one on the whole page (pass 1 sometimes returns the
                    # entire page)
                    bw, bh = box[2] - box[0], box[3] - box[1]
                    zx0 = max(0.0, box[0] - 0.08 * bw)
                    zy0 = max(0.0, box[1] - 0.08 * bh)
                    zx1 = min(1.0, box[2] + 0.08 * bw)
                    zy1 = min(1.0, box[3] + 0.08 * bh)
                    zoom = color[int(zy0 * h):int(zy1 * h),
                                 int(zx0 * w):int(zx1 * w)]
                    if zoom.size:
                        zh, zw = zoom.shape[:2]
                        zs = 1200.0 / max(zh, zw)
                        zsmall = (cv2.resize(zoom, (int(zw * zs), int(zh * zs)))
                                  if zs < 1 else zoom)
                        ok2, buf2 = cv2.imencode(".jpg", zsmall,
                                                 [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok2:
                            try:
                                b2 = claude_figure_boxes(cfg, buf2.tobytes())
                            except Exception:
                                b2 = []
                            if b2:
                                big = max(b2, key=lambda f: (
                                    (f["bbox"][2] - f["bbox"][0])
                                    * (f["bbox"][3] - f["bbox"][1])))["bbox"]
                                zarea = ((big[2] - big[0]) * (big[3] - big[1]))
                                if zarea < 0.93:   # Claude actually tightened
                                    box = [zx0 + big[0] * (zx1 - zx0),
                                           zy0 + big[1] * (zy1 - zy0),
                                           zx0 + big[2] * (zx1 - zx0),
                                           zy0 + big[3] * (zy1 - zy0)]
                    def area(b):
                        return (b[2] - b[0]) * (b[3] - b[1])

                    snapped = refine_figure_bbox(color, box)
                    if snapped is not None:
                        if iou(snapped, box) > 0.5:
                            box = snapped   # CV agrees — take the tighter edges
                        elif area(box) > 0.65 and area(snapped) < area(box):
                            # Claude returned ~the whole page; the CV snap
                            # found the actual figure inside it
                            box = snapped
                    # a "figure" that is basically the whole page is a miss
                    if (area(box) > 0.85
                            or ((box[2] - box[0]) > 0.93
                                and (box[3] - box[1]) > 0.93)):
                        failed += 1
                        continue
                    px = (int(box[0] * w), int(box[1] * h),
                          int(box[2] * w), int(box[3] * h))
                    if px[2] - px[0] < 30 or px[3] - px[1] < 30:
                        failed += 1
                        continue
                    rel = f"figures/{page['id']}_{chr(97 + ri % 26)}.png"
                    cv2.imwrite(str(ws.root / rel),
                                color[px[1]:px[3], px[0]:px[2]])
                    region["bbox_norm"] = [float(v) for v in box]
                    region["auto_refined"] = True
                    region["ai_crop"] = True
                    region["color_ref"] = ref
                    for k in ("user_crop", "stale_crop", "quad_norm"):
                        region.pop(k, None)
                    figs = page.setdefault("figures", [])
                    if rel not in figs:
                        figs.append(rel)
                    refined += 1
            ws.stage_reset("assemble")
            ws.save()
            return {"refined": refined, "kept_manual": skipped_manual,
                    "failed": failed}

        r = await asyncio.to_thread(work)
        return {"ok": True, **r}

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

        def _decode_write():   # 12MP decode off the event loop
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return False
            # keep filename: markdown link stays valid
            cv2.imwrite(str(ws.root / rel), img)
            return True

        if not await asyncio.to_thread(_decode_write):
            raise HTTPException(400, "not a readable image")
        if region is not None:
            region["user_crop"] = True  # never let the figures stage overwrite it
            region["own_image"] = True  # an uploaded photo, not a page crop
            region.pop("needs_reshoot", None)   # re-acquisition fulfilled
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
        page = ws.page(page_id)
        if page is not None and page.get("purged"):
            raise HTTPException(409, "this page's images were cleaned up — "
                                     "re-photograph it instead")
        from ..project import set_page_deleted
        try:
            set_page_deleted(ws, page_id, False)
        except KeyError:
            raise HTTPException(404, "no such page")
        return {"ok": True}

    @app.post("/api/projects/{name}/pages/{page_id}/crop-page")
    async def crop_page_endpoint(name: str, page_id: str, edit: CropEdit):
        """Crop a photo page's own pixels (covers!) with the corner tool."""
        ws = ws_for(name)
        page = ws.page(page_id)
        if page is None:
            raise HTTPException(404, "no such page")
        if not page.get("patched_source"):
            raise HTTPException(400, "only photo-sourced pages can be cropped "
                                     "— video pages are framed automatically")
        if not edit.quad_norm or len(edit.quad_norm) != 4:
            raise HTTPException(400, "need quad_norm corners")
        cfg = load_config(ws.root)
        from ..project import crop_page_photo
        try:
            await asyncio.to_thread(crop_page_photo, ws, cfg, page,
                                    edit.quad_norm, lambda m: None)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/projects/{name}/pages/{page_id}/rotate")
    async def rotate_page_photo(name: str, page_id: str, degrees: int = 180):
        """Rotate a photo page's pixels (EXIF tags on phone photos are
        unreliable); processed images re-derive, transcription re-runs."""
        ws = ws_for(name)
        page = ws.page(page_id)
        if page is None:
            raise HTTPException(404, "no such page")
        if not page.get("patched_source"):
            raise HTTPException(400, "video-sourced page — flip its video on "
                                     "the media tab instead")
        cfg = load_config(ws.root)
        from ..project import rotate_patch
        try:
            await asyncio.to_thread(rotate_patch, ws, cfg, page, degrees,
                                    lambda m: None)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(400, str(e))
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
            from ..project import fix_photo_orientation
            await asyncio.to_thread(fix_photo_orientation, ws, cfg, page,
                                    lambda m: None)
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
            page = ws.page(page_id)
            if page is None:
                raise HTTPException(404, "no such page")
            rel = f"figures/{page_id}_{chr(97 + fig_idx % 26)}.png"

            def _decode_write():   # 12MP decode off the event loop
                import cv2
                import numpy as np
                img = cv2.imdecode(np.frombuffer(tmp.read_bytes(), np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is None:
                    return False
                cv2.imwrite(str(ws.root / rel), img)
                return True

            if not await asyncio.to_thread(_decode_write):
                raise HTTPException(400, "not a readable image")
            if rel not in (page.get("figures") or []):
                page.setdefault("figures", []).append(rel)
            regions = page.get("regions") or []
            if fig_idx < len(regions):
                regions[fig_idx]["user_crop"] = True
                regions[fig_idx]["own_image"] = True   # a standalone close-up
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
            page.pop("regions", None)   # described the old image
            page.pop("figures", None)
            # the wizard told the user "photograph page N" and they did —
            # that's a human confirmation of the number, even if the printed
            # folio isn't visible on the page (full-page plates often have
            # none). Without this the next reconcile drops the number.
            if number is not None:
                page["printed_number"] = number
                page["number_manual"] = True
            for key in ("confidence", "flags", "transcribe_error"):
                page.pop(key, None)
            from ..project import fix_photo_orientation
            from ..stages.preprocess import preprocess_page
            await asyncio.to_thread(preprocess_page, ws, page, cfg)
            await asyncio.to_thread(fix_photo_orientation, ws, cfg, page,
                                    lambda m: None)
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

    # ---------------- storage report + cleanup

    def _storage_report(ws: Workspace) -> dict:
        pages = ws.manifest["pages"]
        referenced: set[str] = set()
        for p in pages:
            for fid in (p.get("cluster_frames") or []):
                referenced.add(fid)
            for k in ("canonical", "fallback"):
                if p.get(k):
                    referenced.add(p[k])
        live_ids = {p["id"] for p in pages}

        def size(f: Path) -> int:
            try:
                return f.stat().st_size
            except OSError:
                return 0

        videos = []
        for v in ws.manifest["videos"]:
            f = ws.root / v["path"]
            videos.append({
                "id": v["id"], "path": v["path"], "size": size(f),
                "exists": f.exists(),
                "deleted": bool(v.get("source_deleted")),
                "deletable": (f.exists()
                              and ws.stage_status("extract") == "done"),
                "frames": v.get("frames_extracted"),
            })

        cats: dict[str, dict] = {}

        def cat(key, label, files, note=""):
            cats[key] = {"label": label, "note": note,
                         "count": len(files),
                         "size": sum(size(f) for f in files),
                         "_files": files}

        frames_dir = ws.root / "frames"
        unused_frames = []
        if frames_dir.exists():
            for f in frames_dir.rglob("*.jpg"):
                fid = f"{f.parent.name}_{f.stem}"
                if fid not in referenced:
                    unused_frames.append(f)
        cat("frames_unused", "Extracted frames no page uses", unused_frames,
            "turn-motion debris and dropped clusters — nothing references them")

        hidden_files = []
        n_hidden = 0
        for p in pages:
            if p.get("status") != "deleted":
                continue
            n_hidden += 1
            rels = ([p.get("patched_source"), p.get("color"), p.get("llm_image"),
                     p.get("md")] + (p.get("figures") or []))
            for rel in rels:
                if rel and (ws.root / rel).exists():
                    hidden_files.append(ws.root / rel)
        cat("hidden_pages", "Hidden pages (images + page entries)", hidden_files,
            f"removes all {n_hidden} hidden pages from the project entirely — "
            f"a clean page list, no un-hide afterwards")

        dup_files = []
        n_dups = 0
        for p in pages:
            if p.get("status") != "duplicate":
                continue
            n_dups += 1
            for rel in ([p.get("patched_source"), p.get("color"),
                         p.get("llm_image"), p.get("md")]
                        + (p.get("figures") or [])):
                if rel and (ws.root / rel).exists():
                    dup_files.append(ws.root / rel)
        cat("duplicate_pages", "Duplicate pages (superseded captures)",
            dup_files,
            f"removes all {n_dups} duplicates — each has a better surviving "
            f"capture of the same page; 'not a duplicate' rescue is gone after")

        orphans = []
        pages_dir = ws.root / "work" / "pages"
        if pages_dir.exists():
            for f in pages_dir.glob("*.*"):
                pid = f.name.split("_")[0]
                if pid not in live_ids:
                    orphans.append(f)
        cat("orphans", "Working images of pages that no longer exist", orphans)

        thumbs = list((ws.root / "work" / "thumbs").glob("*.jpg")) \
            if (ws.root / "work" / "thumbs").exists() else []
        cat("thumbs", "Thumbnail cache", thumbs, "regenerated on demand")

        uploads = [f for f in (ws.root / "uploads").glob("*")
                   if f.is_file()] if (ws.root / "uploads").exists() else []
        cat("uploads", "Leftover upload temp files", uploads)

        return {"videos": videos, "categories": cats}

    @app.get("/api/projects/{name}/storage")
    def storage(name: str):
        r = _storage_report(ws_for(name))
        for c in r["categories"].values():
            c.pop("_files", None)
        return r

    @app.post("/api/projects/{name}/cleanup")
    def cleanup(name: str, req: Cleanup):
        ws = ws_for(name)
        r = _storage_report(ws)
        freed = 0

        def rm(f: Path) -> int:
            try:
                n = f.stat().st_size
                f.unlink()
                return n
            except OSError:
                return 0

        for key in req.categories:
            c = r["categories"].get(key)
            if not c:
                continue
            for f in c["_files"]:
                freed += rm(f)
            if key in ("hidden_pages", "duplicate_pages"):
                status = ("deleted" if key == "hidden_pages" else "duplicate")
                # remove the page entries themselves — a clean project. The
                # deleted_captures markers make sure a future re-cluster that
                # rediscovers these frames hides them again automatically.
                from ..stages.transcribe import cache_key
                dl = ws.manifest.setdefault("deleted_captures", [])
                gone = [p for p in ws.manifest["pages"]
                        if p.get("status") == status]
                for p in gone:
                    ck = cache_key(p)
                    if ck and ck not in dl:
                        dl.append(ck)
                ws.manifest["pages"] = [
                    p for p in ws.manifest["pages"]
                    if p.get("status") != status]
        for vid in req.videos:
            v = next((v for v in ws.manifest["videos"] if v["id"] == vid), None)
            info = next((x for x in r["videos"] if x["id"] == vid), None)
            if v is None or info is None or not info["deletable"]:
                continue
            freed += rm(ws.root / v["path"])
            v["source_deleted"] = True   # extracted frames/pages live on
        ws.save()
        return {"ok": True, "freed": freed}

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
    def proof_finding(name: str, idx: int, fi: int, enabled: bool,
                      apply_all: bool = False, edit: FindingEdit | None = None):
        """Per-finding accept/reject/author-a-fix: rebuilds the proofed copy."""
        from ..proofread import toggle_finding
        ws = ws_for(name)
        set_repl = edit is not None and "replacement" in edit.model_fields_set
        try:
            return toggle_finding(
                ws, idx, fi, enabled, apply_all=apply_all,
                replacement=edit.replacement if set_repl else None,
                set_replacement=set_repl)
        except (FileNotFoundError, IndexError) as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/projects/{name}/proof/{idx}/finding/{fi}/resolve")
    async def proof_resolve(name: str, idx: int, fi: int):
        """Re-read a garbled passage from its source page image."""
        from ..proofread import resolve_finding
        ws = ws_for(name)
        cfg = load_config(ws.root)
        try:
            return await asyncio.to_thread(resolve_finding, ws, cfg, idx, fi)
        except (FileNotFoundError, IndexError, LookupError) as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(502, f"re-read failed: {e}")

    @app.post("/api/projects/{name}/proof/{idx}/reread-stuck")
    async def proof_reread_stuck(name: str, idx: int):
        """Re-read every stuck page-anchored finding in this chapter."""
        from ..proofread import reread_chapter_stuck
        ws = ws_for(name)
        cfg = load_config(ws.root)
        try:
            return await asyncio.to_thread(reread_chapter_stuck, ws, cfg, idx)
        except (FileNotFoundError, IndexError) as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(409, str(e))
        except Exception as e:
            raise HTTPException(502, f"re-read failed: {e}")

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
        ext = {"epub": "epub", "markdown": "zip"}.get(format, "pdf")
        suffix = {"pdf-facsimile": "-facsimile", "markdown": "-markdown"}.get(format, "")
        if device != "none" and format not in ("pdf", "markdown"):
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
            elif format == "markdown":
                from ..build_markdown import build_markdown_zip
                build_markdown_zip(ws, out,
                                   author=ws.manifest["book"].get("author"),
                                   log=lambda m: None)
            else:
                raise HTTPException(400, f"unknown format {format!r}")
        except (RuntimeError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        # record what this output was built from — any later page/figure/
        # proof change makes it stale
        ws.manifest.setdefault("outputs_built", {})[out.name] = \
            _build_signature(ws)
        ws.save()
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

    @app.get("/api/version")
    def app_version():
        """The frontend polls this and reloads itself when the served app
        changed — stale-script tabs caused endless 'feature isn't there'."""
        try:
            return {"v": str((static_dir / "index.html").stat().st_mtime_ns)}
        except OSError:
            return {"v": "0"}

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
