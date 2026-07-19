"""flipscan ui — local web GUI. Thin client over the same stage functions the CLI uses."""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..config import load_config, save_global_config
from ..project import create_project, run_pipeline
from ..workspace import STAGES, Workspace

# workspace files the browser may fetch, by top-level directory
SERVABLE = {"frames", "work", "figures", "review", "out", "pages", "videos"}


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


class Settings(BaseModel):
    provider: str = "ollama"
    ollama_url: str = ""
    ollama_model: str = ""
    anthropic_model: str = ""
    anthropic_api_key: str = ""


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

    @app.get("/api/projects/{name}")
    def project_detail(name: str):
        ws = ws_for(name)
        m = ws.manifest
        return {
            "name": name,
            "book": m["book"],
            "videos": m["videos"],
            "stages": {s: ws.stage_status(s) for s in STAGES},
            "pages": m["pages"],
            "running": name in runs and runs[name]["thread"].is_alive(),
            "outputs": [f.name for f in ws.dir("out").glob("*") if f.is_file()],
            "contact_sheet": (ws.work_file("contact_sheet.jpg")).exists(),
        }

    # ---------------- pipeline

    @app.post("/api/projects/{name}/run")
    def run_project(name: str, provider: str | None = None, force: bool = False):
        ws = ws_for(name)
        if name in runs and runs[name]["thread"].is_alive():
            raise HTTPException(409, "already running")
        q: queue.Queue = queue.Queue()
        cfg = load_config(ws.root)
        if provider:
            cfg["provider"]["name"] = provider

        def work():
            try:
                run_pipeline(ws, cfg, force=force, log=lambda m: q.put(str(m)))
                q.put("[pipeline] finished")
            except Exception as e:  # surface errors into the log stream
                q.put(f"[pipeline] ERROR: {e}")
            finally:
                q.put(None)

        t = threading.Thread(target=work, daemon=True)
        runs[name] = {"queue": q, "thread": t}
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
    def events(name: str):
        if name not in runs:
            raise HTTPException(404, "no active run")
        q = runs[name]["queue"]

        def stream():
            while True:
                try:
                    msg = q.get(timeout=120)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if msg is None:
                    yield "event: done\ndata: done\n\n"
                    break
                yield f"data: {json.dumps(msg)}\n\n"

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
        entry = add_video(ws, dest, log=lambda m: None)
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
        page = add_page_from_photo(ws, cfg, tmp, position=position,
                                   role="cover" if cover else None,
                                   log=lambda m: None)
        tmp.unlink(missing_ok=True)
        return {"ok": True, "id": page["id"]}

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

    @app.post("/api/projects/{name}/pages/{page_id}/patch")
    async def patch_page(name: str, page_id: str, photo: UploadFile):
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
        from ..stages.transcribe import run as transcribe_run
        preprocess_page(ws, page, cfg)
        ws.save()
        transcribe_run(ws, cfg, log=lambda m: None)
        ws.stage_reset("figures")
        return {"ok": True}

    @app.get("/api/projects/{name}/reshoot")
    def reshoot(name: str):
        from ..review import reshoot_list
        return reshoot_list(ws_for(name))

    # ---------------- build + files

    @app.post("/api/projects/{name}/build")
    def build(name: str, format: str = "epub"):
        ws = ws_for(name)
        ext = "epub" if format == "epub" else "pdf"
        suffix = "-facsimile" if format == "pdf-facsimile" else ""
        out = ws.dir("out") / f"{ws.root.name}{suffix}.{ext}"
        try:
            if format == "epub":
                from ..build_epub import build_epub
                build_epub(ws, out, log=lambda m: None)
            elif format == "pdf-facsimile":
                from ..build_pdf import build_pdf_facsimile
                build_pdf_facsimile(ws, out, log=lambda m: None)
            elif format == "pdf":
                from ..build_pdf import build_pdf_reflowed
                build_pdf_reflowed(ws, out, log=lambda m: None)
            else:
                raise HTTPException(400, f"unknown format {format!r}")
        except (RuntimeError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "file": out.name}

    @app.get("/api/projects/{name}/file/{path:path}")
    def get_file(name: str, path: str):
        ws = ws_for(name)
        target = (ws.root / path).resolve()
        if not str(target).startswith(str(ws.root.resolve())) or not target.is_file():
            raise HTTPException(404, "not found")
        if path.split("/")[0] not in SERVABLE:
            raise HTTPException(403, "not servable")
        return FileResponse(target)

    # ---------------- frontend

    static = Path(__file__).parent / "static" / "index.html"

    @app.get("/")
    def index():
        # always revalidate the app shell — stale cached UI on phones is worse
        # than the tiny refetch
        return FileResponse(static, headers={"Cache-Control": "no-cache"})

    return app


def serve(root: Path, host: str = "127.0.0.1", port: int = 8321) -> None:
    import uvicorn
    uvicorn.run(create_app(root), host=host, port=port, log_level="warning")
