"""flipscan ui — local web GUI. Thin client over the same stage functions the CLI uses."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..config import load_config
from ..project import create_project, run_pipeline
from ..workspace import STAGES, Workspace

# workspace files the browser may fetch, by top-level directory
SERVABLE = {"frames", "work", "figures", "review", "out", "pages", "videos"}


class VideoSpec(BaseModel):
    path: str
    pages: str = "all"
    direction: str = "forward"


class NewProject(BaseModel):
    name: str
    videos: list[VideoSpec]
    title: str | None = None
    expected_pages: int | None = None


class MarkdownEdit(BaseModel):
    markdown: str


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
        return FileResponse(static)

    return app


def serve(root: Path, host: str = "127.0.0.1", port: int = 8321) -> None:
    import uvicorn
    uvicorn.run(create_app(root), host=host, port=port, log_level="warning")
