"""Job handlers shared by the web app's in-process worker and the standalone
`flipscan worker` process.

Each handler opens its own workspace from the projects root, so it needs no
FastAPI app — the exact same code runs whether the worker lives inside the web
server or in a separate container. That's what lets docker-compose run a
dedicated `worker` service so web restarts never interrupt a running job.
"""

from __future__ import annotations

import os
from pathlib import Path

from .config import load_config
from .jobs import JobCanceled, JobQueue
from .project import retry_ocr_page, run_pipeline
from .workspace import Workspace


# Lanes group jobs by how they may overlap:
#   main   — LLM + manifest work (pipeline, page re-reads, retry-OCR). Serial:
#            one at a time so they never race manifest.json or flood the model.
#   proof  — chapter proofreads. Each writes its own per-chapter file, so they
#            run several at once.
#   import — PDF/video ingest. No LLM; a same-project import can't overlap that
#            project's pipeline (blocked by the 409 gate), so this lane runs
#            independently of `main` — an import never waits behind another
#            book's pipeline.
KIND_LANES = {
    "proof-chapter": "proof",
    "pdf-import": "import",
    "video-import": "import",
}


def concurrency_config() -> tuple[dict[str, int], dict[str, str]]:
    """(lane_caps, kind_lanes) for the JobQueue. 'proof' runs several chapter
    proofreads at once (default 3, override with FLIPSCAN_PROOF_CONCURRENCY);
    'main' and 'import' are serial within themselves but independent of each
    other."""
    try:
        n = int(os.environ.get("FLIPSCAN_PROOF_CONCURRENCY", "3"))
    except ValueError:
        n = 3
    return {"main": 1, "proof": max(1, n), "import": 1}, KIND_LANES


def register_handlers(jobq: JobQueue, root: Path) -> None:
    """Register every job `kind` on the queue. Call this in the process that
    owns the worker (the web app, or `flipscan worker`)."""
    root = Path(root).resolve()

    def ws_for(name: str) -> Workspace:
        target = (root / name).resolve()
        if not str(target).startswith(str(root)) or not (target / "manifest.json").exists():
            raise FileNotFoundError(f"no project {name!r}")
        return Workspace.open(target)

    def pipeline(project, params, log, should_cancel):
        ws = ws_for(project)
        cfg = load_config(ws.root)
        if params.get("provider"):
            cfg["provider"]["name"] = params["provider"]

        def cb(m):
            if should_cancel():
                raise JobCanceled()
            log(str(m))

        run_pipeline(ws, cfg, force=params.get("force", False), log=cb)
        log("[pipeline] finished")

    def proof_chapter(project, params, log, should_cancel):
        from .proofread import proofread_chapter
        ws = ws_for(project)
        cfg = load_config(ws.root)
        d = proofread_chapter(ws, cfg, int(params["idx"]))
        log(f"chapter {params['idx']}: proofread complete "
            f"({len(d.get('findings', []))} findings)")

    def proof_resolve(project, params, log, should_cancel):
        from .proofread import resolve_finding
        ws = ws_for(project)
        cfg = load_config(ws.root)
        d = resolve_finding(ws, cfg, int(params["idx"]), int(params["fi"]))
        log("re-read complete")
        return d

    def proof_reread_stuck(project, params, log, should_cancel):
        from .proofread import reread_chapter_stuck
        ws = ws_for(project)
        cfg = load_config(ws.root)
        d = reread_chapter_stuck(ws, cfg, int(params["idx"]))
        log(f"re-read stuck findings: {d.get('rescued', 0)} auto-fixed, "
            f"{d.get('still_manual', 0)} still need you")
        return d

    def retry_ocr(project, params, log, should_cancel):
        ws = ws_for(project)
        retry_ocr_page(ws, params["page_id"])
        log(f"retry OCR complete for {params['page_id']}")

    def pdf_import(project, params, log, should_cancel):
        from .project import add_pages_from_pdf
        ws = ws_for(project)
        cfg = load_config(ws.root)
        dest = Path(params["path"])
        try:
            n = add_pages_from_pdf(ws, cfg, dest, log)
        finally:
            try:
                dest.unlink(missing_ok=True)   # don't let cleanup mask success
            except OSError:
                pass
        log(f"imported {n} pages from {dest.name}")
        return {"pages": n}

    def video_import(project, params, log, should_cancel):
        from .project import add_video
        ws = ws_for(project)
        entry = add_video(ws, Path(params["path"]), log=log)
        log(f"added video {entry['id']}")
        return {"id": entry["id"]}

    jobq.register("pipeline", pipeline)
    jobq.register("proof-chapter", proof_chapter)
    jobq.register("proof-resolve", proof_resolve)
    jobq.register("proof-reread-stuck", proof_reread_stuck)
    jobq.register("retry-ocr", retry_ocr)
    jobq.register("pdf-import", pdf_import)
    jobq.register("video-import", video_import)
