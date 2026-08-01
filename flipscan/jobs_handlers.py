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
#   tts    — audiobook synthesis. Serial, and its own lane so an hours-long
#            narration neither blocks nor waits behind OCR work. (Both use the
#            GPU: run a pipeline and an audiobook at the same time only if your
#            VRAM fits both models.)
KIND_LANES = {
    "proof-chapter": "proof",
    "cast-analysis": "proof",   # text-LLM work, parallel-safe like proofreads
    "pdf-import": "import",
    "video-import": "import",
    "audiobook": "tts",
    "tts-preview": "tts",       # shares the GPU lane so it never fights a build
    "voice-gen": "tts",         # Parler needs the GPU too — same serial lane
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
    return {"main": 1, "proof": max(1, n), "import": 1, "tts": 1}, KIND_LANES


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

    def audiobook(project, params, log, should_cancel):
        import re
        import time
        from .audiobook import build_audiobook
        from .outputs import record_output
        from .stages.assemble import run as assemble_run
        ws = ws_for(project)
        cfg = load_config(ws.root)
        # voice: a name from the shared library (root/voices/<name>.wav) or
        # empty for the engine's built-in narrator
        vname = (params.get("voice") or "").strip()
        voice = ""
        if vname:
            from .audiobook import resolve_voice
            vpath = resolve_voice(ws, root / "voices", vname)
            if vpath is None:
                raise FileNotFoundError(f"voice {vname!r} not found")
            voice = str(vpath)
        # narrate from a book.md that reflects the current pages (same
        # regenerate-before-packaging rule as the build endpoint)
        assemble_run(ws, cfg, log=lambda m: None)
        # hours of synthesis must never overwrite an earlier run: the filename
        # carries the voice, the speed, and the generation time
        try:
            speed = float(params.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        use_cast = bool(params.get("use_cast"))
        chapters = [int(x) for x in str(params.get("chapters") or "").split(",")
                    if x.strip().isdigit()] or None
        vslug = re.sub(r"[^A-Za-z0-9_-]+", "-", vname) if vname else "builtin"
        stamp = time.strftime("%Y%m%d-%H%M")
        sslug = f"--{speed:g}x" if speed != 1.0 else ""
        cslug = "--cast" if use_cast else ""
        chslug = ""
        if chapters:
            # name the sample after the chapter's TITLE — the internal index
            # ("ch9") means nothing to a listener whose file opens with
            # "Chapter Five"
            from .audiobook import narration_chapters
            titles = [t for i, (t, _) in enumerate(narration_chapters(ws))
                      if i in chapters]
            tslug = "-".join(re.sub(r"[^A-Za-z0-9]+", "-", t).strip("-")[:24]
                             for t in titles) or "chapters"
            chslug = f"--sample-{tslug}"
        out = (ws.dir("out")
               / f"{ws.root.name}--{vslug}{sslug}{cslug}{chslug}--{stamp}.m4b")
        log(f"voice: {vname or 'built-in narrator'}"
            + (f", {speed:g}x speed" if speed != 1.0 else "")
            + (", full cast" if use_cast else "")
            + (f", chapters {chapters}" if chapters else "") + f" -> {out.name}")
        build_audiobook(ws, cfg, out, voice=voice, speed=speed,
                        use_cast=use_cast, voices_dir=root / "voices",
                        chapters=chapters, log=log, should_cancel=should_cancel)
        # stamp what it was built from, so the output tab's stale/current badge
        # is honest (without this the m4b reads "stale" forever)
        record_output(ws, out.name)
        return {"file": out.name}

    jobq.register("pipeline", pipeline)
    jobq.register("proof-chapter", proof_chapter)
    jobq.register("proof-resolve", proof_resolve)
    jobq.register("proof-reread-stuck", proof_reread_stuck)
    jobq.register("retry-ocr", retry_ocr)
    def cast_analysis(project, params, log, should_cancel):
        from .casting import analyze_book
        ws = ws_for(project)
        cfg = load_config(ws.root)
        cast = analyze_book(ws, cfg, log=log, should_cancel=should_cancel,
                            only_failed=bool(params.get("only_failed")))
        return {"characters": len(cast.get("characters") or {}),
                "quotes": sum(c["quotes"] for c
                              in (cast.get("characters") or {}).values())}

    def tts_preview(project, params, log, should_cancel):
        import hashlib
        from .audiobook import resolve_voice, synthesize_preview
        ws = ws_for(project)
        cfg = load_config(ws.root)
        vname = (params.get("voice") or "").strip()
        voice, vhash = "", ""
        if vname:
            vp = resolve_voice(ws, root / "voices", vname)
            if vp is None:
                raise FileNotFoundError(f"voice {vname!r} not found")
            voice = str(vp)
            vhash = hashlib.sha1(vp.read_bytes()).hexdigest()[:8]
        text = (params.get("text") or "").strip()
        pdir = root / "voices" / "previews"
        # the voice-bytes hash keys the cache, so two books' local voices that
        # share a character name never collide
        key = hashlib.sha1(f"{vname}|{vhash}|{text}".encode("utf-8")).hexdigest()[:12]
        out = pdir / f"{vname or 'builtin'}--{key}.wav"
        if not out.exists():
            log(f"preview: {vname or 'built-in narrator'} — synthesizing…")
            synthesize_preview(cfg, text, voice, out)
        else:
            log("preview: cached")
        return {"file": out.name}

    def voice_gen(project, params, log, should_cancel):
        import re as _re
        from .casting import assign_voice, load_cast
        from .voicegen import build_description, generate_voice_sample
        ws = ws_for(project)
        character = (params.get("character") or "").strip()
        cast = load_cast(ws)
        ch = (cast or {}).get("characters", {}).get(character)
        if ch is None:
            raise FileNotFoundError(f"character {character!r} not in the cast")
        vname = _re.sub(r"[^A-Za-z0-9 _-]+", "", character).strip() or "generated"
        # generated character voices are BOOK-scoped — Count Zeppelin belongs
        # to his book, not every book's voice menu
        out = ws.root / "voices" / f"{vname}.wav"
        desc = build_description(ch.get("description", ""),
                                 ch.get("sounds_like", ""))
        generate_voice_sample(desc, out, log=log)
        assign_voice(ws, character, vname)   # cast it immediately
        log(f"voice {vname!r} added to the library and assigned to {character}")
        return {"voice": vname}

    jobq.register("pdf-import", pdf_import)
    jobq.register("video-import", video_import)
    jobq.register("audiobook", audiobook)
    jobq.register("cast-analysis", cast_analysis)
    jobq.register("tts-preview", tts_preview)
    jobq.register("voice-gen", voice_gen)
