"""FlipScan CLI."""

from __future__ import annotations

from pathlib import Path

import click

from .config import load_config
from .workspace import STAGES, Workspace


@click.group()
@click.version_option()
def main():
    """FlipScan: slow-mo book flip video -> EPUB."""


# ---------------------------------------------------------------- init

@main.command()
@click.argument("directory", type=click.Path(path_type=Path))
@click.option("--video", "videos", multiple=True, required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Source video. Repeat to add several; pages captured in "
                   "more than one video merge automatically (best capture wins).")
@click.option("--direction", "directions", multiple=True,
              type=click.Choice(["forward", "reverse"]),
              help="Optional flip-direction hint per --video (default: forward; "
                   "order is fixed automatically from page matches and printed numbers).")
@click.option("--reverse", is_flag=True, help="Shorthand: single video shot back-to-front.")
@click.option("--title", default=None, help="Book title (EPUB metadata).")
@click.option("--expected-pages", type=int, default=None,
              help="Expected page count for gap detection.")
def init(directory: Path, videos, directions, reverse, title, expected_pages):
    """Create a workspace: copy videos in, probe fps, write manifest.json."""
    if directions and len(directions) != len(videos):
        raise click.UsageError("--direction must be given once per --video (or not at all)")

    from .project import create_project
    specs = [
        {
            "path": str(src),
            "direction": directions[i] if directions
                         else ("reverse" if reverse else "forward"),
        }
        for i, src in enumerate(videos)
    ]
    ws = create_project(directory, specs, title=title,
                        expected_pages=expected_pages, log=click.echo)
    click.echo(f"Workspace ready: {ws.root} — next: flipscan run {ws.root}")


# ---------------------------------------------------------------- addvideo

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option("--direction", type=click.Choice(["forward", "reverse"]), default="forward")
@click.option("--upside-down", is_flag=True, help="Video was shot rotated 180 degrees.")
def addvideo(directory: Path, video: Path, direction: str, upside_down: bool):
    """Add another capture video — shared pages merge, new pages slot in.

    Keep adding videos until every page is covered; duplicate captures collapse
    via printed page numbers after transcription."""
    ws = Workspace.open(directory)
    from .project import add_video
    add_video(ws, video, direction=direction, rotate=180 if upside_down else 0,
              log=click.echo)
    click.echo(f"video added — run `flipscan run {directory}` to merge it in")


# ---------------------------------------------------------------- run

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--stage", "only_stage", type=click.Choice(STAGES), default=None,
              help="Run a single stage (implies re-running it).")
@click.option("--force", is_flag=True, help="Re-run stages even if already done.")
@click.option("--provider", type=click.Choice(["ollama", "anthropic", "hybrid", "mock"]),
              default=None)
@click.option("--model", default=None, help="Override the transcription model name.")
@click.option("--ollama-url", default=None)
def run(directory: Path, only_stage, force, provider, model, ollama_url):
    """Run the pipeline (extract -> ... -> assemble), resuming where it left off."""
    ws = Workspace.open(directory)
    cfg = load_config(ws.root)
    if provider:
        cfg["provider"]["name"] = provider
    if ollama_url:
        cfg["provider"]["ollama_url"] = ollama_url
    if model:
        key = "anthropic_model" if cfg["provider"]["name"] == "anthropic" else "ollama_model"
        cfg["provider"][key] = model

    from .project import run_pipeline
    run_pipeline(ws, cfg, only_stage=only_stage, force=force, log=click.echo)


# ---------------------------------------------------------------- review

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
def review(directory: Path):
    """Generate the HTML review page (frame vs markdown + reshoot list)."""
    ws = Workspace.open(directory)
    from .review import generate_review, reshoot_list
    out = generate_review(ws, log=click.echo)
    items = reshoot_list(ws)
    if items:
        click.echo(f"reshoot list ({len(items)} pages): "
                   + ", ".join(i["id"] for i in items))
    else:
        click.echo("reshoot list: empty — all pages look good")
    click.echo(f"open {out}")


# ---------------------------------------------------------------- patch

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--page", "page_id", required=True, help="Page id, e.g. p0142")
@click.argument("image", type=click.Path(exists=True, path_type=Path))
def patch(directory: Path, page_id: str, image: Path):
    """Replace a page's capture with a re-shot photo and re-process it."""
    import shutil

    ws = Workspace.open(directory)
    cfg = load_config(ws.root)
    page = ws.page(page_id)
    if page is None:
        raise click.UsageError(f"no page {page_id!r} in {ws.root}")

    patches = ws.root / "patches"
    patches.mkdir(exist_ok=True)
    dest = patches / f"{page_id}{image.suffix.lower()}"
    shutil.copy2(image, dest)
    page["patched_source"] = f"patches/{dest.name}"
    page["status"] = "patched"
    for key in ("md", "confidence", "flags", "transcribe_error"):
        page.pop(key, None)
    page["md"] = None

    click.echo(f"{page_id}: preprocessing replacement photo")
    from .stages.preprocess import preprocess_page
    preprocess_page(ws, page, cfg)
    ws.save()

    click.echo(f"{page_id}: transcribing")
    from .stages.transcribe import run as transcribe_run
    transcribe_run(ws, cfg, log=click.echo)

    ws.stage_reset("figures")  # re-run figures + assemble with the new page
    click.echo(f"{page_id}: patched — run `flipscan run {directory}` then "
               f"`flipscan build {directory}` to rebuild outputs")


# ---------------------------------------------------------------- addpage

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.argument("image", type=click.Path(exists=True, path_type=Path))
@click.option("--position", default="end",
              help='"start", "end", or a page index (default: end)')
@click.option("--cover", is_flag=True,
              help="Use this photo as the book cover (EPUB cover image, "
                   "excluded from body text)")
def addpage(directory: Path, image: Path, position: str, cover: bool):
    """Add a page from a photo — covers, inside-cover text, or missed pages."""
    ws = Workspace.open(directory)
    cfg = load_config(ws.root)
    from .project import add_page_from_photo
    page = add_page_from_photo(ws, cfg, image, position=position,
                               role="cover" if cover else None, log=click.echo)
    click.echo(f"{page['id']} added — run `flipscan run {directory}` then "
               f"`flipscan build {directory}` to rebuild outputs")


# ---------------------------------------------------------------- delpage

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.argument("page_id")
@click.option("--restore", is_flag=True, help="Un-delete a previously deleted page.")
def delpage(directory: Path, page_id: str, restore: bool):
    """Delete an erroneous page from the book (soft: restorable, survives re-runs)."""
    ws = Workspace.open(directory)
    from .project import set_page_deleted
    try:
        set_page_deleted(ws, page_id, not restore)
    except KeyError:
        raise click.UsageError(f"no page {page_id!r}")
    click.echo(f"{page_id} {'restored' if restore else 'deleted'} — "
               f"run `flipscan build {directory}` to rebuild outputs")


# ---------------------------------------------------------------- build

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None,
              help="Output file (default: out/<workspace>.<ext>)")
@click.option("--format", "formats", multiple=True,
              type=click.Choice(["epub", "pdf", "pdf-latex", "pdf-facsimile"]),
              help="Output format; repeatable (default: epub). pdf-latex is the "
                   "high-quality pandoc+XeLaTeX PDF (needs those tools installed).")
@click.option("--title", default=None)
@click.option("--author", default=None)
@click.option("--device", default="none",
              type=click.Choice(["none", "xteink-x3", "xteink-x4", "eink-6in",
                                 "remarkable-2", "tablet"]),
              help="Size for a target reader (images + reflowed/LaTeX PDF page).")
def build(directory: Path, output, formats, title, author, device):
    """Build the book (epub / pdf / pdf-latex / pdf-facsimile) from markdown."""
    ws = Workspace.open(directory)
    formats = formats or ("epub",)
    for fmt in formats:
        ext = "epub" if fmt == "epub" else "pdf"
        if output and len(formats) == 1:
            out = output
        else:
            suffix = {"pdf-facsimile": "-facsimile", "pdf-latex": "-latex"}.get(fmt, "")
            if device != "none":
                suffix += f"-{device}"
            out = ws.dir("out") / f"{ws.root.name}{suffix}.{ext}"
        if fmt == "epub":
            from .build_epub import build_epub
            build_epub(ws, out, title=title, author=author, device=device,
                       log=click.echo)
        elif fmt == "pdf-facsimile":
            from .build_pdf import build_pdf_facsimile
            build_pdf_facsimile(ws, out, title=title, device=device, log=click.echo)
        elif fmt == "pdf-latex":
            from .build_pdf_latex import build_pdf_latex
            build_pdf_latex(ws, out, title=title, author=author, device=device,
                            log=click.echo)
        else:
            from .build_pdf import build_pdf_reflowed
            build_pdf_reflowed(ws, out, title=title, device=device, log=click.echo)


# ---------------------------------------------------------------- status

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
def status(directory: Path):
    """Show pipeline stage status and page counts."""
    ws = Workspace.open(directory)
    for stage in STAGES:
        click.echo(f"{stage:12s} {ws.stage_status(stage)}")
    pages = ws.manifest["pages"]
    if pages:
        suspects = [p["id"] for p in pages if p.get("status") == "suspect"]
        click.echo(f"pages: {len(pages)} ({len(suspects)} suspect)")


# ---------------------------------------------------------------- ui

@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="Directory containing project workspaces (default: FLIPSCAN_ROOT or cwd)")
@click.option("--host", default="0.0.0.0",
              help="Bind address (default 0.0.0.0 = reachable from other "
                   "devices on your network; use 127.0.0.1 for this machine only)")
@click.option("--port", type=int, default=8321)
def ui(root, host, port):
    """Start the local web GUI (requires `pip install flipscan[ui]`)."""
    import os
    import socket

    try:
        from .ui import serve
    except ImportError:
        raise click.ClickException(
            "GUI dependencies missing — install with: pip install 'flipscan[ui]'")
    root = root or Path(os.environ.get("FLIPSCAN_ROOT", "."))
    root.mkdir(parents=True, exist_ok=True)   # ensure the projects folder exists
    click.echo(f"FlipScan GUI  (projects root: {root.resolve()})")
    click.echo(f"  this machine:  http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan_ip = s.getsockname()[0]
            s.close()
            click.echo(f"  your network:  http://{lan_ip}:{port}  (phone, tablet, ...)")
        except OSError:
            pass
    serve(root, host=host, port=port)


# ---------------------------------------------------------------- worker

@main.command()
@click.option("--root", type=click.Path(path_type=Path), default=None,
              help="Directory containing project workspaces (default: FLIPSCAN_ROOT or cwd)")
def worker(root):
    """Run the durable background job worker as its own process.

    The pipeline, proofreads, and page re-reads run as durable jobs in a
    SQLite queue (`jobs.db` in the projects root). `flipscan ui` already runs
    an in-process worker, so you only need this to run the worker separately —
    e.g. a second docker-compose service — so restarting the web server never
    interrupts a running job. Point it at the SAME root as the web server and
    set FLIPSCAN_EXTERNAL_WORKER=1 on the web server so it doesn't double up.
    """
    import os
    import time

    from .jobs import JobQueue
    from .jobs_handlers import concurrency_config, register_handlers

    root = root or Path(os.environ.get("FLIPSCAN_ROOT", "."))
    root.mkdir(parents=True, exist_ok=True)
    lane_caps, kind_lanes = concurrency_config()
    jobq = JobQueue(root / "jobs.db", lane_caps=lane_caps, kind_lanes=kind_lanes)
    register_handlers(jobq, root)
    resumed = jobq.requeue_orphans()
    click.echo(f"FlipScan worker  (projects root: {root.resolve()})")
    click.echo(f"  resumed {resumed} orphaned job(s); waiting for work…")
    jobq.start_worker()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        jobq.stop()
        click.echo("worker stopped")


if __name__ == "__main__":
    main()
