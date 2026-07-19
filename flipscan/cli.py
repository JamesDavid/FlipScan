"""FlipScan CLI."""

from __future__ import annotations

import importlib
from pathlib import Path

import click

from .config import load_config
from .ffmpeg import probe_video
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
              help="Source video. Repeat for two-pass capture.")
@click.option("--pages", "pages_list", multiple=True,
              type=click.Choice(["odd", "even", "all"]),
              help="Which pages the corresponding --video captures (default: all).")
@click.option("--direction", "directions", multiple=True,
              type=click.Choice(["forward", "reverse"]),
              help="Flip direction of the corresponding --video (default: forward).")
@click.option("--reverse", is_flag=True, help="Shorthand: single video shot back-to-front.")
@click.option("--title", default=None, help="Book title (EPUB metadata).")
@click.option("--expected-pages", type=int, default=None,
              help="Expected page count for gap detection.")
def init(directory: Path, videos, pages_list, directions, reverse, title, expected_pages):
    """Create a workspace: copy videos in, probe fps, write manifest.json."""
    if pages_list and len(pages_list) != len(videos):
        raise click.UsageError("--pages must be given once per --video (or not at all)")
    if directions and len(directions) != len(videos):
        raise click.UsageError("--direction must be given once per --video (or not at all)")

    video_entries = []
    ws = Workspace.create(directory, videos=[], title=title, expected_pages=expected_pages)
    for i, src in enumerate(videos):
        vid = f"v{i}"
        pages = pages_list[i] if pages_list else "all"
        direction = directions[i] if directions else ("reverse" if reverse else "forward")
        click.echo(f"{vid}: importing {src} (pages={pages}, direction={direction})")
        rel = ws.import_video(src, vid)
        meta = probe_video(ws.root / rel)
        click.echo(f"{vid}: {meta['fps_actual']} fps, {meta.get('nb_frames') or '?'} frames, "
                   f"{meta.get('width')}x{meta.get('height')}")
        video_entries.append({
            "id": vid,
            "path": str(rel).replace("\\", "/"),
            "source": str(src),
            "pages": pages,
            "direction": direction,
            **meta,
        })
    ws.manifest["videos"] = video_entries
    ws.save()
    click.echo(f"Workspace ready: {ws.root} — next: flipscan run {ws.root}")


# ---------------------------------------------------------------- run

# stage name -> implementing module (registered as milestones land)
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


def _run_stage(ws: Workspace, cfg: dict, stage: str) -> None:
    mod = importlib.import_module(STAGE_MODULES[stage])
    mod.run(ws, cfg, log=click.echo)


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

    stages = [only_stage] if only_stage else STAGES
    for stage in stages:
        try:
            importlib.import_module(STAGE_MODULES[stage])
        except ModuleNotFoundError:
            click.echo(f"[{stage}] not implemented yet, skipping")
            continue
        if not only_stage and not force and ws.stage_status(stage) == "done":
            click.echo(f"[{stage}] done, skipping")
            continue
        click.echo(f"[{stage}] running")
        _run_stage(ws, cfg, stage)
        click.echo(f"[{stage}] ok")


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


# ---------------------------------------------------------------- build

@main.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None,
              help="Output file (default: out/<workspace>.<ext>)")
@click.option("--format", "formats", multiple=True,
              type=click.Choice(["epub", "pdf", "pdf-facsimile"]),
              help="Output format; repeatable (default: epub)")
@click.option("--title", default=None)
@click.option("--author", default=None)
def build(directory: Path, output, formats, title, author):
    """Build the book (epub / pdf / pdf-facsimile) from the assembled markdown."""
    ws = Workspace.open(directory)
    formats = formats or ("epub",)
    for fmt in formats:
        ext = "epub" if fmt == "epub" else "pdf"
        if output and len(formats) == 1:
            out = output
        else:
            suffix = "" if fmt != "pdf-facsimile" else "-facsimile"
            out = ws.dir("out") / f"{ws.root.name}{suffix}.{ext}"
        if fmt == "epub":
            from .build_epub import build_epub
            build_epub(ws, out, title=title, author=author, log=click.echo)
        elif fmt == "pdf-facsimile":
            from .build_pdf import build_pdf_facsimile
            build_pdf_facsimile(ws, out, title=title, log=click.echo)
        else:
            from .build_pdf import build_pdf_reflowed
            build_pdf_reflowed(ws, out, title=title, log=click.echo)


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


if __name__ == "__main__":
    main()
