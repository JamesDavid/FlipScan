"""PDF outputs.

pdf-facsimile: corrected page images as pages with the transcription embedded
as an invisible text layer -> searchable, layout-exact, robust to transcription
errors. Positioning is line-flow (no word-level boxes), which is enough for
search/select/copy.

pdf (reflowed): rendered from work/book.md via weasyprint, falling back to
pandoc if installed.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

from .workspace import Workspace


def build_pdf_facsimile(ws: Workspace, out_path: Path, title: str | None = None,
                        log=print) -> Path:
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    pages = [p for p in ws.manifest["pages"] if p.get("color")]
    if not pages:
        raise RuntimeError("no preprocessed pages — run the pipeline first")

    c = canvas.Canvas(str(out_path))
    c.setTitle(title or ws.manifest["book"].get("title") or ws.root.name)
    for page in pages:
        img_path = ws.root / page["color"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            iw, ih = im.size
        # scale to a ~US-letter-width page, preserving aspect
        pw = 612.0
        ph = pw * ih / iw
        c.setPageSize((pw, ph))
        c.drawImage(ImageReader(str(img_path)), 0, 0, pw, ph)

        md = ""
        if page.get("md") and (ws.root / page["md"]).exists():
            md = (ws.root / page["md"]).read_text(encoding="utf-8")
        if md:
            text = c.beginText(36, ph - 36)
            text.setTextRenderMode(3)  # invisible — search/copy layer only
            text.setFont("Helvetica", 9)
            usable_lines = max(1, int((ph - 72) / 11))
            lines = []
            for para in md.splitlines():
                lines.extend(textwrap.wrap(para, width=100) or [""])
            for line in lines[:usable_lines]:
                text.textLine(line)
            c.drawText(text)
        c.showPage()
    c.save()
    log(f"facsimile PDF written: {out_path} ({len(pages)} pages, searchable)")
    return out_path


def build_pdf_reflowed(ws: Workspace, out_path: Path, title: str | None = None,
                       log=print) -> Path:
    book_md = ws.work_file("book.md")
    if not book_md.exists():
        raise RuntimeError("work/book.md missing — run the pipeline (assemble) first")

    try:
        import markdown as md_lib
        from weasyprint import HTML  # needs GTK/Pango native libs
    except (ImportError, OSError):
        pandoc = shutil.which("pandoc")
        if pandoc:
            subprocess.run(
                [pandoc, str(book_md), "-o", str(out_path),
                 "--metadata", f"title={title or ws.root.name}",
                 "--resource-path", str(ws.root)],
                check=True,
            )
            log(f"reflowed PDF written via pandoc: {out_path}")
            return out_path
        raise RuntimeError(
            "reflowed PDF needs weasyprint (pip install 'flipscan[pdf]' + GTK libs) "
            "or pandoc on PATH. The pdf-facsimile format has no such dependency."
        )

    html_body = md_lib.markdown(book_md.read_text(encoding="utf-8"),
                                extensions=["tables"])
    css = ("body{font-family:Georgia,serif;font-size:11pt;line-height:1.5;margin:2cm}"
           "img{max-width:100%}h1{page-break-before:always}")
    HTML(string=f"<style>{css}</style>{html_body}",
         base_url=str(ws.root)).write_pdf(str(out_path))
    log(f"reflowed PDF written via weasyprint: {out_path}")
    return out_path
