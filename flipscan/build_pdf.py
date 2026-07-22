"""PDF outputs.

pdf-facsimile: corrected page images as pages with the transcription embedded
as an invisible text layer -> searchable, layout-exact, robust to transcription
errors. Positioning is line-flow (no word-level boxes), which is enough for
search/select/copy.

pdf (reflowed): rendered from work/book.md via reportlab (no extra deps).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from .workspace import Workspace


def build_pdf_facsimile(ws: Workspace, out_path: Path, title: str | None = None,
                        device: str = "none", log=print) -> Path:
    import io

    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    from .device import process_image

    pages = [p for p in ws.manifest["pages"]
             if p.get("color") and p.get("status") not in ("duplicate", "deleted")]
    if not pages:
        raise RuntimeError("no preprocessed pages — run the pipeline first")

    c = canvas.Canvas(str(out_path))
    c.setTitle(title or ws.manifest["book"].get("title") or ws.root.name)
    for page in pages:
        img_path = ws.root / page["color"]
        if not img_path.exists():
            continue
        if device != "none":
            data, _ = process_image(img_path.read_bytes(), device)
            reader = ImageReader(io.BytesIO(data))
            iw, ih = Image.open(io.BytesIO(data)).size
        else:
            reader = ImageReader(str(img_path))
            with Image.open(img_path) as im:
                iw, ih = im.size
        # scale to a ~US-letter-width page, preserving aspect
        pw = 612.0
        ph = pw * ih / iw
        c.setPageSize((pw, ph))
        c.drawImage(reader, 0, 0, pw, ph)

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


def _pdf_sized(ws: Workspace, src: Path, long_edge: int = 1400) -> Path:
    """Downscaled JPEG copy for embedding — full-res PNGs made 260MB PDFs."""
    import cv2
    out_dir = ws.work_file("pdf_imgs")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / (src.stem + ".jpg")
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    img = cv2.imread(str(src))
    if img is None:
        return src
    h, w = img.shape[:2]
    sc = long_edge / max(h, w)
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return out


def build_pdf_reflowed(ws: Workspace, out_path: Path, title: str | None = None,
                       log=print) -> Path:
    """Reflowed text PDF straight from book.md via reportlab — no weasyprint/
    GTK/pandoc needed (they were never installed on a normal setup)."""
    import re
    from xml.sax.saxutils import escape

    from reportlab.lib.pagesizes import A5
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer)
    from reportlab.lib.utils import ImageReader

    book_md = ws.work_file("book.md")
    if not book_md.exists():
        raise RuntimeError("work/book.md missing — run the pipeline (assemble) first")
    text = book_md.read_text(encoding="utf-8")

    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11,
                          leading=15.5, spaceAfter=6)
    quote = ParagraphStyle("quote", parent=body, leftIndent=10 * mm,
                           textColor="#555555")
    h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=20,
                        leading=24, spaceAfter=12)
    h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=14,
                        leading=18, spaceBefore=8, spaceAfter=6)
    cap = ParagraphStyle("cap", parent=body, fontName="Times-Italic",
                         fontSize=9.5, leading=12, spaceBefore=2,
                         alignment=1, textColor="#444444")

    def inline(s: str) -> str:
        s = escape(s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
        return s

    page_w = A5[0] - 30 * mm  # usable width inside margins

    story, first_h1 = [], True
    img_re = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        m = img_re.match(block)
        if m:
            src = ws.root / m.group(2)
            if src.exists():
                try:
                    src = _pdf_sized(ws, src)
                    iw, ih = ImageReader(str(src)).getSize()
                    max_h = 440  # fit the A5 frame with caption headroom
                    w = min(page_w, iw)
                    if ih * w / iw > max_h:
                        w = max_h * iw / ih
                    story.append(RLImage(str(src), width=w, height=ih * w / iw))
                    if m.group(1).strip():
                        story.append(Paragraph(inline(m.group(1)), cap))
                    story.append(Spacer(1, 6))
                except Exception:
                    pass
            continue
        if block.startswith("# "):
            if not first_h1:
                story.append(PageBreak())
            first_h1 = False
            story.append(Paragraph(inline(block[2:]), h1))
        elif block.startswith("## "):
            story.append(Paragraph(inline(block[3:]), h2))
        elif block.startswith(">"):
            cleaned = " ".join(ln.lstrip("> ").strip()
                               for ln in block.splitlines())
            story.append(Paragraph(inline(cleaned), quote))
        elif block.startswith("- "):
            for ln in block.splitlines():
                story.append(Paragraph("•&nbsp;&nbsp;" + inline(ln[2:].strip()),
                                       body))
        else:
            story.append(Paragraph(inline(" ".join(block.splitlines())), body))

    doc = SimpleDocTemplate(str(out_path), pagesize=A5,
                            leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title=title or ws.manifest["book"].get("title")
                            or ws.root.name)
    doc.build(story)
    log(f"reflowed PDF written: {out_path}")
    return out_path
