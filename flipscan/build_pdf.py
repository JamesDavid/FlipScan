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

# Page geometry (mm) for the reflowed PDF, keyed by device. The reMarkable 2
# canvas and the e-ink typography choices below are adapted from reCompose
# (MIT-licensed, github.com/mrodger/reCompose): the RM2 viewer won't crop
# margins, so the page must be pre-sized to the exact panel; on e-ink you also
# want darker "grayscale" tones (light grays wash out), generous line spacing
# so lines don't blend, and no widows/orphans on the small page.
PAGE_GEOMETRY_MM: dict[str, dict] = {
    "remarkable-2": {"w": 157.8, "h": 210.4, "margin": 10.0, "eink": True},
    "eink-6in":     {"w": 90.0,  "h": 122.0, "margin": 7.0,  "eink": True},
    "xteink-x3":    {"w": 61.0,  "h": 101.0, "margin": 5.0,  "eink": True},
    "xteink-x4":    {"w": 61.0,  "h": 101.0, "margin": 5.0,  "eink": True},
    "tablet":       {"w": 148.0, "h": 210.0, "margin": 14.0, "eink": False},
}


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
                       device: str = "none", log=print) -> Path:
    """Reflowed text PDF straight from book.md via reportlab — no weasyprint/
    GTK/pandoc needed. A device preset (e.g. remarkable-2) sizes the page to
    that panel and switches to an e-ink-optimised palette/spacing; e-ink
    geometry and typography adapted from reCompose (MIT, github.com/mrodger)."""
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

    geo = PAGE_GEOMETRY_MM.get(device)
    if geo:
        page_size = (geo["w"] * mm, geo["h"] * mm)
        margin = geo["margin"] * mm
        eink = geo["eink"]
    else:
        page_size, margin, eink = A5, 15 * mm, False

    # e-ink washes out light grays and blends tight lines, so darken the
    # secondary tones and open up the leading; on the small page suppress
    # widows/orphans. Print/tablet keeps the softer greys.
    quote_col = "#333333" if eink else "#555555"
    cap_col = "#333333" if eink else "#444444"
    lead = 16.5 if eink else 15.5
    nowid = {"allowWidows": 0, "allowOrphans": 0} if eink else {}

    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11,
                          leading=lead, spaceAfter=6, **nowid)
    quote = ParagraphStyle("quote", parent=body, leftIndent=8 * mm,
                           textColor=quote_col)
    h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=20,
                        leading=24, spaceAfter=12, **nowid)
    h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=14,
                        leading=18, spaceBefore=8, spaceAfter=6, **nowid)
    cap = ParagraphStyle("cap", parent=body, fontName="Times-Italic",
                         fontSize=9.5, leading=12, spaceBefore=2,
                         alignment=1, textColor=cap_col)

    def inline(s: str) -> str:
        s = escape(s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
        return s

    page_w = page_size[0] - 2 * margin  # usable width inside margins

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
                    # fit the text frame, leaving ~1/4 page for caption headroom
                    max_h = (page_size[1] - 2 * margin) * 0.72
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

    doc = SimpleDocTemplate(str(out_path), pagesize=page_size,
                            leftMargin=margin, rightMargin=margin,
                            topMargin=margin, bottomMargin=margin,
                            title=title or ws.manifest["book"].get("title")
                            or ws.root.name)
    doc.build(story)
    log(f"reflowed PDF written: {out_path}"
        + (f" ({device})" if geo else ""))
    return out_path
