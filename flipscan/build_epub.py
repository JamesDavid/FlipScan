"""EPUB output: work/book.md -> .epub via ebooklib, chapters from # headings."""

from __future__ import annotations

import re
from pathlib import Path

import markdown as md_lib
from ebooklib import epub

from .workspace import Workspace

_IMG = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def split_chapters(book_md: str) -> list[tuple[str, str]]:
    """Split on level-1 headings; content before the first heading becomes front matter."""
    chapters: list[tuple[str, str]] = []
    title, buf = None, []
    for line in book_md.splitlines():
        if line.startswith("# "):
            if buf or title:
                chapters.append((title or "Front Matter", "\n".join(buf)))
            title, buf = line[2:].strip(), [line]
        else:
            buf.append(line)
    if buf:
        chapters.append((title or "Front Matter", "\n".join(buf)))
    return chapters or [("Book", book_md)]


def build_epub(ws: Workspace, out_path: Path, title: str | None = None,
               author: str | None = None, device: str = "none", log=print) -> Path:
    book_md_path = ws.work_file("book.md")
    if not book_md_path.exists():
        raise FileNotFoundError("work/book.md missing — run the pipeline (assemble) first")
    book_md = book_md_path.read_text(encoding="utf-8")

    book = epub.EpubBook()
    book_title = title or ws.manifest["book"].get("title") or ws.root.name
    book.set_title(book_title)
    book.set_language("en")
    if author:
        book.add_author(author)

    from .device import process_image

    cover_page = next((p for p in ws.manifest["pages"] if p.get("role") == "cover"), None)
    if cover_page and cover_page.get("color") and (ws.root / cover_page["color"]).exists():
        cover_bytes, cover_type = process_image(
            (ws.root / cover_page["color"]).read_bytes(), device)
        ext = ".jpg" if cover_type == "image/jpeg" else ".png"
        book.set_cover(f"cover{ext}", cover_bytes)

    # embed referenced figure images, processed for the target device
    added_images: dict[str, str] = {}
    for rel in set(_IMG.findall(book_md)):
        src = ws.root / rel
        if src.exists():
            data, media_type = process_image(src.read_bytes(), device)
            ext = ".jpg" if media_type == "image/jpeg" else src.suffix
            epub_name = f"images/{src.stem}{ext}"
            book.add_item(epub.EpubItem(
                file_name=epub_name, media_type=media_type, content=data,
            ))
            added_images[rel] = epub_name

    chapters = []
    for i, (ch_title, ch_md) in enumerate(split_chapters(book_md)):
        for rel, epub_name in added_images.items():
            ch_md = ch_md.replace(f"({rel})", f"({epub_name})")
        html = md_lib.markdown(ch_md, extensions=["tables"])
        ch = epub.EpubHtml(title=ch_title, file_name=f"ch{i:03d}.xhtml", lang="en")
        ch.content = f"<html><body>{html}</body></html>"
        book.add_item(ch)
        chapters.append(ch)

    book.toc = chapters
    book.spine = ["nav"] + chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(out_path), book)
    log(f"EPUB written: {out_path} ({len(chapters)} chapters, "
        f"{len(added_images)} images)")
    return out_path
