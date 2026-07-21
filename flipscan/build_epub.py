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

    # merge consecutive same-title chapters (a heading repeated across pages
    # is a section header, not a new chapter) and drop heading-only shells
    merged: list[tuple[str, str]] = []
    for t, body in chapters:
        lines = body.splitlines()
        rest = "\n".join(lines[1:]) if lines and lines[0].startswith("# ") else body
        if merged and merged[-1][0] == t:
            merged[-1] = (t, merged[-1][1] + "\n" + rest)
        elif rest.strip():
            merged.append((t, body))
    return merged or [("Book", book_md)]


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

    from .proofread import chapter_hash, load_proof
    proofed_count = 0

    def with_proof(i: int, md: str) -> str:
        """Use the accepted proofed copy — but only while the underlying
        chapter text is unchanged (otherwise the proof is stale)."""
        nonlocal proofed_count
        d = load_proof(ws, i)
        if (d and d.get("status") == "accepted" and d.get("proofed_md")
                and d.get("base_hash") == chapter_hash(md)):
            proofed_count += 1
            return d["proofed_md"]
        return md

    from .stages.assemble import _NUM_WORDS, _TOC_ENTRY

    chs = split_chapters(book_md)
    titles = [t for t, _ in chs]

    def target_for(label: str) -> str | None:
        """Map a printed-contents entry to its chapter file, translating
        'Chapter N' to the book's own opener names (ONE, TWO...)."""
        e = label.strip().casefold()
        cands = [e]
        m = re.match(r"chapter\s+(\d+)$", e)
        if m and int(m.group(1)) < len(_NUM_WORDS):
            w = _NUM_WORDS[int(m.group(1))]
            cands += [w.casefold(), w.title().casefold()]
        for j, t in enumerate(titles):
            if t.strip().casefold() in cands:
                return f"ch{j:03d}.xhtml"
        return None

    def format_contents(body: str) -> str:
        out = []
        for ln in body.splitlines():
            mt = _TOC_ENTRY.match(ln.strip())
            if mt and not ln.lstrip().startswith(("#", "!", "-")):
                title2 = re.sub(r"[.·\s]+$", "", mt.group(1)).strip()
                href = target_for(title2)
                item = f"[{title2}]({href})" if href else title2
                out.append(f"- {item} — {mt.group(2)}")
            else:
                out.append(ln)
        return "\n".join(out)

    chapters = []
    for i, (ch_title, ch_md) in enumerate(chs):
        ch_md = with_proof(i, ch_md)
        if ch_title.strip().casefold() == "contents":
            # single-newline TOC lines render as one run-together paragraph;
            # make it a real, linked list
            ch_md = format_contents(ch_md)
        for rel, epub_name in added_images.items():
            ch_md = ch_md.replace(f"({rel})", f"({epub_name})")
        html = md_lib.markdown(ch_md, extensions=["tables"])
        # figure identity survives the reader's blob-URL rewriting via a data
        # attribute — the in-browser reader maps it back to page + crop slot
        html = re.sub(r'<img([^>]*?)src="images/(p\d+_[a-z])',
                      r'<img data-fig="\2"\1src="images/\2', html)

        # captions live in the markdown alt text, which HTML never displays —
        # render them as visible figcaptions (skipped when the same caption
        # already appears as text nearby, e.g. a printed caption line)
        def _figify(m: re.Match) -> str:
            tag, alt = m.group(0), m.group(2)
            if not alt.strip() or alt.strip() in html_nocap[m.end():m.end() + 400]:
                return tag
            return (f'<figure style="margin:1em 0;text-align:center">{tag}'
                    f'<figcaption style="font-size:.85em;font-style:italic;'
                    f'opacity:.85;margin-top:.3em">{alt}</figcaption></figure>')

        html_nocap = html
        html = re.sub(r'<img([^>]*?)alt="([^"]*)"([^>]*?)/?>', _figify, html)
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
        f"{proofed_count} proofed, {len(added_images)} images)")
    return out_path
