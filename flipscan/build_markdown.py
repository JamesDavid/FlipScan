"""Markdown export: a zip of the assembled book (Markdown) with its images.

Layout inside the zip — the portable convention every Markdown editor
(Obsidian, Typora, VS Code, GitHub, Pandoc) understands:

    <title>.md         YAML frontmatter (title/author) + the book text
    images/            only the figures the book actually references

Image links are rewritten from the internal figures/ path to images/, and
only referenced files are bundled (no orphan crops).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .workspace import Workspace

_IMG = re.compile(r"(!\[[^\]]*\]\()(figures/[^)]+)(\))")


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s.-]", "", s).strip()
    return re.sub(r"\s+", "_", s) or "book"


def build_markdown_zip(ws: Workspace, out_path: Path,
                       title: str | None = None, author: str | None = None,
                       log=print) -> Path:
    book_md = ws.work_file("book.md")
    if not book_md.exists():
        raise RuntimeError("work/book.md missing — run the pipeline (assemble) first")
    text = book_md.read_text(encoding="utf-8")

    title = title or ws.manifest["book"].get("title") or ws.root.name
    author = author or ws.manifest["book"].get("author")

    # collect referenced figures, rewrite links figures/ -> images/
    referenced: dict[str, str] = {}   # original rel -> images/<name>

    def rewrite(m: re.Match) -> str:
        rel = m.group(2)
        name = Path(rel).name
        referenced[rel] = f"images/{name}"
        return f"{m.group(1)}images/{name}{m.group(3)}"

    body = _IMG.sub(rewrite, text)

    # YAML frontmatter — standard, read by Obsidian/Jekyll/Hugo/Pandoc
    def yaml_escape(v: str) -> str:
        return '"' + v.replace('"', '\\"') + '"'

    front = ["---", f"title: {yaml_escape(title)}"]
    if author:
        front.append(f"author: {yaml_escape(author)}")
    front += ["---", "", ""]
    md = "\n".join(front) + body

    md_name = f"{_slug(title)}.md"
    n_imgs = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_name, md)
        # include the cover image too, if this book has one
        cover = next((p for p in ws.manifest["pages"]
                      if p.get("role") == "cover" and p.get("color")), None)
        if cover and (ws.root / cover["color"]).exists():
            z.write(ws.root / cover["color"], "images/cover.png")
            n_imgs += 1
        for rel, dest in referenced.items():
            src = ws.root / rel
            if src.exists():
                z.write(src, dest)
                n_imgs += 1
    log(f"markdown zip written: {out_path} ({md_name} + {n_imgs} images)")
    return out_path
