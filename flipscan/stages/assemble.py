"""Stage 9: assemble — stitch per-page markdown into one book document.

Heals hyphenated words split across pages, merges paragraphs continuing
across page breaks, and strips residual running headers/footers.
Writes work/book.md (the input for all build formats).
"""

from __future__ import annotations

import re
from collections import Counter

from ..workspace import Workspace

_SENTENCE_END = re.compile(r'[.!?:;"”’)\]]\s*$')
_HEADING = re.compile(r"^#{1,6}\s")


def _norm_line(ln: str) -> str:
    """Normalize for running-header comparison: drop markdown heading markers,
    case, punctuation, and spacing — '## DIRIGIBLE DREAMS' and 'Dirigible
    Dreams.' are the same running header."""
    ln = re.sub(r"^#{1,6}\s*", "", ln.strip())
    ln = re.sub(r"[^\w\s]", "", ln).lower()
    return re.sub(r"\s+", " ", ln).strip()


def _strip_repeated_lines(page_texts: list[str], threshold: float = 0.15) -> list[str]:
    """Remove running headers/footers the LLM failed to omit: lines whose
    NORMALIZED form opens or closes many pages (book title on one parity,
    author on the other — each ~50% of half the pages, so the threshold is
    low). A real chapter heading appears once and is never touched.
    Pure page numbers are always stripped."""
    firsts, lasts = Counter(), Counter()
    for t in page_texts:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        for ln in lines[:2]:
            firsts[_norm_line(ln)] += 1
        for ln in lines[-2:]:
            lasts[_norm_line(ln)] += 1
    n = max(1, len(page_texts))
    repeated = ({ln for ln, c in firsts.items() if ln and c / n > threshold}
                | {ln for ln, c in lasts.items() if ln and c / n > threshold})

    def is_junk(ln: str) -> bool:
        s = ln.strip()
        return (not s or _norm_line(s) in repeated
                or re.fullmatch(r"\d{1,4}", s) is not None)

    out = []
    for t in page_texts:
        lines = t.splitlines()
        # only trim the page edges: up to 3 junk lines from each end
        for _ in range(3):
            while lines and not lines[0].strip():
                lines.pop(0)
            if lines and is_junk(lines[0]):
                lines.pop(0)
            else:
                break
        for _ in range(3):
            while lines and not lines[-1].strip():
                lines.pop()
            if lines and is_junk(lines[-1]):
                lines.pop()
            else:
                break
        out.append("\n".join(lines).strip())
    return out


def _join_pages(pages: list[str]) -> str:
    """Concatenate page texts, healing hyphenation and mid-sentence page breaks."""
    book = ""
    for text in pages:
        text = text.strip()
        if not text:
            continue
        if not book:
            book = text
            continue
        first_word = text.split(None, 1)[0] if text.split() else ""
        starts_lower = bool(first_word) and first_word[0].islower()
        if book.endswith("-") and starts_lower:
            # hyphenated word split across the page boundary
            book = book.rstrip("-") + text
        elif (starts_lower and not _SENTENCE_END.search(book.splitlines()[-1])
              and not _HEADING.match(text)):
            # paragraph continues across the page break
            book = book + " " + text
        else:
            book = book + "\n\n" + text
    return book


def run(ws: Workspace, cfg: dict, log=print) -> None:
    texts, missing = [], []
    for page in ws.manifest["pages"]:
        if page.get("role") == "cover" or page.get("status") in ("duplicate", "deleted"):
            continue  # covers are images; duplicates merged; deleted excluded
        if page.get("md"):
            texts.append((ws.root / page["md"]).read_text(encoding="utf-8"))
        else:
            missing.append(page["id"])
            texts.append("")  # keep a page boundary; content is simply absent
    if missing:
        log(f"  WARNING: {len(missing)} pages have no transcription: {', '.join(missing)}")

    texts = _strip_repeated_lines(texts)
    # unresolved figure placeholders (region never cropped) must not reach
    # the book — the review/reshoot flow surfaces them instead
    texts = [re.sub(r"\[\[region-\d+\]\]\n?", "", t) for t in texts]
    book = _join_pages(texts)
    out = ws.work_file("book.md")
    out.write_text(book, encoding="utf-8")
    headings = sum(1 for ln in book.splitlines() if ln.startswith("# "))
    log(f"  assembled {len(texts)} pages -> {out} ({headings} chapter headings)")
    ws.stage_done("assemble", missing=missing)
