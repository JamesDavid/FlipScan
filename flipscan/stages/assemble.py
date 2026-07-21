"""Stage 9: assemble — stitch per-page markdown into one book document.

Heals hyphenated words split across pages, merges paragraphs continuing
across page breaks, and strips residual running headers/footers.
Writes work/book.md (the input for all build formats).
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from ..workspace import Workspace

_SENTENCE_END = re.compile(r'[.!?:;"”’)\]]\s*$')
_HEADING = re.compile(r"^#{1,6}\s")
_TOC_ENTRY = re.compile(r"^(.{2,60}?)[\s.·]+(\d{1,4})\s*$")

# words that legitimately follow a suspended hyphen ("nineteenth- and
# twentieth-century") — never glue these onto the previous fragment
_HYPHEN_STOP = {"and", "or", "to", "the", "a", "an", "in", "of", "on", "at",
                "by", "for", "with", "but", "nor", "not", "so", "yet"}


def heal_hyphenation(text: str) -> str:
    """Remove print-line-break hyphenation: reflowable output must not keep
    'wing- less' or 'poten-\\ntial'. Real compounds ('well-known') have no
    space after the hyphen and are left alone."""
    text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)

    def join(m):
        if m.group(2).lower() in _HYPHEN_STOP:
            return m.group(0)
        return m.group(1) + m.group(2)

    return re.sub(r"\b([A-Za-z]{2,})- ([a-z]{2,})\b", join, text)


def _norm_line(ln: str) -> str:
    """Normalize for running-header comparison: drop markdown heading markers,
    case, punctuation, and spacing — '## DIRIGIBLE DREAMS' and 'Dirigible
    Dreams.' are the same running header."""
    ln = re.sub(r"^#{1,6}\s*", "", ln.strip())
    ln = re.sub(r"[^\w\s]", "", ln).lower()
    return re.sub(r"\s+", " ", ln).strip()


def _strip_repeated_lines(page_texts: list[str], threshold: float = 0.15,
                          extra_refs: set[str] | None = None) -> list[str]:
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
    # bad crops truncate the running header differently on every page
    # ('BLE DREAMS', 'GIB DREAMS'), so exact counting misses them — an edge
    # line is also junk when it fuzzy-matches a known header / title / author
    refs = {r for r in repeated | (extra_refs or set()) if len(r) >= 6}

    def fuzzy_header(s: str) -> bool:
        if not (3 <= len(s) <= 40):
            return False
        return any(s in r or SequenceMatcher(None, s, r).ratio() >= 0.65
                   for r in refs)

    def is_junk(ln: str) -> bool:
        s = ln.strip()
        if s.startswith("!["):   # figure images are never running headers
            return False
        # bare digit runs at page edges: page numbers, and printer's keys
        # like the copyright page's "5 4 3 2 1"
        if not s or re.fullmatch(r"[\d\s.·]{1,24}", s):
            return True
        norm = _norm_line(s)
        return norm in repeated or fuzzy_header(norm)

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


def parse_printed_toc(page_texts: list[str]) -> list[tuple[str, int]]:
    """Find the book's own contents page and read the chapter list from it:
    lines like 'Chapter 3 74' or 'Bibliography 247' under a CONTENTS heading.
    This is ground truth for the book's structure."""
    for t in page_texts:
        if not re.search(r"^#{0,6}\s*contents\s*$", t, re.M | re.I):
            continue
        entries = []
        for ln in t.splitlines():
            mt = _TOC_ENTRY.match(ln.strip())
            if not mt:
                continue
            title = re.sub(r"[.·\s]+$", "", mt.group(1)).strip()
            if title and not title.startswith(("#", "!", "|")):
                entries.append((title, int(mt.group(2))))
        # a real TOC lists several ascending page numbers
        nums = [n for _, n in entries]
        if len(entries) >= 3 and nums == sorted(nums):
            return entries
    return []


def _dedupe_headings(page_texts: list[str]) -> list[str]:
    """A level-1 heading repeated on later pages ('INDEX' atop every index
    page, a chapter name re-shown mid-chapter) is a section running header —
    keep the first occurrence, drop the rest."""
    seen: set[str] = set()
    out = []
    for t in page_texts:
        lines = []
        for ln in t.splitlines():
            if ln.startswith("# "):
                key = _norm_line(ln)
                if key in seen:
                    continue
                seen.add(key)
            lines.append(ln)
        out.append("\n".join(lines))
    return out


def _insert_chapter_breaks(pages: list[dict], texts: list[str],
                           toc: list[tuple[str, int]], log=print) -> list[str]:
    """The printed TOC says chapter N starts on printed page P. If that page
    exists but lost its opening heading (bad crop, missed transcription),
    prepend one so the built book keeps the chapter structure."""
    by_num = {}
    for i, p in enumerate(pages):
        n = p.get("printed_number")
        if isinstance(n, int) and n not in by_num:
            by_num[n] = i
    added = 0
    for title, start in toc:
        i = by_num.get(start)
        if i is None or not texts[i].strip():
            continue
        lines = texts[i].splitlines()
        first = next((j for j, ln in enumerate(lines) if ln.strip()), None)
        if first is not None and _HEADING.match(lines[first]):
            # the page opens with its own heading ('## ONE') — promote it to
            # level 1 so it splits a chapter, keeping the book's own title
            promoted = re.sub(r"^#{1,6}\s*", "# ", lines[first])
            if promoted != lines[first]:
                lines[first] = promoted
                texts[i] = "\n".join(lines)
                added += 1
            continue
        head = [ln for ln in lines if ln.strip()][:2]
        if any(_HEADING.match(ln) for ln in head):
            continue  # a heading close to the top — leave the page alone
        texts[i] = f"# {title}\n\n{texts[i]}"
        added += 1
    if added:
        log(f"  adjusted {added} chapter openings from the printed contents page")
    return texts


def _apply_manual_sections(pages: list[dict], texts: list[str]) -> list[str]:
    """A user-assigned section heading on a page outranks everything: that
    page opens a chapter with exactly that title."""
    for i, p in enumerate(pages):
        title = (p.get("section") or "").strip()
        if not title:
            continue
        lines = texts[i].splitlines()
        first = next((j for j, ln in enumerate(lines) if ln.strip()), None)
        if first is not None and _HEADING.match(lines[first]):
            lines[first] = f"# {title}"      # replace the page's own heading
            texts[i] = "\n".join(lines)
        elif texts[i].strip():
            texts[i] = f"# {title}\n\n{texts[i]}"
        else:
            texts[i] = f"# {title}"          # content still missing — keep
            #                                  the chapter break regardless
    return texts


def _with_gap_markers(pages: list[dict], texts: list[str]) -> list[str]:
    """Make missing pages visible in the built book itself: a jump in the
    printed numbers gets an inline ⟦ missing ⟧ notice at the exact spot.
    (In practice these get fixed, but a production copy must say so.)"""
    out, last = [], None
    for p, t in zip(pages, texts):
        n = p.get("printed_number")
        if isinstance(n, int) and n >= 1:
            if last is not None and n > last + 1:
                a, b = last + 1, n - 1
                rng = f"pages {a}–{b}" if b > a else f"page {a}"
                out.append(f"> ⟦ {rng} missing from this scan — "
                           f"not yet captured. ⟧")
            last = n
        out.append(t)
    return out


def _split_leading_figures(text: str) -> tuple[str | None, str]:
    """Peel a page-top image (and its italic caption) off the page text so a
    sentence continuing across the page break isn't interrupted by it."""
    lines = text.splitlines()
    figs, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("![") or (figs and s.startswith("*") and s.endswith("*")):
            figs.append(s)
            i += 1
            continue
        break
    if not figs:
        return None, text
    return "\n\n".join(figs), "\n".join(lines[i:]).strip()


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
        # a photo page often opens with its figure; if the sentence continues
        # from the previous page, the figure must not sit mid-sentence
        figs, body = _split_leading_figures(text)
        first_word = body.split(None, 1)[0] if body.split() else ""
        starts_lower = bool(first_word) and first_word[0].islower()
        last_line = book.splitlines()[-1]
        # a line with almost no letters (printer's key, stray digits) never
        # continues a sentence
        gluey = sum(c.isalpha() for c in last_line) >= 3
        if figs is not None and not starts_lower:
            text = f"{figs}\n\n{body}" if body else figs
            book = book + "\n\n" + text
        elif book.endswith("-") and starts_lower:
            # hyphenated word split across the page boundary
            book = book.rstrip("-") + (body if figs else text)
            if figs:
                book = book + "\n\n" + figs
        elif (starts_lower and gluey and not _SENTENCE_END.search(last_line)
              and not last_line.startswith("> ⟦")     # gap notices
              and not _HEADING.match(body or text)):  # stand alone
            # paragraph continues across the page break; a page-top figure
            # moves below the joined paragraph
            book = book + " " + (body if figs else text)
            if figs:
                book = book + "\n\n" + figs
        else:
            book = book + "\n\n" + text
    return book


def run(ws: Workspace, cfg: dict, log=print) -> None:
    texts, kept, missing = [], [], []
    for page in ws.manifest["pages"]:
        if page.get("role") == "cover" or page.get("status") in ("duplicate", "deleted"):
            continue  # covers are images; duplicates merged; deleted excluded
        kept.append(page)
        if page.get("md"):
            texts.append((ws.root / page["md"]).read_text(encoding="utf-8"))
        else:
            missing.append(page["id"])
            texts.append("")  # keep a page boundary; content is simply absent
    if missing:
        log(f"  WARNING: {len(missing)} pages have no transcription: {', '.join(missing)}")

    book_meta = ws.manifest["book"]
    extra_refs = {_norm_line(s) for s in (book_meta.get("title"),
                                          book_meta.get("author")) if s}
    toc = parse_printed_toc(texts)
    if toc:
        log(f"  printed contents page found: {len(toc)} entries")
    texts = _strip_repeated_lines(texts, extra_refs=extra_refs)
    texts = _dedupe_headings(texts)
    texts = _insert_chapter_breaks(kept, texts, toc, log)
    texts = _apply_manual_sections(kept, texts)  # user's word is final
    # unresolved figure placeholders (region never cropped) must not reach
    # the book — the review/reshoot flow surfaces them instead
    texts = [re.sub(r"\[\[region-\d+\]\]\n?", "", t) for t in texts]
    texts = [heal_hyphenation(t) for t in texts]
    book = _join_pages(_with_gap_markers(kept, texts))
    missing_nums = ws.manifest.get("missing_pages") or []
    if missing_nums:
        from .transcribe import format_ranges
        book = (f"> ⟦ PRODUCTION COPY — {len(missing_nums)} pages not yet "
                f"captured: {format_ranges(missing_nums)}. ⟧\n\n" + book)
    out = ws.work_file("book.md")
    out.write_text(book, encoding="utf-8")
    headings = sum(1 for ln in book.splitlines() if ln.startswith("# "))
    log(f"  assembled {len(texts)} pages -> {out} ({headings} chapter headings)")
    ws.stage_done("assemble", missing=missing)
