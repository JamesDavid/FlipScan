"""Text post-processing shared by the transcribe stage and the on-demand
'reflow' action.

Vision models sometimes transcribe a page column line-for-line, preserving the
printed line wrapping (a newline at every physical line, words split by wrap
hyphens like "Archae-\nology"). We want flowing prose that breaks only at real
paragraph boundaries — the way the model does it on a good page.

`reflow_wrapped` turns hard-wrapped prose back into flowing paragraphs. It is a
deliberate no-op on text that is already one-line-per-paragraph, and it leaves
headings, lists, block quotes, tables, code fences, region placeholders and
blank-line paragraph breaks exactly as they are — so it can run on every page
without harming the well-behaved ones.
"""

from __future__ import annotations

import re

_BULLET = re.compile(r"[-*+]\s")
_ORDERED = re.compile(r"\d+[.)]\s")
_HR = re.compile(r"(-{3,}|\*{3,}|_{3,})\s*$")
_FENCE = re.compile(r"(```|~~~)")


def _is_block_line(s: str) -> bool:
    """A line that is its own block and must never be merged with a neighbour:
    headings, block quotes, table rows, horizontal rules, region placeholders,
    and displayed math."""
    if s.startswith(("#", ">", "|")):
        return True
    if s.startswith("[[region"):
        return True
    if s.startswith("$$") or s.endswith("$$"):
        return True
    if _HR.match(s):
        return True
    return False


def _starts_paragraph(s: str) -> bool:
    """A line that begins a paragraph/list item but whose following wrapped
    lines should still flow INTO it (bullet items, numbered notes)."""
    return bool(_BULLET.match(s) or _ORDERED.match(s))


def _join(acc: str, nxt: str) -> str:
    a, b = acc.rstrip(), nxt.lstrip()
    if not b:
        return a
    if not a:
        return b
    # a word split by a line-wrap hyphen ("Archae-" + "ology" -> "Archaeology").
    # Drop the hyphen only when the next fragment continues in lowercase; keep it
    # (with no space) for real hyphenated compounds and ranges that happen to
    # wrap ("Potassium-" + "Argon", "186-" + "190").
    if re.search(r"\w-$", a):
        return (a[:-1] + b) if b[:1].islower() else (a + b)
    return a + " " + b


def reflow_wrapped(md: str) -> str:
    """Reflow hard-wrapped prose into paragraphs that break only at real
    paragraph boundaries. Safe/no-op on already-flowing text and on structured
    markdown (headings, lists, tables, code, placeholders)."""
    if not md or "\n" not in md:
        return md
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    buf: str | None = None
    in_fence = False

    def flush() -> None:
        nonlocal buf
        if buf is not None:
            out.append(buf)
            buf = None

    for raw in lines:
        line = raw.rstrip()
        s = line.strip()
        if _FENCE.search(s):            # code fence delimiter: copy verbatim
            flush()
            out.append(line)
            in_fence = not in_fence
            continue
        if in_fence:                    # inside a code block: never reflow
            out.append(line)
            continue
        if s == "":                     # paragraph break
            flush()
            out.append("")
            continue
        if _is_block_line(s):           # its own block — keep as-is
            flush()
            out.append(line)
            continue
        if _starts_paragraph(s):        # new list item / note — flush, start fresh
            flush()
            buf = line
            continue
        # plain prose line: begin or extend the current flowing paragraph
        buf = line if buf is None else _join(buf, line)
    flush()
    return "\n".join(out)
