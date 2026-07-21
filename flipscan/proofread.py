"""Chapter-by-chapter proofread: a distinct, non-destructive layer.

The compiled book (work/book.md) is split into chapters; each chapter goes to
the LLM which returns FINDINGS (quote → replacement edit list), never a
rewritten chapter — so nothing can be silently paraphrased and every change
is inspectable. Safe edits (exact, unique quote match) are applied to produce
a proofed copy; the rest are notes for the user. Page OCR data is never
touched. The EPUB builder substitutes an accepted proofed chapter only while
its base text is unchanged (content hash).

Stored per chapter in work/proof/proof_NNN.json:
  {title, base_hash, findings[], proofed_md, status: review|accepted|rejected}
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .workspace import Workspace

MAX_FINDINGS = 40

PROOF_PROMPT = """You are proofreading one chapter of a book that was digitized \
with OCR from photographs. Find MECHANICAL problems only. Respond with JSON:

{"findings": [{"type": "spelling|ocr|formatting|continuity|other",
  "severity": "low|medium|high",
  "quote": "EXACT substring copied verbatim from the text, 10-120 characters",
  "replacement": "the corrected text, or null when you cannot safely fix it",
  "note": "one short sentence explaining the problem"}]}

Rules:
- NEVER paraphrase, restyle, or improve the prose. Only fix objective errors:
  OCR misreads, garbled characters (like �), broken hyphen- ation, doubled or
  missing spaces, obviously wrong words, stray page artifacts.
- "quote" must match the chapter text EXACTLY, character for character, and be
  unique enough to locate. Include surrounding words if needed.
- Suspected missing text, duplicated passages, contradictions, or sentences
  that do not follow: report them with "replacement": null.
- Lines like "> ⟦ pages 12–14 missing from this scan … ⟧" mark KNOWN gaps.
  Do not flag them, and expect discontinuities right there.
- Markdown is intentional: # headings, ![figure](images/...) images, > quotes.
  Flag markdown only when it is malformed.
- If the chapter looks clean, return {"findings": []}.

The chapter:
"""


def chapter_hash(md: str) -> str:
    return hashlib.sha1(md.strip().encode("utf-8")).hexdigest()[:16]


def proof_dir(ws: Workspace) -> Path:
    d = ws.work_file("proof")
    d.mkdir(exist_ok=True)
    return d


def proof_path(ws: Workspace, idx: int) -> Path:
    return proof_dir(ws) / f"proof_{idx:03d}.json"


def load_proof(ws: Workspace, idx: int) -> dict | None:
    f = proof_path(ws, idx)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_proof(ws: Workspace, idx: int, data: dict) -> None:
    proof_path(ws, idx).write_text(json.dumps(data, indent=1), encoding="utf-8")


def chapters(ws: Workspace) -> list[tuple[str, str]]:
    """The book's chapters as (title, markdown) from the assembled book.md."""
    from .build_epub import split_chapters
    book = ws.work_file("book.md")
    if not book.exists():
        raise FileNotFoundError("work/book.md missing — run the pipeline first")
    return split_chapters(book.read_text(encoding="utf-8"))


# ---------------- deterministic lint (free, runs before the LLM)

def lint_chapter(md: str) -> list[dict]:
    finds: list[dict] = []

    def add(type_, severity, quote, note):
        finds.append({"type": type_, "severity": severity, "quote": quote[:120],
                      "replacement": None, "note": note, "source": "lint"})

    for m in re.finditer(r"\[\[region-\d+\]\]", md):
        add("formatting", "high", m.group(0), "unresolved figure placeholder")
    # words containing the U+FFFD replacement character: encoding damage
    bad_words = sorted({w for w in re.findall(r"\S*�\S*", md)})
    for w in bad_words[:8]:
        add("ocr", "medium", w, "unreadable character — likely a lost accent")
    if len(bad_words) > 8:
        add("ocr", "medium", bad_words[8],
            f"...and {len(bad_words) - 8} more words with unreadable characters")
    for m in re.finditer(r"\b(\w{3,})\s+\1\b", md, re.I):  # "the the"
        add("ocr", "low", m.group(0), "doubled word")
    # consecutive duplicate paragraphs (same page captured twice)
    paras = [p.strip() for p in md.split("\n\n") if len(p.strip()) > 80]
    for a, b in zip(paras, paras[1:]):
        if a == b:
            add("continuity", "high", a[:120],
                "identical paragraph appears twice in a row — duplicated page?")
    return finds[:MAX_FINDINGS]


# ---------------- LLM pass

def llm_findings(cfg: dict, chapter_md: str) -> list[dict]:
    provider = cfg["provider"]["name"]
    if provider == "mock":
        return []
    if provider == "anthropic":
        raw = _anthropic_review(cfg, chapter_md)
    else:  # ollama / hybrid — local is plenty for an edit list
        raw = _ollama_review(cfg, chapter_md)
    return _parse_findings(raw)


def _ollama_review(cfg: dict, chapter_md: str) -> str:
    import httpx
    p = cfg["provider"]
    resp = httpx.post(
        f"{p['ollama_url'].rstrip('/')}/api/chat",
        json={
            "model": p["ollama_model"],
            "stream": False,
            "format": "json",
            "think": p.get("ollama_think", False),
            # long chapters need a big context window; findings stay small
            "options": {"num_predict": 4096, "temperature": 0,
                        "num_ctx": int(p.get("proof_num_ctx", 32768))},
            "messages": [{"role": "user", "content": PROOF_PROMPT + chapter_md}],
        },
        timeout=900.0,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _anthropic_review(cfg: dict, chapter_md: str) -> str:
    import anthropic
    p = cfg["provider"]
    client = anthropic.Anthropic(api_key=p.get("anthropic_api_key") or None)
    msg = client.messages.create(
        model=p["anthropic_model"],
        max_tokens=4096,
        messages=[{"role": "user", "content": PROOF_PROMPT + chapter_md}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def _parse_findings(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON in proofread response: {raw[:200]!r}")
    obj = json.loads(text[start:end + 1])
    out = []
    for f in obj.get("findings", []):
        if not isinstance(f, dict) or not f.get("quote"):
            continue
        out.append({
            "type": str(f.get("type") or "other"),
            "severity": str(f.get("severity") or "low"),
            "quote": str(f["quote"])[:200],
            "replacement": (str(f["replacement"])
                            if f.get("replacement") is not None else None),
            "note": str(f.get("note") or "")[:300],
            "source": "llm",
        })
    return out[:MAX_FINDINGS]


# ---------------- applying the edit list

def _quote_pattern(quote: str) -> re.Pattern:
    """Whitespace/typography-tolerant matcher: the model quotes text with
    normalized spacing and straight quotes, while the chapter has line
    breaks, curly quotes, and long dashes at the same spots."""
    parts = []
    for tok in quote.split():
        chars = []
        for ch in tok:
            if ch in "'’‘":
                chars.append("['’‘]")
            elif ch in '"“”':
                chars.append('["“”]')
            elif ch in "-–—":
                chars.append("[-–—]")
            else:
                chars.append(re.escape(ch))
        # OCR hyphenation: the text may split a word as 'dirigi- ble'
        parts.append(r"(?:[-–—]\s+)?".join(chars))
    return re.compile(r"\s+".join(parts))


def apply_edits(md: str, findings: list[dict]) -> tuple[str, int]:
    """Apply the safe edits: a quote matching exactly once (or one the user
    explicitly promoted to apply-to-all). Matching is exact first, then
    whitespace/typography-tolerant. Everything else records WHY it was
    skipped so the UI can offer the right action."""
    applied = 0
    for f in findings:
        f["applied"] = False
        f.pop("skip_reason", None)
        q, r = f.get("quote"), f.get("replacement")
        if not q or r is None or r == q:
            continue
        n = md.count(q)
        if n > 0:
            if n == 1 or f.get("apply_all"):
                md = md.replace(q, r)   # every occurrence when apply_all
                f["applied"] = True
                applied += 1
            else:
                f["skip_reason"] = f"ambiguous:{n}"
            continue
        pat = _quote_pattern(q)
        hits = list(pat.finditer(md))
        if not hits:
            f["skip_reason"] = "not_found"
        elif len(hits) == 1 or f.get("apply_all"):
            md = pat.sub(lambda _m: r, md)
            f["applied"] = True
            applied += 1
        else:
            f["skip_reason"] = f"ambiguous:{len(hits)}"
    return md, applied


def proofread_chapter(ws: Workspace, cfg: dict, idx: int) -> dict:
    """Run lint + LLM on one chapter; store findings and the proofed copy."""
    chs = chapters(ws)
    if not 0 <= idx < len(chs):
        raise IndexError(f"no chapter {idx}")
    title, md = chs[idx]
    findings = lint_chapter(md)
    findings += llm_findings(cfg, md)
    proofed, applied = apply_edits(md, findings)

    from .review import find_page_by_text
    for f in findings:  # anchor every finding to its source page if possible
        p = find_page_by_text(ws, f["quote"])
        f["page"] = p["id"] if p else None
        f["printed_number"] = p.get("printed_number") if p else None

    data = {
        "title": title,
        "base_hash": chapter_hash(md),
        "findings": findings,
        "applied": applied,
        "proofed_md": proofed,
        "status": "review",
        # a proof that changed a suspicious amount of text deserves scrutiny
        "heavy": abs(len(proofed) - len(md)) > max(200, len(md) * 0.05),
    }
    save_proof(ws, idx, data)
    return data


def toggle_finding(ws: Workspace, idx: int, fi: int, enabled: bool,
                   apply_all: bool = False) -> dict:
    """Reject or re-enable one finding's fix; the proofed copy is rebuilt
    from the base chapter + all still-enabled edits. apply_all promotes an
    ambiguous fix (quote occurs N times) to replace every occurrence."""
    d = load_proof(ws, idx)
    if d is None:
        raise FileNotFoundError("chapter not proofread yet")
    chs = chapters(ws)
    if not 0 <= idx < len(chs):
        raise IndexError(f"no chapter {idx}")
    _title, md = chs[idx]
    if chapter_hash(md) != d.get("base_hash"):
        raise ValueError("chapter text changed since this proof — re-run it")
    if not 0 <= fi < len(d["findings"]):
        raise IndexError(f"no finding {fi}")
    d["findings"][fi]["rejected"] = not enabled
    if apply_all:
        d["findings"][fi]["apply_all"] = True
    elif not enabled:
        d["findings"][fi].pop("apply_all", None)
    for f in d["findings"]:
        f["applied"] = False
        f.pop("skip_reason", None)
    proofed, applied = apply_edits(
        md, [f for f in d["findings"] if not f.get("rejected")])
    d["proofed_md"] = proofed
    d["applied"] = applied
    d["heavy"] = abs(len(proofed) - len(md)) > max(200, len(md) * 0.05)
    save_proof(ws, idx, d)
    return d


def proof_status(ws: Workspace) -> list[dict]:
    """One row per chapter for the UI: proof state vs the current text."""
    rows = []
    for i, (title, md) in enumerate(chapters(ws)):
        d = load_proof(ws, i)
        cur = chapter_hash(md)
        status = "unproofed"
        if d:
            status = d.get("status", "review")
            if d.get("base_hash") != cur:
                status = "stale"   # the chapter text changed since the proof
        rows.append({
            "idx": i, "title": title, "hash": cur, "status": status,
            "findings": len(d["findings"]) if d else None,
            "applied": d.get("applied") if d else None,
            "heavy": bool(d.get("heavy")) if d else False,
            "chars": len(md),
        })
    return rows
