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
- A replacement must be a MINIMAL edit of the quote: same content, same
  images, same footnote markers, one small correction. Never delete words,
  names, or ![image](...) tags; never invent text that is not clearly implied.
- Leave proper names alone unless the SAME name is spelled differently
  elsewhere in this chapter (then match the majority spelling). Do not
  "correct" names against your outside knowledge.
- Leave footnote/endnote markers exactly as printed (superscripts, <sup>,
  bracketed numbers).
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
        return _parse_findings(_anthropic_review(cfg, chapter_md))
    # ollama / hybrid — local is plenty for an edit list, but greedy decoding
    # can loop until the budget truncates the JSON: salvage what parsed, and
    # retry once with anti-repetition sampling before giving up
    raw = _ollama_review(cfg, chapter_md)
    try:
        return _parse_findings(raw)
    except (ValueError, json.JSONDecodeError):
        found = _salvage_findings(raw)
        if found:
            return found
    raw = _ollama_review(cfg, chapter_md,
                         extra={"repeat_penalty": 1.15, "repeat_last_n": 256,
                                "num_predict": 8192})
    try:
        return _parse_findings(raw)
    except (ValueError, json.JSONDecodeError):
        return _salvage_findings(raw)  # possibly [] — a clean 'no findings'
        #                               beats failing the whole chapter


def _salvage_findings(raw: str) -> list[dict]:
    """Pull whatever complete finding objects exist out of a truncated reply
    (findings are flat objects, so brace matching is trivial)."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("quote"):
            out.append(_norm_finding(obj))
    return out[:MAX_FINDINGS]


def _ollama_review(cfg: dict, chapter_md: str, extra: dict | None = None) -> str:
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
                        "num_ctx": int(p.get("proof_num_ctx", 32768)),
                        **(extra or {})},
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
    out = [_norm_finding(f) for f in obj.get("findings", [])
           if isinstance(f, dict) and f.get("quote")]
    return out[:MAX_FINDINGS]


def _norm_finding(f: dict) -> dict:
    return {
        "type": str(f.get("type") or "other"),
        "severity": str(f.get("severity") or "low"),
        "quote": str(f["quote"])[:200],
        "replacement": (str(f["replacement"])
                        if f.get("replacement") is not None else None),
        "note": str(f.get("note") or "")[:300],
        "source": "llm",
    }


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


_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def edit_is_destructive(q: str, r: str) -> str | None:
    """LLM 'fixes' that delete content or rewrite too much are the dangerous
    kind (observed: dropped image tags with invented continuations, deleted
    clauses/names, footnote markers replaced with garbage). Returns a human
    reason, or None when the edit looks like a genuine small correction."""
    from difflib import SequenceMatcher
    if len(_IMG_MD.findall(r)) < len(_IMG_MD.findall(q)):
        return "the fix would delete a figure reference"
    if len(q) > 40 and len(r) < 0.6 * len(q):
        return "the fix deletes most of the quoted text"
    digits_q = re.findall(r"\d+", q)
    digits_r = re.findall(r"\d+", r)
    marker = any(s in q or s in r
                 for s in ("<sup", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸",
                           "⁹", "⁰", "ⁱ"))
    if digits_q != digits_r and marker:
        return "the fix rewrites footnote/reference numbers"
    if SequenceMatcher(None, q, r).ratio() < 0.55 and len(q) > 20:
        return "the fix rewrites rather than corrects"
    return None


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
        if (f.get("reference") and not f.get("user_edit")
                and not f.get("apply_all")):
            # reference matter (index page numbers, bibliography names):
            # the model can't verify these — never auto-apply
            f["skip_reason"] = "reference"
            continue
        if not f.get("user_edit") and not f.get("apply_all"):
            danger = edit_is_destructive(q, r)
            if danger:
                f["skip_reason"] = f"unsafe:{danger}"
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


REFERENCE_TITLES = {"index", "notes", "bibliography", "contents"}


def name_consistency_findings(ws: Workspace, chapter_md: str) -> list[dict]:
    """Deterministic proper-name check: a spelling that appears once or twice
    while a near-identical one appears many times across the WHOLE book is a
    misread (Mater vs Mather). Majority vote inside the book only — outside
    knowledge is never consulted (that's how 'Cross' becomes 'Croix')."""
    from collections import Counter
    from difflib import SequenceMatcher

    book_file = ws.work_file("book.md")
    text = (book_file.read_text(encoding="utf-8") if book_file.exists()
            else chapter_md)
    words = re.findall(r"\b[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ]{3,}\b", text)
    cnt = Counter(words)
    out = []
    for minority, mc in cnt.items():
        if mc > 2 or minority not in chapter_md:
            continue
        if text.count(minority.lower()) > 2:
            continue    # a common word capitalized at sentence start
        if re.search(rf"{re.escape(minority)}[a-z]", chapter_md):
            continue    # substring of longer words — plain replace unsafe
        best, best_n = None, 0
        for major, jc in cnt.items():
            if (jc >= 4 and jc > best_n and major[0] == minority[0]
                    and abs(len(major) - len(minority)) <= 2
                    and major not in (minority + "s", minority + "es")
                    and minority not in (major + "s", major + "es")
                    and SequenceMatcher(None, minority, major).ratio() >= 0.84):
                best, best_n = major, jc
        if best:
            out.append({
                "type": "spelling", "severity": "medium", "quote": minority,
                "replacement": best, "apply_all": True, "source": "lint",
                "note": f"'{best}' appears {best_n}× in this book, this "
                        f"variant {mc}× — majority spelling wins",
            })
    return out[:10]


def proofread_chapter(ws: Workspace, cfg: dict, idx: int) -> dict:
    """Run lint + LLM on one chapter; store findings and the proofed copy."""
    chs = chapters(ws)
    if not 0 <= idx < len(chs):
        raise IndexError(f"no chapter {idx}")
    title, md = chs[idx]
    reference = title.strip().lower() in REFERENCE_TITLES
    findings = lint_chapter(md)
    if not reference:
        findings += name_consistency_findings(ws, md)
    llm = llm_findings(cfg, md)
    if reference:
        for f in llm:
            f["reference"] = True   # suggestions only, never auto-applied
    findings += llm
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
                   apply_all: bool = False, replacement: str | None = None,
                   set_replacement: bool = False) -> dict:
    """Reject or re-enable one finding's fix; the proofed copy is rebuilt
    from the base chapter + all still-enabled edits. apply_all promotes an
    ambiguous fix (quote occurs N times) to replace every occurrence.
    set_replacement stores a user-authored fix on the finding — the way to
    resolve a note-only finding without ever touching the page OCR text
    (an empty replacement deletes the quoted text)."""
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
    if set_replacement:
        d["findings"][fi]["replacement"] = replacement or ""
        d["findings"][fi]["user_edit"] = True
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


REREAD_PROMPT = """This is a photograph of one book page. An OCR pass produced \
this garbled passage from it:

"{quote}"

Find that spot on the page and read it again carefully. Respond with JSON:
{{"replacement": "the exact printed text for that span, corrected"}}
Rules: transcribe ONLY what is printed — same span, same length, no additions.
If you cannot locate or read the passage, respond {{"replacement": null}}."""


def reread_from_image(cfg: dict, image_path: Path, quote: str) -> str | None:
    """Ask a vision model to re-read one garbled passage straight from the
    page image. Anthropic when configured, else the local Ollama model."""
    import base64
    p = cfg["provider"]
    prompt = REREAD_PROMPT.format(quote=quote[:300])
    use_anthropic = (p["name"] == "anthropic"
                     or (p.get("anthropic_api_key") and p["name"] == "hybrid"))
    if use_anthropic:
        import anthropic
        client = anthropic.Anthropic(api_key=p.get("anthropic_api_key") or None)
        msg = client.messages.create(
            model=p["anthropic_model"], max_tokens=800,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(
                        image_path.read_bytes()).decode()}},
                {"type": "text", "text": prompt}]}])
        raw = "".join(b.text for b in msg.content if b.type == "text")
    else:
        import httpx
        resp = httpx.post(
            f"{p['ollama_url'].rstrip('/')}/api/chat",
            json={"model": p["ollama_model"], "stream": False, "format": "json",
                  "think": p.get("ollama_think", False),
                  "options": {"num_predict": 800, "temperature": 0},
                  "messages": [{"role": "user", "content": prompt,
                                "images": [base64.standard_b64encode(
                                    image_path.read_bytes()).decode()]}]},
            timeout=600.0)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        r = json.loads(raw[start:end + 1]).get("replacement")
    except json.JSONDecodeError:
        return None
    return str(r).strip() if r else None


def resolve_finding(ws: Workspace, cfg: dict, idx: int, fi: int) -> dict:
    """Re-read one finding's passage from its source page image and store the
    result as this finding's suggested replacement (still guard-checked and
    user-reviewable — the page OCR text is untouched)."""
    d = load_proof(ws, idx)
    if d is None:
        raise FileNotFoundError("chapter not proofread yet")
    _title, md = chapters(ws)[idx]
    if chapter_hash(md) != d.get("base_hash"):
        raise ValueError("chapter text changed since this proof — re-run it")
    f = d["findings"][fi]
    page = ws.page(f.get("page") or "")
    if page is None or not page.get("llm_image"):
        raise LookupError("no source page image for this finding")
    corrected = reread_from_image(cfg, ws.root / page["llm_image"], f["quote"])
    if not corrected or _squashed(corrected) == _squashed(f["quote"]):
        raise LookupError("the page re-read matched the OCR — fix it with ✎ "
                          "or re-photograph the page")
    f["replacement"] = corrected
    f["note"] = (f.get("note", "") + " [re-read from the page image]").strip()
    f.pop("rejected", None)
    for x in d["findings"]:
        x["applied"] = False
        x.pop("skip_reason", None)
    proofed, applied = apply_edits(
        md, [x for x in d["findings"] if not x.get("rejected")])
    d["proofed_md"], d["applied"] = proofed, applied
    save_proof(ws, idx, d)
    return d


def _squashed(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


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
