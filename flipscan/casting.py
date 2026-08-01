"""Full-cast narration: find quoted speech, attribute it to characters, and
let each character be voiced from the shared voice library.

One LLM pass per chapter produces (a) a cast list with short descriptions and
(b) each quote verbatim with its speaker. We never let the model rewrite the
text — quotes are located in the narration text by tolerant string matching
(the same philosophy as the proofreader's edit application), and anything that
doesn't match, or whose speaker is uncertain, simply stays with the narrator.
The cast and the user's voice assignments live in work/audiobook/cast.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

from .workspace import Workspace

MAX_QUOTES_PER_CHAPTER = 400
SAMPLES_PER_CHARACTER = 3
ANALYZE_CHUNK_CHARS = 18000   # a dialogue-dense 100k+ chapter overflows one
#                               reply's token budget — analyze in windows


def _split_for_analysis(text: str, target: int = ANALYZE_CHUNK_CHARS) -> list[str]:
    """Paragraph-boundary windows of roughly `target` chars."""
    if len(text) <= target * 1.3:
        return [text]
    parts, buf = [], ""
    for p in text.split("\n\n"):
        if buf and len(buf) + len(p) > target:
            parts.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        parts.append(buf)
    return parts

CAST_PROMPT = """\
You are analyzing one chapter of a book for audiobook production. Find the
DIRECT quoted speech — words a person actually says or wrote that are quoted
in the text — and identify who says each one.

Return ONLY a JSON object, no code fences, exactly this shape:

{
  "characters": [
    {"name": "Hugo Eckener", "description": "airship commander; German, formal, measured",
     "sounds_like": "gravelly, deliberate elder-statesman baritone — similar to Anthony Hopkins in The Remains of the Day"}
  ],
  "quotes": [
    {"quote": "the exact quoted words, copied verbatim from the text", "speaker": "Hugo Eckener"}
  ]
}

Rules:
- "quote": the text BETWEEN the quotation marks, copied EXACTLY — same
  spelling, punctuation, capitalization. Do not paraphrase or fix anything.
- Only direct speech or quoted writing by a person. NOT titles, scare quotes,
  emphasized words, or quoted phrases shorter than 4 words.
- EPIGRAPHS count: a quotation at a chapter/section opening followed by the
  originator's name on its own line (or after a dash) is spoken by that person
  — attribute it to them, not NARRATOR.
- "speaker": the person who says/wrote it. Use a consistent canonical name for
  the same person throughout. If you are not confident who speaks, use
  "NARRATOR" (those stay in the narrator's voice).
- "characters": every speaker you used (except NARRATOR), with a description
  in under 15 words drawn from the text: who they are, and anything relevant
  to voice (age, gender, nationality, temperament) that the TEXT supports.
- "sounds_like": a casting tip in under 20 words. It MUST name a specific
  famous person — an actor (ideally with a specific role, e.g. "similar to
  Robert De Niro in Casino"), broadcaster, or public figure — plus a few words
  describing the voice quality. For real historical figures, use your general
  knowledge of how they actually sounded (or name the person themselves if
  their voice is famous). Never answer with only a generic description; always
  include a name. It's a hint for a human choosing voices, not a claim.
- No commentary, just the JSON.

CHAPTER:
"""


# ---------------- LLM call (mirrors the proofreader's provider handling)

def _call_llm(cfg: dict, prompt: str) -> str:
    from .backends import anthropic_enabled
    provider = cfg["provider"]["name"]
    if provider == "mock":
        return '{"characters": [], "quotes": []}'
    if provider == "anthropic" and anthropic_enabled(cfg):
        import anthropic
        p = cfg["provider"]
        client = anthropic.Anthropic(api_key=p.get("anthropic_api_key") or None)
        msg = client.messages.create(
            model=p["anthropic_model"], max_tokens=8192,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")
    from .proofread import _ollama_post
    p = cfg["provider"]
    return _ollama_post(cfg, {
        "model": p["ollama_model"], "stream": False, "format": "json",
        "think": p.get("ollama_think", False),
        "options": {"num_predict": 8192, "temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=600.0)


def _parse_cast(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON in cast response: {raw[:200]!r}")
    obj = json.loads(text[start:end + 1])
    chars = [{"name": str(c.get("name", "")).strip(),
              "description": str(c.get("description", "")).strip(),
              "sounds_like": str(c.get("sounds_like", "")).strip()}
             for c in obj.get("characters", []) if isinstance(c, dict)
             and str(c.get("name", "")).strip()]
    quotes = [{"q": str(q.get("quote", "")).strip(),
               "speaker": str(q.get("speaker", "")).strip() or "NARRATOR"}
              for q in obj.get("quotes", []) if isinstance(q, dict)
              and len(str(q.get("quote", "")).strip()) >= 8]
    return {"characters": chars, "quotes": quotes[:MAX_QUOTES_PER_CHAPTER]}


# ---------------- tolerant quote location

def _salvage_cast(raw: str) -> dict:
    """Broken/truncated JSON (a dense chapter can blow the token budget) still
    contains complete flat objects — harvest every {"quote"...} and
    {"name"...} that parses individually, instead of losing the chapter."""
    chars, quotes = [], []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("quote"):
            q = str(obj["quote"]).strip()
            if len(q) >= 8:
                quotes.append({"q": q,
                               "speaker": str(obj.get("speaker", "")).strip()
                               or "NARRATOR"})
        elif obj.get("name"):
            chars.append({"name": str(obj["name"]).strip(),
                          "description": str(obj.get("description", "")).strip(),
                          "sounds_like": str(obj.get("sounds_like", "")).strip()})
    return {"characters": chars, "quotes": quotes[:MAX_QUOTES_PER_CHAPTER]}


def _normalize(s: str) -> tuple[str, list[int]]:
    """Lowercased, typography-straightened, whitespace-collapsed copy of `s`
    plus a map from each normalized char back to its original index."""
    out, idx = [], []
    prev_space = False
    trans = {"‘": "'", "’": "'", "“": '"', "”": '"',
             "–": "-", "—": "-", "…": "..."}
    for i, ch in enumerate(s):
        ch = trans.get(ch, ch)
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
        else:
            for c in ch.lower():
                out.append(c)
                idx.append(i)
            prev_space = False
    return "".join(out), idx


def locate_quotes(text: str, quotes: list[dict]) -> list[dict]:
    """Attach (start, end) spans to each quote, searching FORWARD through the
    chapter (quotes arrive in reading order, so repeated lines resolve to
    successive occurrences). Unmatched quotes are dropped — they stay with the
    narrator by construction."""
    norm, idx = _normalize(text)
    out, cursor = [], 0
    for q in quotes:
        nq, _ = _normalize(q["q"])
        nq = nq.strip()
        if len(nq) < 8:
            continue
        at = norm.find(nq, cursor)
        if at == -1:
            at = norm.find(nq)          # out-of-order fallback: first anywhere
        if at == -1:
            continue
        start = idx[at]
        end = idx[at + len(nq) - 1] + 1
        out.append({**q, "start": start, "end": end})
        cursor = at + len(nq)
    return out


def segments_for(text: str, located: list[dict],
                 voice_of: dict[str, str]) -> list[tuple[str, str]]:
    """Split a chapter into [(voice_name, segment_text)] runs. '' = narrator.
    Only quotes whose speaker has an assigned voice switch; attribution tags
    and everything else stay with the narrator. Overlapping/nested spans keep
    the first."""
    spans = []
    last_end = 0
    for q in sorted(located, key=lambda q: q["start"]):
        v = voice_of.get(q["speaker"], "")
        if not v or q["start"] < last_end:
            continue
        spans.append((q["start"], q["end"], v))
        last_end = q["end"]
    segs, pos = [], 0
    for start, end, v in spans:
        if start > pos:
            segs.append(("", text[pos:start]))
        segs.append((v, text[start:end]))
        pos = end
    if pos < len(text):
        segs.append(("", text[pos:]))
    return [(v, t) for v, t in segs if t.strip()]


# ---------------- cast store

def _cast_path(ws: Workspace) -> Path:
    return ws.work_file("audiobook") / "cast.json"


def load_cast(ws: Workspace) -> dict | None:
    f = _cast_path(ws)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_cast(ws: Workspace, cast: dict) -> None:
    f = _cast_path(ws)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cast, indent=1, ensure_ascii=False),
                 encoding="utf-8")


def assign_voice(ws: Workspace, character: str, voice: str) -> dict:
    cast = load_cast(ws)
    if cast is None:
        raise FileNotFoundError("no cast analysis yet — run Analyze characters")
    ch = cast["characters"].get(character)
    if ch is None:
        raise KeyError(f"unknown character {character!r}")
    ch["voice"] = voice
    save_cast(ws, cast)
    return cast


def set_voice_prompt(ws: Workspace, character: str, prompt: str) -> dict:
    """Store a user-edited voice-generator description for a character — used
    VERBATIM by 🪄 instead of the auto-composed one ('' reverts to auto)."""
    cast = load_cast(ws)
    if cast is None:
        raise FileNotFoundError("no cast analysis yet — run Analyze characters")
    ch = cast["characters"].get(character)
    if ch is None:
        raise KeyError(f"unknown character {character!r}")
    if prompt.strip():
        ch["voice_prompt"] = prompt.strip()
    else:
        ch.pop("voice_prompt", None)
    save_cast(ws, cast)
    return cast


# ---------------- the analysis job

def _aggregate(chapters: list[dict], prev_chars: dict[str, dict]) -> dict:
    """Book-wide character map rebuilt from the per-chapter results; voice
    assignments and user-edited voice prompts carried over by name."""
    def fresh(name: str) -> dict:
        old = prev_chars.get(name) or {}
        entry = {"description": "", "sounds_like": "", "quotes": 0,
                 "samples": [], "voice": old.get("voice", "")}
        if old.get("voice_prompt"):
            entry["voice_prompt"] = old["voice_prompt"]
        return entry

    characters: dict[str, dict] = {}
    for row in chapters:
        for c in row.get("characters") or []:
            entry = characters.setdefault(c["name"], fresh(c["name"]))
            if not entry["description"] and c.get("description"):
                entry["description"] = c["description"]
            if not entry["sounds_like"] and c.get("sounds_like"):
                entry["sounds_like"] = c["sounds_like"]
        for q in row.get("quotes") or []:
            if q["speaker"] == "NARRATOR":
                continue
            entry = characters.setdefault(q["speaker"], fresh(q["speaker"]))
            entry["quotes"] += 1
            if len(entry["samples"]) < SAMPLES_PER_CHARACTER:
                entry["samples"].append(q["q"][:160])
    # characters the model listed but never actually quoted add noise — drop
    return {n: c for n, c in characters.items() if c["quotes"] > 0}


def analyze_book(ws: Workspace, cfg: dict,
                 log: Callable[[str], None] = print,
                 should_cancel: Callable[[], bool] = lambda: False,
                 call_llm: Callable[[dict, str], str] | None = None,
                 only_failed: bool = False,
                 only_chapters: list[int] | None = None) -> dict:
    """One LLM pass per chapter; aggregates a book-wide cast. Existing voice
    assignments survive re-analysis (matched by character name). A chapter
    whose analysis fails is recorded with its error (and stays narrator-only)
    so the UI can offer a retry; only_failed re-analyzes just those chapters,
    only_chapters re-analyzes just the given indices — both keep every other
    chapter's existing result."""
    from .audiobook import narration_chapters
    call = call_llm or _call_llm
    prev = load_cast(ws) or {}
    prev_chars = prev.get("characters") or {}
    prev_rows = {r.get("title"): r for r in (prev.get("chapters") or [])}

    chapters = []
    chs = narration_chapters(ws)
    for i, (title, text) in enumerate(chs):
        if should_cancel():
            from .jobs import JobCanceled
            raise JobCanceled()
        old = prev_rows.get(title)
        skip = (only_failed and old is not None and not old.get("error")) or \
               (only_chapters is not None and i not in only_chapters)
        if skip and old is not None:
            chapters.append(old)      # keep the existing result untouched
            continue
        if skip:
            chapters.append({"title": title, "quotes": [], "characters": []})
            continue
        parts = _split_for_analysis(text)
        log(f"  [{i + 1}/{len(chs)}] {title[:46]!r}: analyzing"
            + (f" in {len(parts)} parts" if len(parts) > 1 else "") + "…")
        all_chars: list[dict] = []
        all_quotes: list[dict] = []
        errs: list[str] = []
        for part in parts:
            if should_cancel():
                from .jobs import JobCanceled
                raise JobCanceled()
            raw = None
            try:
                raw = call(cfg, CAST_PROMPT + part)
                parsed = _parse_cast(raw)
            except Exception as e:
                # a truncated/broken reply still holds complete objects —
                # harvest them rather than losing the window
                parsed = _salvage_cast(raw) if raw else {"characters": [],
                                                         "quotes": []}
                if parsed["quotes"] or parsed["characters"]:
                    log(f"    a window's reply was malformed — salvaged "
                        f"{len(parsed['quotes'])} quotes from it")
                else:
                    errs.append(str(e)[:120])
                    continue
            all_chars += parsed["characters"]
            all_quotes += parsed["quotes"]
        if errs and not all_quotes and not all_chars:
            err = errs[0]
            log(f"    analysis failed ({err[:80]}) — chapter stays "
                f"narrator-only")
            chapters.append({"title": title, "quotes": [], "characters": [],
                             "error": err})
            continue
        if errs:
            log(f"    note: {len(errs)}/{len(parts)} window(s) yielded "
                f"nothing usable")
        parsed = {"characters": all_chars,
                  "quotes": all_quotes[:MAX_QUOTES_PER_CHAPTER]}
        located = locate_quotes(text, parsed["quotes"])
        real = [q for q in located if q["speaker"] != "NARRATOR"]
        log(f"    {len(parsed['quotes'])} quotes reported, "
            f"{len(located)} matched in text, {len(real)} attributed")
        chapters.append({"title": title,
                         "quotes": [{"q": q["q"], "speaker": q["speaker"]}
                                    for q in located],
                         "characters": parsed["characters"]})
    characters = _aggregate(chapters, prev_chars)
    cast = {"analyzed_at": time.time(), "chapters": chapters,
            "characters": characters}
    save_cast(ws, cast)
    failed = [r["title"] for r in chapters if r.get("error")]
    log(f"  cast: {len(characters)} speaking character(s), "
        f"{sum(c['quotes'] for c in characters.values())} attributed quotes"
        + (f" — {len(failed)} chapter(s) FAILED: {', '.join(failed[:4])}"
           if failed else ""))
    return cast
