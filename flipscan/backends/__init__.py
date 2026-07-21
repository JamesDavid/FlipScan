"""Transcription backends: shared prompt, JSON validation, backend selection."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

PROMPT = """\
You are transcribing a photographed page of a printed book. The photo may show an
open two-page spread — typically ONE page lies flat and readable while the other
is curved mid-turn — and there may be desk clutter around the book (sticky notes,
papers, other objects).

Transcribe ONLY the flat, clearly readable page. Completely ignore the curved or
foreshortened page and everything that is not part of the book.

Return ONLY a JSON object, no code fences, no commentary, matching exactly this schema:

{
  "markdown": "the page text as clean markdown",
  "page_number_printed": 143,
  "confidence": "high",
  "regions": [
    {"type": "figure", "bbox_norm": [0.1, 0.2, 0.9, 0.5], "caption": "optional caption"}
  ],
  "flags": []
}

Rules:
- "markdown": transcribe the body text faithfully. Use # / ## for chapter/section headings
  that appear on the page. OMIT running headers, running footers, and the printed page
  number from the markdown. Keep paragraph breaks. If a word is hyphenated across the
  page boundary, keep the trailing hyphen.
- Simple, cleanly readable tables -> markdown tables inline. Complex tables -> add a
  region with type "table_as_image" and put a placeholder line [[region-N]] in the markdown.
- For each figure, photo, or complex table on the page: add a region with a normalized
  bbox [x0, y0, x1, y1] (0-1, relative to image width/height) and put the placeholder
  [[region-N]] (N = index into regions, starting at 0) where it belongs in the markdown.
- "page_number_printed": the page number printed on the page you transcribed,
  or null if none is visible.
- "confidence": "high" | "medium" | "low" — your overall transcription confidence.
- "flags": any of "cut_off_text", "blur", "multi_column", "handwriting" that apply, else [].
"""

ESCALATION_FLAGS = {"cut_off_text", "blur", "multi_column", "handwriting"}


class TranscriptionError(Exception):
    pass


def parse_result(raw: str) -> dict[str, Any]:
    """Parse + validate the model's JSON. Raises TranscriptionError on garbage."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise TranscriptionError(f"no JSON object in response: {raw[:200]!r}")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise TranscriptionError(f"JSON parse failed: {e}") from e

    if not isinstance(obj, dict) or not isinstance(obj.get("markdown"), str):
        raise TranscriptionError("missing/invalid 'markdown'")
    if obj.get("confidence") not in ("high", "medium", "low"):
        obj["confidence"] = "low"
    pn = obj.get("page_number_printed")
    obj["page_number_printed"] = int(pn) if isinstance(pn, (int, float)) else None

    regions = []
    for r in obj.get("regions") or []:
        if not isinstance(r, dict):
            continue
        bbox = r.get("bbox_norm")
        if (isinstance(bbox, list) and len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)):
            vals = [max(0.0, min(1.0, float(v))) for v in bbox]
            x0, x1 = sorted((vals[0], vals[2]))  # models sometimes emit
            y0, y1 = sorted((vals[1], vals[3]))  # inverted corners
            regions.append({
                "type": r.get("type", "figure"),
                "bbox_norm": [x0, y0, x1, y1],
                "caption": r.get("caption") or "",
            })
    obj["regions"] = regions
    obj["flags"] = [f for f in (obj.get("flags") or []) if isinstance(f, str)]
    return obj


ORIENTATION_PROMPT = """\
Look at the printed text in this photo of a book. Is the text upside down
(rotated 180 degrees)? Return ONLY a JSON object: {"upside_down": true} or
{"upside_down": false}."""


class TranscriptionBackend(ABC):
    """Transcribe page images. Results are validated dicts keyed by page id;
    a failure is recorded as {"error": "..."} instead of raising."""

    name = "base"

    @abstractmethod
    def transcribe(self, pages: list[tuple[str, Path]],
                   log: Callable[[str], None] = print) -> dict[str, dict]:
        ...

    def check_orientation(self, image_path: Path) -> bool | None:
        """True if the image's text is upside down, None if this backend
        can't tell (callers then assume normal orientation)."""
        return None


def parse_orientation(raw: str) -> bool | None:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        v = json.loads(text[start:end + 1]).get("upside_down")
        return v if isinstance(v, bool) else None
    except json.JSONDecodeError:
        return None


def needs_escalation(result: dict, escalate_on: list[str]) -> bool:
    if "error" in result:
        return "malformed_json" in escalate_on
    if "low_confidence" in escalate_on and result["confidence"] == "low":
        return True
    if "flags" in escalate_on and set(result["flags"]) & ESCALATION_FLAGS:
        return True
    return False


def anthropic_enabled(cfg: dict) -> bool:
    """The master switch: the key can stay saved while all Anthropic API
    calls are turned off in settings."""
    return bool(cfg["provider"].get("anthropic_enabled", True))


def get_backend(cfg: dict) -> TranscriptionBackend:
    name = cfg["provider"]["name"]
    if name == "ollama":
        from .ollama import OllamaBackend
        return OllamaBackend(cfg)
    if name == "anthropic":
        if not anthropic_enabled(cfg):
            raise RuntimeError("Anthropic API is disabled in settings — "
                               "enable it or switch the provider to ollama")
        from .anthropic_backend import AnthropicBackend
        return AnthropicBackend(cfg)
    if name == "mock":
        from .mock import MockBackend
        return MockBackend(cfg)
    raise ValueError(f"unknown provider {name!r} (hybrid is handled by the transcribe stage)")
