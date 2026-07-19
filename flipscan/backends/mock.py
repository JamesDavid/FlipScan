"""Mock backend: deterministic placeholder output for offline pipeline testing.

No OCR happens — this exists so extract->...->assemble->build can be exercised
without an Ollama server or API key (`flipscan run DIR --provider mock`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import TranscriptionBackend


class MockBackend(TranscriptionBackend):
    name = "mock"

    def __init__(self, cfg: dict):
        pass

    def transcribe(self, pages: list[tuple[str, Path]],
                   log: Callable[[str], None] = print) -> dict[str, dict]:
        results = {}
        for i, (page_id, image_path) in enumerate(pages):
            heading = f"# Chapter {i // 4 + 1}\n\n" if i % 4 == 0 else ""
            results[page_id] = {
                "markdown": (
                    f"{heading}This is mock transcription text for {page_id}. "
                    f"It stands in for real page content so the downstream "
                    f"assemble and build stages can be tested offline.\n\n"
                    f"A second paragraph continues the mock content of {page_id}."
                ),
                "page_number_printed": None,  # a real backend reads it off the page
                "confidence": "high",
                "regions": [],
                "flags": [],
            }
        log(f"  mock: transcribed {len(pages)} pages (placeholder text)")
        return results
