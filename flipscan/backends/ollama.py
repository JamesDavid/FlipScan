"""Ollama backend: local vision model over HTTP, sequential with one retry."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable

import httpx

from . import (ORIENTATION_PROMPT, PROMPT, TranscriptionBackend,
               TranscriptionError, parse_orientation, parse_result)


class OllamaBackend(TranscriptionBackend):
    name = "ollama"

    def __init__(self, cfg: dict):
        p = cfg["provider"]
        self.base_url = p["ollama_url"].rstrip("/")
        self.model = p["ollama_model"]
        self.num_predict = p.get("ollama_num_predict", 4096)
        self.max_retries = cfg["transcribe"]["max_retries"]

    def _request(self, image_b64: str) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                "options": {"num_predict": self.num_predict, "temperature": 0},
                "messages": [
                    {"role": "user", "content": PROMPT, "images": [image_b64]}
                ],
            },
            timeout=600.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def check_orientation(self, image_path: Path) -> bool | None:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    # thinking models spend tokens reasoning before answering
                    "options": {"num_predict": 512, "temperature": 0},
                    "messages": [{
                        "role": "user", "content": ORIENTATION_PROMPT,
                        "images": [base64.standard_b64encode(
                            image_path.read_bytes()).decode()],
                    }],
                },
                timeout=300.0,
            )
            resp.raise_for_status()
            return parse_orientation(resp.json()["message"]["content"])
        except (httpx.HTTPError, KeyError):
            return None

    def transcribe(self, pages: list[tuple[str, Path]],
                   log: Callable[[str], None] = print) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for i, (page_id, image_path) in enumerate(pages):
            image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
            last_err = None
            for attempt in range(self.max_retries + 1):
                try:
                    results[page_id] = parse_result(self._request(image_b64))
                    break
                except (TranscriptionError, httpx.HTTPError) as e:
                    last_err = e
            else:
                results[page_id] = {"error": str(last_err)}
            log(f"  ollama: {page_id} "
                f"({'ok' if 'error' not in results[page_id] else 'FAILED'}) "
                f"[{i + 1}/{len(pages)}]")
        return results
