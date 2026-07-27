"""OpenAI-compatible vision backend.

One backend for the many providers that speak the OpenAI Chat Completions API
with image input + JSON mode: OpenAI itself, Google Gemini (its OpenAI endpoint),
OpenRouter (dozens of vision models behind one key), Groq, Together, xAI, Azure
OpenAI, and local servers (vLLM / LM Studio / Ollama's /v1). The user just sets
a base URL, an API key, and a model name.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Callable

import httpx

from . import (ORIENTATION_PROMPT, PROMPT, TranscriptionBackend,
               TranscriptionError, parse_orientation, parse_result,
               salvage_result)


class OpenAICompatBackend(TranscriptionBackend):
    name = "openai"

    def __init__(self, cfg: dict):
        p = cfg["provider"]
        self.base_url = (p.get("openai_base_url")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = p.get("openai_model") or "gpt-4o"
        self.api_key = (p.get("openai_api_key")
                        or os.environ.get("FLIPSCAN_OPENAI_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or "")
        self.max_tokens = int(p.get("openai_max_tokens", 4096))
        self.max_retries = cfg["transcribe"]["max_retries"]
        self._json_mode = True          # disabled if a model rejects it (400)
        self.name = self.model          # informative in logs / transcribed_by

    def _chat(self, image_b64: str, prompt: str, max_tokens: int) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        def call(json_mode: bool) -> str:
            body = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"}}]}],
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            r = httpx.post(f"{self.base_url}/chat/completions", json=body,
                           headers=headers, timeout=180.0)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return msg.get("content") or ""

        try:
            return call(self._json_mode)
        except httpx.HTTPStatusError as e:
            # some models/endpoints reject response_format=json_object — the
            # prompt already demands raw JSON, so fall back without it
            if (self._json_mode and e.response is not None
                    and e.response.status_code == 400):
                self._json_mode = False
                return call(False)
            raise

    def transcribe(self, pages: list[tuple[str, Path]],
                   log: Callable[[str], None] = print) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for i, (page_id, image_path) in enumerate(pages):
            b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
            last_err = last_raw = None
            for attempt in range(self.max_retries + 1):
                try:
                    raw = self._chat(b64, PROMPT, self.max_tokens * (2 ** attempt))
                except httpx.HTTPError as e:
                    last_err = e
                    continue
                try:
                    results[page_id] = parse_result(raw)
                    break
                except TranscriptionError as e:
                    last_err, last_raw = e, raw
            else:
                results[page_id] = (salvage_result(last_raw)
                                    or {"error": str(last_err)})
            ok = "error" not in results[page_id]
            log(f"  {self.model}: {page_id} ({'ok' if ok else 'FAILED'}) "
                f"[{i + 1}/{len(pages)}]")
        return results

    def check_orientation(self, image_path: Path) -> bool | None:
        try:
            b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
            return parse_orientation(self._chat(b64, ORIENTATION_PROMPT, 200))
        except Exception:
            return None
