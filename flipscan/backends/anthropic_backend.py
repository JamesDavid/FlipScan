"""Anthropic backend: Message Batches API (50% discount, fine for offline processing)."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Callable

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from . import (ORIENTATION_PROMPT, PROMPT, TranscriptionBackend,
               TranscriptionError, parse_orientation, parse_result)


class AnthropicBackend(TranscriptionBackend):
    name = "anthropic"

    def __init__(self, cfg: dict):
        api_key = (os.environ.get("FLIPSCAN_ANTHROPIC_API_KEY")
                   or os.environ.get("ANTHROPIC_API_KEY")
                   or cfg["provider"].get("anthropic_api_key") or None)
        # None -> SDK resolves its own credential chain (env, `ant auth login` profile)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = cfg["provider"]["anthropic_model"]

    def check_orientation(self, image_path: Path) -> bool | None:
        try:
            image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=64,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/jpeg",
                                    "data": image_b64}},
                        {"type": "text", "text": ORIENTATION_PROMPT},
                    ],
                }],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            return parse_orientation(text)
        except anthropic.APIError:
            return None

    def transcribe(self, pages: list[tuple[str, Path]],
                   log: Callable[[str], None] = print) -> dict[str, dict]:
        requests = []
        for page_id, image_path in pages:
            image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
            requests.append(Request(
                custom_id=page_id,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=8192,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image",
                             "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": image_b64}},
                            {"type": "text", "text": PROMPT},
                        ],
                    }],
                ),
            ))

        batch = self.client.messages.batches.create(requests=requests)
        log(f"  anthropic: batch {batch.id} submitted ({len(requests)} pages)")

        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            log(f"  anthropic: {batch.processing_status} "
                f"({batch.request_counts.processing} processing, "
                f"{batch.request_counts.succeeded} done)")
            time.sleep(30)

        results: dict[str, dict] = {}
        for result in self.client.messages.batches.results(batch.id):
            pid = result.custom_id
            if result.result.type == "succeeded":
                msg = result.result.message
                text = next((b.text for b in msg.content if b.type == "text"), "")
                try:
                    results[pid] = parse_result(text)
                except TranscriptionError as e:
                    results[pid] = {"error": str(e)}
            else:
                results[pid] = {"error": f"batch result: {result.result.type}"}
        log(f"  anthropic: batch done, "
            f"{sum(1 for r in results.values() if 'error' not in r)}/{len(pages)} ok")
        return results


FIGURE_BBOX_PROMPT = """This is a photograph of one book page containing printed \
photograph(s)/illustration(s). For EVERY printed figure on the page, give its \
tight bounding box as fractions of the full image (0.0 top-left to 1.0 \
bottom-right), enclosing exactly the figure's printed edges — exclude captions, \
body text, page margins, and anything outside the figure. Respond with JSON only:
{"figures": [{"bbox": [x0, y0, x1, y1], "caption": "nearby caption or empty"}]}
If the page has no printed figure, respond {"figures": []}."""


def claude_figure_boxes(cfg: dict, image_jpeg: bytes) -> list[dict]:
    """Ask Claude vision for figure bounding boxes on a page image.
    Returns [{"bbox": [x0,y0,x1,y1], "caption": str}] with sane boxes only."""
    import base64
    import json as _json
    import re as _re

    p = cfg["provider"]
    client = anthropic.Anthropic(api_key=p.get("anthropic_api_key") or None)
    msg = client.messages.create(
        model=p["anthropic_model"], max_tokens=700,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(image_jpeg).decode()}},
            {"type": "text", "text": FIGURE_BBOX_PROMPT}]}])
    raw = "".join(b.text for b in msg.content if b.type == "text")
    m = _re.search(r"\{.*\}", raw, _re.S)
    if not m:
        return []
    try:
        obj = _json.loads(m.group(0))
    except _json.JSONDecodeError:
        return []
    out = []
    for f in obj.get("figures", []):
        bb = f.get("bbox")
        if (isinstance(bb, list) and len(bb) == 4
                and all(isinstance(v, (int, float)) for v in bb)):
            x0, y0, x1, y1 = (max(0.0, min(1.0, float(v))) for v in bb)
            if x1 - x0 > 0.03 and y1 - y0 > 0.03:
                out.append({"bbox": [x0, y0, x1, y1],
                            "caption": str(f.get("caption") or "")[:200]})
    return out
