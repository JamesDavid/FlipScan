"""Ollama backend: local vision model over HTTP, sequential with one retry."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Callable

import httpx

from . import (ORIENTATION_PROMPT, PROMPT, TranscriptionBackend,
               TranscriptionError, parse_orientation, parse_result,
               salvage_result)


def _looks_multicolumn(gray) -> bool:
    """True if a printed page has a pronounced vertical whitespace gutter near
    the centre — i.e. a two-column layout, which should be split left/right
    rather than top/bottom so each column stays in reading order."""
    import numpy as np
    col_ink = (gray < 160).sum(axis=0).astype(float)  # ink per column
    n = col_ink.shape[0]
    if n < 40:
        return False
    lo, hi = int(n * 0.38), int(n * 0.62)
    center_min = col_ink[lo:hi].min() if hi > lo else col_ink.min()
    nz = col_ink[col_ink > 0]
    med = float(np.median(nz)) if nz.size else 1.0
    return center_min < 0.12 * med


def _gutter(gray, by: str) -> int:
    """Row (by='row') or column (by='col') near the middle with the least ink,
    so a split falls in a gap between lines/columns rather than through text."""
    import numpy as np
    prof = (gray < 160).sum(axis=1 if by == "row" else 0).astype(float)
    n = prof.shape[0]
    lo, hi = int(n * 0.40), int(n * 0.60)
    if hi <= lo:
        return n // 2
    return lo + int(np.argmin(prof[lo:hi]))


def _join_split(parts: list[str]) -> str:
    """Join transcribed page-halves without a spurious blank line where the
    split fell: stitch a word hyphenated across the cut, flow a paragraph that
    continued mid-sentence with a single space, and only keep a paragraph break
    at a real boundary (sentence end, heading, list, blockquote)."""
    import re
    out = ""
    for raw in parts:
        seg = (raw or "").strip()
        if not seg:
            continue
        if not out:
            out = seg
            continue
        out = out.rstrip()
        last = out.rsplit("\n", 1)[-1].strip()
        first = seg.split("\n", 1)[0].strip()
        starts_block = bool(re.match(r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s|>)", first))
        if last.endswith("-"):                       # hyphenated word across cut
            out = out[:-1] + seg
        elif last and last[-1] not in ".!?:;\"')]}" and not starts_block:
            out = out + " " + seg                    # same paragraph — flow it
        else:
            out = out + "\n\n" + seg                 # real break
    return out


def _split_halves(img, cols: bool):
    """Two sub-images: left/right if `cols` else top/bottom, cut at the quietest
    gutter so no line/column is sliced through."""
    import cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cols:
        c = _gutter(gray, "col")
        return [img[:, :c], img[:, c:]]
    r = _gutter(gray, "row")
    return [img[:r, :], img[r:, :]]


class OllamaBackend(TranscriptionBackend):
    name = "ollama"

    def __init__(self, cfg: dict):
        p = cfg["provider"]
        self.base_url = p["ollama_url"].rstrip("/")
        self.model = p["ollama_model"]
        self.num_predict = p.get("ollama_num_predict", 4096)
        self.think = p.get("ollama_think", False)
        self.concurrency = int(p.get("ollama_concurrency", 1))
        self.max_retries = cfg["transcribe"]["max_retries"]

    def _request(self, image_b64: str, num_predict: int | None = None,
                 extra_options: dict | None = None) -> str:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "format": "json",
                # thinking off: transcription is perception, not reasoning —
                # thinking models otherwise ramble for minutes on hard pages
                "think": self.think,
                "options": {"num_predict": num_predict or self.num_predict,
                            "temperature": 0, **(extra_options or {})},
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
                    "think": self.think,
                    # headroom in case thinking is enabled in config
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

    def _attempt(self, image_b64: str):
        """Retry ladder for one image. Returns (parsed_result | None, last_raw).
        Retries get a doubled token budget (dense pages truncate their JSON) and
        a repetition penalty (greedy decoding can lock into one phrase)."""
        last_raw = None
        for attempt in range(self.max_retries + 1):
            budget = self.num_predict * (2 ** attempt)
            opts = None if attempt == 0 else {"repeat_penalty": 1.15,
                                              "repeat_last_n": 256}
            try:
                raw = self._request(image_b64, budget, opts)
            except httpx.HTTPError:
                continue
            try:
                return parse_result(raw), raw
            except TranscriptionError:
                last_raw = raw
        return None, last_raw

    def _b64(self, img) -> str:
        import cv2
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.standard_b64encode(buf.tobytes()).decode()

    def _recover_split(self, img, depth: int = 1) -> dict | None:
        """A page that won't transcribe whole (the JSON keeps hitting the token
        limit) is split in half — left/right for columns, else top/bottom — and
        each half is transcribed separately, then joined in reading order. Still
        too dense? Split again (up to 2 deep). Last resort per part: salvage its
        partial text. This recovers the WHOLE page instead of a fragment."""
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cols = _looks_multicolumn(gray)
        mds, pnum = [], None
        for part in _split_halves(img, cols):
            if part.shape[0] < 8 or part.shape[1] < 8:
                continue
            res, raw = self._attempt(self._b64(part))
            if res is None and depth < 2:
                res = self._recover_split(part, depth + 1)
            if res is None:
                res = salvage_result(raw)
            if res is None:
                return None                       # a part yielded nothing
            if res.get("markdown", "").strip():
                mds.append(res["markdown"].strip())
            if pnum is None:
                pnum = res.get("page_number_printed")
        if not mds:
            return None
        return {"markdown": _join_split(mds), "page_number_printed": pnum,
                "confidence": "low", "regions": [],
                "flags": ["split_recovered"] + (["multi_column"] if cols else [])}

    def transcribe(self, pages: list[tuple[str, Path]],
                   log: Callable[[str], None] = print) -> dict[str, dict]:
        results: dict[str, dict] = {}
        for i, (page_id, image_path) in enumerate(pages):
            img_bytes = image_path.read_bytes()
            res, last_raw = self._attempt(
                base64.standard_b64encode(img_bytes).decode())
            state = "ok"
            if res is None:                        # whole page failed → split it
                import cv2
                import numpy as np
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),
                                   cv2.IMREAD_COLOR)
                res = self._recover_split(img) if img is not None else None
                if res is not None:
                    state = "recovered (split)"
                else:                              # split failed too → salvage
                    res = salvage_result(last_raw)
                    state = "salvaged" if res else "FAILED"
            elif "cut_off_text" in (res.get("flags") or []):
                # the whole page PARSED, but the model says text is cut off. On a
                # two-column layout that almost always means it read one column
                # and silently dropped the other. Split the columns, OCR each,
                # and keep the result if it recovers more text. (Gate on the
                # image gutter so a genuinely single-column page whose edge was
                # cut off isn't sliced down the middle.)
                import cv2
                import numpy as np
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8),
                                   cv2.IMREAD_COLOR)
                gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if img is not None else None)
                if gray is not None and _looks_multicolumn(gray):
                    split = self._recover_split(img)
                    if split and len(split.get("markdown", "")) > len(
                            res.get("markdown", "")):
                        if res.get("page_number_printed") is not None:
                            split["page_number_printed"] = res["page_number_printed"]
                        split["flags"] = sorted(set(res.get("flags", []))
                                                | set(split.get("flags", [])))
                        res = split
                        state = "recovered (2-col split)"
            results[page_id] = res or {"error": "no usable response from model"}
            log(f"  ollama: {page_id} ({state}) [{i + 1}/{len(pages)}]")
        return results
