"""Build-output freshness: fingerprint what an output was built from.

Shared by the web app (build endpoint, output tab) and the job worker (the
audiobook job), so every artifact in out/ — EPUB, PDFs, markdown zip, m4b —
records the same signature and gets an honest stale/current badge.
"""

from __future__ import annotations

import hashlib
import json
import re

from .workspace import Workspace


def build_signature(ws: Workspace) -> str:
    """Fingerprint of everything a built output depends on: the assembled
    book text, every referenced figure file, proof states, and whether
    assemble is even up to date. Any page/figure/proof change after a
    build makes the output stale."""
    h = hashlib.sha1()
    h.update(ws.stage_status("assemble").encode())
    book = ws.work_file("book.md")
    if book.exists():
        text = book.read_text(encoding="utf-8")
        h.update(text.encode())
        for rel in sorted(set(re.findall(r"\]\((figures/[^)]+)\)", text))):
            f = ws.root / rel
            h.update(rel.encode())
            if f.exists():
                h.update(str(f.stat().st_mtime_ns).encode())
    pdir = ws.work_file("proof")
    if pdir.exists():
        for f in sorted(pdir.glob("proof_*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            h.update(f"{f.name}|{d.get('status')}|{d.get('base_hash')}|"
                     f"{len(d.get('proofed_md') or '')}".encode())
    cover = next((p for p in ws.manifest["pages"]
                  if p.get("role") == "cover"), None)
    if cover and cover.get("color"):
        f = ws.root / cover["color"]
        if f.exists():
            h.update(str(f.stat().st_mtime_ns).encode())
    return h.hexdigest()[:16]


def record_output(ws: Workspace, filename: str) -> None:
    """Stamp `filename` (in out/) as built from the current state — the output
    tab shows it 'current' until pages/figures/proofs change again."""
    ws.manifest.setdefault("outputs_built", {})[filename] = build_signature(ws)
    ws.save()
