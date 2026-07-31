"""Build-output freshness: fingerprint what an output was built from.

Shared by the web app (build endpoint, output tab) and the job worker (the
audiobook job), so every artifact in out/ — EPUB, PDFs, markdown zip, m4b —
records the same signature and gets an honest stale/current badge. Signatures
are stored per COMPONENT (book text, figures, proofs, cover, assemble state)
as well as combined, so a stale badge can say WHY: "the book text changed",
not just "stale".
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from .workspace import Workspace


def _sha(data: str) -> str:
    return hashlib.sha1(data.encode()).hexdigest()[:12]


def signature_parts(ws: Workspace) -> dict[str, str]:
    """Independent fingerprints of each thing an output is built from."""
    book = ws.work_file("book.md")
    text = book.read_text(encoding="utf-8") if book.exists() else ""

    figs = []
    for rel in sorted(set(re.findall(r"\]\((figures/[^)]+)\)", text))):
        f = ws.root / rel
        figs.append(f"{rel}|{f.stat().st_mtime_ns if f.exists() else 'gone'}")

    proofs = []
    pdir = ws.work_file("proof")
    if pdir.exists():
        for f in sorted(pdir.glob("proof_*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            proofs.append(f"{f.name}|{d.get('status')}|{d.get('base_hash')}|"
                          f"{len(d.get('proofed_md') or '')}")

    cover = next((p for p in ws.manifest["pages"]
                  if p.get("role") == "cover"), None)
    cover_key = ""
    if cover and cover.get("color"):
        f = ws.root / cover["color"]
        if f.exists():
            cover_key = str(f.stat().st_mtime_ns)

    return {
        "assemble": ws.stage_status("assemble"),
        "book": _sha(text),
        "figures": _sha("\n".join(figs)),
        "proofs": _sha("\n".join(proofs)),
        "cover": _sha(cover_key),
    }


def build_signature(ws: Workspace) -> str:
    """Combined fingerprint. (Kept byte-compatible with the original
    single-hash algorithm so outputs stamped before per-component tracking
    keep an honest current/stale verdict.)"""
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
    """Stamp `filename` (in out/) with what it was built from — combined sig
    for the verdict, per-component parts for the WHY, and a build time."""
    ws.manifest.setdefault("outputs_built", {})[filename] = {
        "sig": build_signature(ws),
        "parts": signature_parts(ws),
        "at": time.time(),
    }
    ws.save()


_REASON = {
    "book": "the book text changed",
    "figures": "figure images changed",
    "proofs": "proofread decisions changed",
    "cover": "the cover changed",
    "assemble": "the book was reassembled",
}


def output_status(ws: Workspace) -> list[dict]:
    """[{name, stale, reasons, built_at}] for every file in out/."""
    sig = build_signature(ws)
    parts = signature_parts(ws)
    built = ws.manifest.get("outputs_built", {})
    rows = []
    for f in sorted(ws.dir("out").glob("*")):
        if not f.is_file():
            continue
        stored = built.get(f.name)
        stale, reasons, at = True, [], None
        if isinstance(stored, dict):
            stale = stored.get("sig") != sig
            at = stored.get("at")
            if stale:
                old = stored.get("parts") or {}
                reasons = [_REASON[k] for k in _REASON
                           if old.get(k) != parts[k]]
                if not reasons:
                    reasons = ["the book state changed"]
        elif isinstance(stored, str):     # stamped before per-component tracking
            stale = stored != sig
            if stale:
                reasons = ["built from older book text (before change tracking "
                           "— rebuild once and future badges say exactly what "
                           "changed)"]
        else:                             # never stamped at all
            reasons = ["no build record — rebuild once to start tracking"]
        rows.append({"name": f.name, "stale": stale, "reasons": reasons,
                     "built_at": at})
    return rows
