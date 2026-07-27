"""Static review page: canonical frame vs rendered markdown, plus the reshoot list."""

from __future__ import annotations

from pathlib import Path

import markdown as md_lib
from jinja2 import Environment, PackageLoader, select_autoescape

from .workspace import Workspace


def find_page_by_text(ws: Workspace, snippet: str) -> dict | None:
    """Map a passage of assembled/rendered text back to its source page.
    Squashing to lowercase alphanumerics makes the match survive hyphenation,
    spacing, and punctuation differences between page OCR and book text."""
    import re

    def squash(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    target = squash(snippet)[:160]
    if len(target) < 12:
        return None
    for p in ws.manifest["pages"]:
        if (p.get("status") in ("duplicate", "deleted") or p.get("role")
                or not p.get("md")):
            continue
        f = ws.root / p["md"]
        if f.exists() and target in squash(f.read_text(encoding="utf-8")):
            return p
    return None


def page_reasons(p: dict) -> list[str]:
    """Why this page needs attention — shared by the reshoot list, the page
    cells, and the capture wizards.

    'Ignore suspect' only waives the MODEL's quality complaints. Explicit
    user actions (re-acquisition marks) and structural problems (number
    conflicts, missing crops, failures) always count — otherwise marking an
    accepted page for re-acquisition would silently do nothing."""
    ignored = bool(p.get("suspect_ignored"))
    reasons = []
    if p.get("transcribe_error"):
        reasons.append(f"transcription failed ({p['transcribe_error'][:80]})")
    if not ignored:
        if p.get("confidence") == "low":
            reasons.append("low transcription confidence")
        for f in p.get("flags") or []:
            if f == "multi_column":
                # a reshoot can't fix a layout; the model reads both columns —
                # this is only a check-the-reading-order note
                reasons.append("note: multi-column layout — verify text order")
            else:
                reasons.append(f"flagged: {f}")
        if p.get("figure_quality"):
            reasons.append("figure source frame is low quality")
    if p.get("needs_reshoot"):
        note = f" — “{p['flag_note']}”" if p.get("flag_note") else ""
        reasons.append(f"marked for re-acquisition by you{note}")
    if any(r.get("needs_reshoot") for r in p.get("regions") or []):
        reasons.append("figure marked for re-acquisition (shoot it close-up)")
    if any(r.get("stale_crop") for r in p.get("regions") or []):
        reasons.append("figure crop was drawn on an older page image — re-crop it")
    if p.get("number_rejected"):
        reasons.append("page number misread (broke the page order)")
    if p.get("number_conflict"):
        reasons.append("more captures than page numbers fit here")
    figs = p.get("figures") or []
    for ri, r in enumerate(p.get("regions") or []):
        expected = f"figures/{p['id']}_{chr(97 + ri % 26)}.png"
        if not r.get("deleted") and expected not in figs:
            reasons.append(f"figure region {ri} has no crop")
            break
    if p["status"] == "suspect" and not reasons and not ignored:
        reasons.append("weak capture (short cluster or low frame score)")
    if p["status"] == "missing":
        reasons.append("no usable frame captured")
    return reasons


def uncropped_regions(p: dict) -> list[int]:
    """Region indices that have a [[region-N]] placeholder but no cropped figure
    file yet — i.e. a crop still needs to be drawn."""
    figs = p.get("figures") or []
    out = []
    for ri, r in enumerate(p.get("regions") or []):
        expected = f"figures/{p['id']}_{chr(97 + ri % 26)}.png"
        if not r.get("deleted") and expected not in figs:
            out.append(ri)
    return out


def _needs_reshoot_work(p: dict) -> bool:
    """True if the PAGE itself needs re-photographing or a structural fix —
    something a crop can't solve and 'looks fine' can't waive."""
    if p.get("transcribe_error") or p.get("needs_reshoot"):
        return True
    if p.get("number_rejected") or p.get("number_conflict"):
        return True
    if p.get("status") == "missing":
        return True
    return any(r.get("needs_reshoot") or r.get("stale_crop")
               for r in p.get("regions") or [])


def crop_list(ws: Workspace) -> list[dict]:
    """Pages whose only real to-do is drawing a figure crop (the region marker
    is there but no crop exists) — including pages whose OTHER complaints are
    all dismissible quality suspicions ('looks fine'). Pages that genuinely
    need re-shooting stay on the reshoot list instead."""
    items = []
    for p in ws.manifest["pages"]:
        if p["status"] in ("duplicate", "deleted") or _needs_reshoot_work(p):
            continue
        regions = uncropped_regions(p)
        if regions:
            items.append({
                "id": p["id"], "printed_number": p.get("printed_number"),
                "regions": [{"idx": ri,
                             "caption": (p["regions"][ri].get("caption") or "")}
                            for ri in regions],
                # remember whether we also cleared a dismissible suspicion here
                "suspect": p["status"] == "suspect" and not p.get("suspect_ignored"),
            })
    return items


def reocr_list(ws: Workspace) -> list[dict]:
    """Pages whose transcription outright failed — a re-OCR (which now uses the
    split-and-recover pass) is the fix, not a reshoot."""
    items = []
    for p in ws.manifest["pages"]:
        if p["status"] in ("duplicate", "deleted"):
            continue
        if p.get("transcribe_error"):
            items.append({
                "id": p["id"], "printed_number": p.get("printed_number"),
                "error": str(p["transcribe_error"])[:140],
                "can_reocr": bool(p.get("llm_image")),
            })
    return items


def reshoot_list(ws: Workspace) -> list[dict]:
    """Pages that should be re-photographed, with reasons. Pages whose only work
    is a missing crop (crop_list) or a failed transcription (reocr_list) are
    excluded; the 'no crop' / 'transcription failed' reasons are dropped from
    pages that remain for a genuine reshoot."""
    items = []
    for p in ws.manifest["pages"]:
        if p["status"] in ("duplicate", "deleted"):
            continue
        # a failed transcription belongs to the re-OCR list, not here
        if p.get("transcribe_error"):
            continue
        # a pure crop page belongs to the crop list, not here
        if uncropped_regions(p) and not _needs_reshoot_work(p):
            continue
        # informational notes (multi-column etc.) don't justify a reshoot; the
        # 'no crop' note is handled by the crop list
        reasons = [r for r in page_reasons(p)
                   if not r.startswith("note:")
                   and not (r.startswith("figure region ") and "has no crop" in r)]
        if reasons:
            items.append({"id": p["id"], "printed_number": p.get("printed_number"),
                          "reasons": reasons})
    return items


def generate_review(ws: Workspace, log=print) -> Path:
    env = Environment(loader=PackageLoader("flipscan", "templates"),
                      autoescape=select_autoescape(["html"]))
    tmpl = env.get_template("review.html.j2")

    pages = []
    for p in ws.manifest["pages"]:
        md_text = ""
        if p.get("md") and (ws.root / p["md"]).exists():
            md_text = (ws.root / p["md"]).read_text(encoding="utf-8")
            md_text = md_text.replace("](figures/", "](../figures/")
        image = None
        if p.get("color") and (ws.root / p["color"]).exists():
            image = f"../{p['color'].replace(chr(92), '/')}"
        pages.append({
            **p,
            "image": image,
            "html": md_lib.markdown(md_text, extensions=["tables"]) if md_text
                    else "<em>no transcription</em>",
        })

    from .stages.transcribe import format_ranges
    missing = ws.manifest.get("missing_pages", [])
    out = ws.dir("review") / "index.html"
    out.write_text(tmpl.render(
        title=ws.manifest["book"].get("title") or ws.root.name,
        workspace=str(ws.root),
        pages=pages,
        suspect_count=sum(1 for p in pages if p["status"] == "suspect"),
        reshoot=reshoot_list(ws),
        missing=missing,
        missing_ranges=format_ranges(missing),
    ), encoding="utf-8")
    log(f"review page: {out}")
    return out
