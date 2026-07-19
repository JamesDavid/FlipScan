"""Static review page: canonical frame vs rendered markdown, plus the reshoot list."""

from __future__ import annotations

from pathlib import Path

import markdown as md_lib
from jinja2 import Environment, PackageLoader, select_autoescape

from .workspace import Workspace


def reshoot_list(ws: Workspace) -> list[dict]:
    """Pages that should be re-photographed, with reasons."""
    items = []
    for p in ws.manifest["pages"]:
        reasons = []
        if p.get("transcribe_error"):
            reasons.append(f"transcription failed ({p['transcribe_error'][:80]})")
        if p.get("confidence") == "low":
            reasons.append("low transcription confidence")
        for f in p.get("flags") or []:
            reasons.append(f"flagged: {f}")
        if p.get("figure_quality"):
            reasons.append("figure source frame is low quality")
        if p["status"] == "suspect" and not reasons:
            reasons.append("weak capture (short cluster or low frame score)")
        if p["status"] == "missing":
            reasons.append("no usable frame captured")
        if reasons and p["status"] not in ("patched", "duplicate"):
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

    out = ws.dir("review") / "index.html"
    out.write_text(tmpl.render(
        title=ws.manifest["book"].get("title") or ws.root.name,
        workspace=str(ws.root),
        pages=pages,
        suspect_count=sum(1 for p in pages if p["status"] == "suspect"),
        reshoot=reshoot_list(ws),
    ), encoding="utf-8")
    log(f"review page: {out}")
    return out
