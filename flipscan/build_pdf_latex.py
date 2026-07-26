"""Optional high-quality PDF via Pandoc + XeLaTeX.

An opt-in output that produces much nicer typography than the reportlab
reflowed PDF, at the cost of two external tools: `pandoc` and `xelatex`. We use
TeX Gyre Termes/Heros (they ship with texlive-fonts-recommended) so there's no
extra font to install. The e-ink page geometry and typography settings are
adapted from reCompose (MIT-licensed, github.com/mrodger/reCompose) — its
rm2.latex template pioneered the reMarkable-2 preset; here we apply the same
ideas through pandoc's default template + a small header include.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .build_pdf import PAGE_GEOMETRY_MM
from .workspace import Workspace

# e-ink typography niceties layered on top of pandoc's default LaTeX template:
# optical margins, block paragraphs, no widows/orphans, looser tolerance for a
# narrow column (all adapted from reCompose's rm2.latex).
_EINK_HEADER = r"""
\usepackage[protrusion=true]{microtype}
\usepackage{parskip}
\clubpenalty=10000
\widowpenalty=10000
\setlength{\emergencystretch}{2em}
\tolerance=400
\setlength{\parskip}{5pt plus 1pt minus 1pt}
"""


def latex_tools_available() -> bool:
    return bool(shutil.which("pandoc") and shutil.which("xelatex"))


def _require(tool: str, hint: str) -> None:
    if not shutil.which(tool):
        raise RuntimeError(f"{tool} not found — {hint}")


def build_pdf_latex(ws: Workspace, out_path: Path, title: str | None = None,
                    author: str | None = None, device: str = "none",
                    log=print) -> Path:
    """Render work/book.md to a PDF via pandoc + xelatex. Raises RuntimeError
    (with an install hint) if the tools are missing, or with the tail of the
    LaTeX log if the compile fails."""
    _require("pandoc", "install pandoc (https://pandoc.org/installing.html)")
    _require("xelatex", "install a TeX distribution with XeLaTeX — "
                        "texlive-xetex + texlive-fonts-recommended on Debian/Ubuntu, "
                        "or MiKTeX (Windows) / MacTeX (macOS)")

    book_md = ws.work_file("book.md")
    if not book_md.exists():
        raise RuntimeError("work/book.md missing — run the pipeline (assemble) first")

    geo = PAGE_GEOMETRY_MM.get(device)
    if geo:
        geometry = (f"paperwidth={geo['w']}mm,paperheight={geo['h']}mm,"
                    f"margin={geo['margin']}mm")
        eink = geo["eink"]
    else:
        geometry = "paperwidth=148mm,paperheight=210mm,margin=15mm"  # A5
        eink = False

    title = title or ws.manifest["book"].get("title") or ws.root.name
    author = author or ws.manifest["book"].get("author") or ""

    header = ws.work_file("_eink_header.tex")
    header.write_text(_EINK_HEADER, encoding="utf-8")

    # run with cwd=ws.root so the figures/... image paths in book.md resolve
    args = [
        "pandoc", "work/book.md", "-o", str(out_path.resolve()),
        "--pdf-engine=xelatex",
        "--from", "markdown+tex_math_dollars",
        "-V", "documentclass=article",
        "-V", "fontsize=11pt",
        "-V", "mainfont=TeX Gyre Termes",
        "-V", "sansfont=TeX Gyre Heros",
        "-V", f"geometry:{geometry}",
        "-V", f"linestretch={1.15 if eink else 1.1}",
        "-V", "colorlinks=true",
        "-V", f"linkcolor={'black' if eink else 'RoyalBlue'}",
        "-V", f"urlcolor={'black' if eink else 'RoyalBlue'}",
        "-H", str(header.resolve()),
        "--metadata", f"title={title}",
    ]
    if author:
        args += ["--metadata", f"author={author}"]

    try:
        r = subprocess.run(args, cwd=str(ws.root), capture_output=True,
                           text=True, timeout=900)
    finally:
        header.unlink(missing_ok=True)

    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout or "").strip().splitlines()[-10:])
        raise RuntimeError("pandoc/xelatex failed:\n" + tail)
    log(f"LaTeX PDF written: {out_path}"
        + (f" ({device})" if geo else ""))
    return out_path
