"""Stage 8: figures — crop LLM-reported regions from the corrected color frames.

The LLM bbox is approximate: expand it, then snap to actual content using
background-deviation analysis on the full-res corrected frame. Cropped images
are inserted at their [[region-N]] placeholders in the page markdown.
"""

from __future__ import annotations

import string

import cv2
import numpy as np

from ..imaging import sharpness
from ..workspace import Workspace

EXPAND = 0.075          # grow the LLM bbox by 7.5% per side before snapping
SNAP_MARGIN = 8         # px kept around detected content
MIN_FIGURE_SHARPNESS = 40.0


def snap_bbox(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
    """Tighten a bbox to the content inside it (anything deviating from the
    page background, estimated from the crop's border pixels)."""
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return x0, y0, x1, y1
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    bg = float(np.median(border))
    mask = np.abs(crop.astype(np.int16) - bg) > 25
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return x0, y0, x1, y1
    ry, cx = np.where(rows)[0], np.where(cols)[0]
    return (
        max(0, x0 + int(cx[0]) - SNAP_MARGIN),
        max(0, y0 + int(ry[0]) - SNAP_MARGIN),
        min(gray.shape[1], x0 + int(cx[-1]) + 1 + SNAP_MARGIN),
        min(gray.shape[0], y0 + int(ry[-1]) + 1 + SNAP_MARGIN),
    )


def run(ws: Workspace, cfg: dict, log=print) -> None:
    fig_dir = ws.dir("figures")
    total = 0
    for page in ws.manifest["pages"]:
        regions = page.get("regions") or []
        if (not regions or not page.get("md") or not page.get("color")
                or page.get("status") in ("duplicate", "deleted")):
            continue
        color = cv2.imread(str(ws.root / page["color"]))
        if color is None:
            continue
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        md_path = ws.root / page["md"]
        md = md_path.read_text(encoding="utf-8")
        page_figs = []

        for i, region in enumerate(regions):
            if region.get("user_crop") or region.get("deleted"):
                continue  # user-cropped: never overwrite; deleted: stay gone
            bx0, by0, bx1, by1 = region["bbox_norm"]
            dx, dy = (bx1 - bx0) * EXPAND, (by1 - by0) * EXPAND
            x0 = int(max(0.0, bx0 - dx) * w)
            y0 = int(max(0.0, by0 - dy) * h)
            x1 = int(min(1.0, bx1 + dx) * w)
            y1 = int(min(1.0, by1 + dy) * h)
            x0, y0, x1, y1 = snap_bbox(gray, x0, y0, x1, y1)
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            crop = color[y0:y1, x0:x1]
            letter = string.ascii_lowercase[i % 26]
            name = f"{page['id']}_{letter}.png"
            cv2.imwrite(str(fig_dir / name), crop)
            rel = f"figures/{name}"
            page_figs.append(rel)
            total += 1

            if sharpness(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < MIN_FIGURE_SHARPNESS:
                page["figure_quality"] = True
                if page["status"] == "ok":
                    page["status"] = "suspect"

            img_md = f"![{region.get('caption') or ''}]({rel})"
            placeholder = f"[[region-{i}]]"
            if placeholder in md:
                md = md.replace(placeholder, img_md)
            else:
                md = md.rstrip() + f"\n\n{img_md}\n"

        md_path.write_text(md, encoding="utf-8")
        page["figures"] = page_figs
    ws.save()
    ws.stage_done("figures", figure_count=total)
    log(f"  {total} figures cropped -> figures/")
