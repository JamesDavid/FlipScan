"""Stage 6: preprocess — crop canonical frames to the page and perspective-correct.

Per page writes into work/pages/:
  <id>_color.png  full-res corrected color frame (source for figure crops)
  <id>_llm.jpg    contrast-normalized grayscale copy downscaled for the LLM
Dewarp (page curl) is deferred; config["preprocess"]["dewarp"] is a no-op hook.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging import detect_page_quad
from ..workspace import Workspace
from .score import scores_by_frame_id
from .select import frame_path


def correct_page(bgr: np.ndarray, quad_norm) -> np.ndarray:
    """Perspective-correct the page quad to an upright rectangle."""
    h, w = bgr.shape[:2]
    quad = np.array(quad_norm, dtype=np.float64) * [w, h]
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    tw = int(round((top + bottom) / 2))
    th = int(round((left + right) / 2))
    if tw < 50 or th < 50:
        return bgr
    dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float64)
    m = cv2.getPerspectiveTransform(quad.astype(np.float32), dst.astype(np.float32))
    return cv2.warpPerspective(bgr, m, (tw, th))


def llm_copy(color: np.ndarray, long_edge: int) -> np.ndarray:
    """Grayscale, CLAHE contrast normalization, downscale to the LLM budget."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape
    scale = long_edge / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


def preprocess_page(ws: Workspace, page: dict, cfg: dict,
                    scores: dict[str, dict] | None = None) -> None:
    """Correct one page's canonical frame; used by the stage and the patch flow."""
    out_dir = ws.work_file("pages")
    out_dir.mkdir(exist_ok=True)
    if page.get("patched_source"):
        fid = None
        bgr = cv2.imread(str(ws.root / page["patched_source"]))
    else:
        fid = page["canonical"]
        bgr = cv2.imread(str(frame_path(ws, fid)))

    quad = None
    if fid is not None and scores and fid in scores:
        quad = scores[fid].get("quad")
    if quad is None:  # patched photos have no score record: detect now
        q, _ = detect_page_quad(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        quad = q.tolist() if q is not None else None

    color = correct_page(bgr, quad) if quad is not None else bgr
    color_path = out_dir / f"{page['id']}_color.png"
    llm_path = out_dir / f"{page['id']}_llm.jpg"
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(llm_path), llm_copy(color, cfg["preprocess"]["llm_long_edge"]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    page["color"] = f"work/pages/{page['id']}_color.png"
    page["llm_image"] = f"work/pages/{page['id']}_llm.jpg"


def run(ws: Workspace, cfg: dict, log=print) -> None:
    scores = scores_by_frame_id(ws)
    pages = ws.manifest["pages"]
    for i, page in enumerate(pages):
        preprocess_page(ws, page, cfg, scores)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(pages)}")
    ws.save()
    ws.stage_done("preprocess")
    log(f"  {len(pages)} pages corrected -> work/pages/")
