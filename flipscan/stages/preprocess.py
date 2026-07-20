"""Stage 6: preprocess — crop canonical frames to the page and perspective-correct.

Per page writes into work/pages/:
  <id>_color.png  full-res corrected color frame (source for figure crops)
  <id>_llm.jpg    contrast-normalized grayscale copy downscaled for the LLM

Handles per-video 180-degree rotation (video shot upside down) and pads the
page quad so edge content (printed page numbers!) survives the crop.
Set config [preprocess] dewarp=true to apply simple cylindrical curl correction.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..imaging import (detect_page_quad, find_flat_page, isolate_book,
                       mask_outside, order_quad)
from ..workspace import Workspace
from .score import scores_by_frame_id
from .select import frame_path


def _video_rotation(ws: Workspace, frame_id: str | None) -> int:
    if not frame_id:
        return 0
    vid = frame_id.split("_", 1)[0]
    for v in ws.manifest["videos"]:
        if v["id"] == vid:
            return v.get("rotate", 0)
    return 0


def _pad_quad(quad: np.ndarray, pad: float) -> np.ndarray:
    """Expand the quad outward about its centroid (page numbers live at the
    very edges; the detected contour often sits just inside them)."""
    center = quad.mean(axis=0)
    return np.clip(center + (quad - center) * (1.0 + 2.0 * pad), 0.0, 1.0)


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


def dewarp_cylindrical(color: np.ndarray) -> np.ndarray:
    """Correct page curl with a simple cylindrical model: fit quadratics to the
    top and bottom envelopes of the ink and remap each column so both run straight.
    Falls back to the input untouched when the page has too little ink to fit."""
    h, w = color.shape[:2]
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    ink = cv2.dilate(ink, np.ones((5, 25), np.uint8))  # merge letters into lines

    xs, tops, bots = [], [], []
    step = max(1, w // 60)
    for x in range(0, w, step):
        col = np.nonzero(ink[:, x:x + step].any(axis=1))[0]
        if len(col) > 10:
            xs.append(x + step / 2)
            tops.append(col[0])
            bots.append(col[-1])
    if len(xs) < 10:
        return color

    xs_a = np.array(xs, dtype=np.float64)
    top_fit = np.poly1d(np.polyfit(xs_a, tops, 2))
    bot_fit = np.poly1d(np.polyfit(xs_a, bots, 2))

    col_x = np.arange(w, dtype=np.float32)
    top_c = top_fit(col_x).astype(np.float32)
    bot_c = bot_fit(col_x).astype(np.float32)
    span = np.maximum(bot_c - top_c, 1.0)
    if float(np.ptp(top_c) + np.ptp(bot_c)) < 4.0:
        return color  # already flat — skip the remap

    top_t, bot_t = float(top_c.min()), float(bot_c.max())
    rows = np.arange(h, dtype=np.float32)[:, None]           # target y
    # invert the per-column linear stretch: source y for each target y
    map_y = top_c[None, :] + (rows - top_t) * (span[None, :] / max(bot_t - top_t, 1.0))
    map_x = np.broadcast_to(col_x[None, :], (h, w)).copy()
    return cv2.remap(color, map_x, map_y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


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
    pad = cfg["preprocess"].get("quad_pad", 0.025)

    if page.get("patched_source"):
        # a deliberate photo is already flat and framed by a human — any crop
        # or warp risks cutting exactly what they framed (page numbers!).
        # Use it as-is.
        bgr = cv2.imread(str(ws.root / page["patched_source"]))
        if bgr is None:
            return
        color = bgr
        if cfg["preprocess"].get("dewarp"):
            color = dewarp_cylindrical(color)
        color_path = out_dir / f"{page['id']}_color.png"
        llm_path = out_dir / f"{page['id']}_llm.jpg"
        cv2.imwrite(str(color_path), color)
        cv2.imwrite(str(llm_path), llm_copy(color, cfg["preprocess"]["llm_long_edge"]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        page["color"] = f"work/pages/{page['id']}_color.png"
        page["llm_image"] = f"work/pages/{page['id']}_llm.jpg"
        page.pop("isolated", None)
        return

    fid = page["canonical"]
    bgr = cv2.imread(str(frame_path(ws, fid)))
    if bgr is None:
        return

    rotation = _video_rotation(ws, fid)
    if rotation == 180:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)

    # edge-density page isolation: crop straight to the flat readable page
    # (lighting-invariant; falls back to the quad path when not confident)
    if (cfg["preprocess"].get("isolate_page", True)
            and not page.get("patched_source") and page.get("role") != "cover"):
        box = find_flat_page(bgr)
        if box is not None:
            color = bgr[box[1]:box[3], box[0]:box[2]]
            if cfg["preprocess"].get("dewarp"):
                color = dewarp_cylindrical(color)
            color_path = out_dir / f"{page['id']}_color.png"
            llm_path = out_dir / f"{page['id']}_llm.jpg"
            cv2.imwrite(str(color_path), color)
            cv2.imwrite(str(llm_path),
                        llm_copy(color, cfg["preprocess"]["llm_long_edge"]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            page["color"] = f"work/pages/{page['id']}_color.png"
            page["llm_image"] = f"work/pages/{page['id']}_llm.jpg"
            page["isolated"] = True
            return
    page.pop("isolated", None)

    quad = None
    if cfg["preprocess"].get("mask_clutter", False):  # experimental: needs even lighting
        # isolate the book (bright + colorless region, spine fold inside),
        # hide the desk, and take the quad from the book itself — the warp
        # then rectifies to the book's true aspect ratio
        book_mask, book_quad, _spine = isolate_book(bgr)
        if book_quad is not None:
            bgr = mask_outside(bgr, book_mask)
            quad = book_quad
    if quad is None and fid is not None and scores and fid in scores:
        quad = scores[fid].get("quad")
        if quad is not None and rotation == 180:
            quad = order_quad(1.0 - np.array(quad, dtype=np.float64))
    if quad is None:  # patched photos have no score record: detect now
        q, _ = detect_page_quad(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        quad = q if q is not None else None

    if quad is not None:
        quad = _pad_quad(np.array(quad, dtype=np.float64), pad)
        color = correct_page(bgr, quad)
    else:
        color = bgr
    if cfg["preprocess"].get("dewarp"):
        color = dewarp_cylindrical(color)

    color_path = out_dir / f"{page['id']}_color.png"
    llm_path = out_dir / f"{page['id']}_llm.jpg"
    cv2.imwrite(str(color_path), color)
    cv2.imwrite(str(llm_path), llm_copy(color, cfg["preprocess"]["llm_long_edge"]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    page["color"] = f"work/pages/{page['id']}_color.png"
    page["llm_image"] = f"work/pages/{page['id']}_llm.jpg"


def _orientation_sample(ws: Workspace, cfg: dict, video: dict,
                        scores: dict[str, dict]):
    """Un-rotated corrected crop of a middle page from this video, for the
    'is this upside down?' check."""
    vid = video["id"]
    pages = [p for p in ws.manifest["pages"]
             if (p.get("canonical") or "").startswith(vid + "_")]
    if not pages:
        return None
    fid = pages[len(pages) // 2]["canonical"]
    bgr = cv2.imread(str(frame_path(ws, fid)))
    if bgr is None:
        return None
    quad = (scores.get(fid) or {}).get("quad")
    if quad is not None:
        bgr = correct_page(bgr, _pad_quad(np.array(quad, dtype=np.float64), 0.02))
    out = ws.work_file(f"_orient_{vid}.jpg")
    cv2.imwrite(str(out), llm_copy(bgr, 1200), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def auto_detect_orientation(ws: Workspace, cfg: dict,
                            scores: dict[str, dict], log=print) -> None:
    """Ask the vision model once per video whether its text is upside down.
    Only runs for videos whose orientation was never determined; the GUI
    toggle / --upside-down flag always wins because they set the field."""
    pending = [v for v in ws.manifest["videos"] if "rotate" not in v]
    if not pending:
        return
    provider = cfg["provider"]["name"]
    check_cfg = cfg
    if provider == "hybrid":  # one cheap local call is plenty
        check_cfg = {**cfg, "provider": {**cfg["provider"], "name": "ollama"}}
    from ..backends import get_backend
    from ..project import set_video_rotation
    backend = get_backend(check_cfg)
    for video in pending:
        sample = _orientation_sample(ws, cfg, video, scores)
        verdict = backend.check_orientation(sample) if sample else None
        if verdict is None:
            video["rotate"] = 0  # can't tell (or mock) — assume normal
            log(f"  {video['id']}: orientation check unavailable, assuming normal")
        else:
            set_video_rotation(ws, video["id"], 180 if verdict else 0,
                               log=lambda m: None)
            log(f"  {video['id']}: auto-detected "
                f"{'UPSIDE DOWN — will rotate' if verdict else 'normal orientation'}")
    ws.save()


def run(ws: Workspace, cfg: dict, log=print) -> None:
    scores = scores_by_frame_id(ws)
    auto_detect_orientation(ws, cfg, scores, log)
    pages = ws.manifest["pages"]
    for i, page in enumerate(pages):
        preprocess_page(ws, page, cfg, scores)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(pages)}")
    ws.save()
    ws.stage_done("preprocess")
    log(f"  {len(pages)} pages corrected -> work/pages/")
