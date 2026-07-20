"""Shared low-level image analysis: perceptual hash, page-quad detection, skin mask."""

from __future__ import annotations

import cv2
import numpy as np


def phash64(gray: np.ndarray) -> int:
    """64-bit perceptual hash (DCT low-frequency signs vs median)."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8].flatten()
    ac = np.delete(dct, 0)  # drop DC term
    bits = dct.flatten() > np.median(ac)
    h = 0
    for b in bits[:64]:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def majority_hash(hashes: list[int]) -> int:
    """Bitwise majority vote across 64-bit hashes."""
    if len(hashes) == 1:
        return hashes[0]
    counts = [0] * 64
    for h in hashes:
        for i in range(64):
            counts[i] += (h >> i) & 1
    half = len(hashes) / 2
    out = 0
    for i in range(64):
        if counts[i] > half:
            out |= 1 << i
    return out


def sharpness(gray: np.ndarray, center_crop: float = 0.6) -> float:
    """Variance of Laplacian on a center crop."""
    h, w = gray.shape
    ch, cw = int(h * center_crop), int(w * center_crop)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = gray[y0:y0 + ch, x0:x0 + cw]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def detect_page_quad(gray: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Find the page as the largest bright contour.

    Returns (quad, flatness) where quad is a 4x2 float array of corner points
    (normalized 0..1 coords) or None, and flatness in [0, 1] scores how
    rectangular, large, and centered the page region is.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    frame_area = float(h * w)
    if area < 0.05 * frame_area:
        return None, 0.0

    rect = cv2.minAreaRect(cnt)
    rect_area = rect[1][0] * rect[1][1]
    rectangularity = area / rect_area if rect_area > 0 else 0.0

    size_score = min(area / (0.5 * frame_area), 1.0)

    m = cv2.moments(cnt)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    center_offset = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)  # 0 centered, ~0.7 corner
    center_score = max(0.0, 1.0 - 2.0 * center_offset)

    flatness = rectangularity * size_score * center_score

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cv2.convexHull(cnt), 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float64)
    else:
        quad = cv2.boxPoints(rect).astype(np.float64)
    quad[:, 0] /= w
    quad[:, 1] /= h
    return order_quad(quad), float(np.clip(flatness, 0.0, 1.0))


def isolate_book(bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """Find the open book and separate it from desk clutter.

    Book pages are bright AND colorless; sticky notes, drawings, and desk are
    either colored (high saturation) or dark. Returns (mask, quad, spine_x):
    a uint8 mask of the book region, its normalized corner quad, and the x
    position (pixels) of the dark spine fold — or Nones when not found.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    _, v_th = cv2.threshold(cv2.GaussianBlur(val, (5, 5), 0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((v_th > 0) & (sat < 70)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, None, None
    # the book is big AND centered — desk papers/flyers are big but off-center
    best, best_score = None, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.06 * h * w:
            continue
        cx, cy = centroids[i]
        offset = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)  # 0 center, ~0.7 corner
        score = area * max(0.05, 1.0 - 2.0 * offset)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None, None, None
    book = (labels == best).astype(np.uint8) * 255
    # fill holes (dark photos/figures inside the page must stay part of the book)
    contours, _ = cv2.findContours(book, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    book = np.zeros_like(book)
    cv2.drawContours(book, [cnt], -1, 255, -1)

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cv2.convexHull(cnt), 0.02 * peri, True)
    quad = (approx.reshape(-1, 2).astype(np.float64) if len(approx) == 4
            else cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64))
    quad[:, 0] /= w
    quad[:, 1] /= h

    # the spine fold: darkest smoothed column valley inside the book's middle
    x0, x1 = int(quad[:, 0].min() * w), int(quad[:, 0].max() * w)
    lo = x0 + int((x1 - x0) * 0.30)
    hi = x0 + int((x1 - x0) * 0.70)
    spine_x = None
    if hi - lo > 30:
        vals = np.where(book > 0, val, 255).astype(np.float32)
        col = vals[:, lo:hi].mean(axis=0)
        sm = np.convolve(col, np.ones(15) / 15, mode="same")
        base = float(np.median(val[book > 0]))
        valley = float(sm.min())
        if (base - valley) / max(base, 1.0) > 0.08:
            spine_x = lo + int(np.argmin(sm))
    return book, order_quad(quad), spine_x


def find_flat_page(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the flat, readable page in an open-book photo via edge density.

    Printed text is a dense edge field regardless of lighting (which defeats
    color/brightness segmentation). Dilated Canny edges form text blocks; the
    seed block is big, rectangular, and central; blocks stacked in the same
    column (text above/below figures) merge, guarded so the result stays
    page-shaped. Returns a pixel bbox, or None when no confident page exists
    (blank pages, mid-turn chaos) — callers then fall back to the page quad.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    med = float(np.median(gray))
    edges = cv2.Canny(gray, 0.66 * med, 1.33 * med)
    text = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)))
    text = cv2.morphologyEx(text, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (13, 25)))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(text)
    cands = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.015 * h * w:
            continue
        comp = (labels == i).astype(np.uint8)
        cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        rw, rh = cv2.minAreaRect(cnt)[1]
        if min(rw, rh) < 1:
            continue
        rectangularity = cv2.contourArea(cnt) / (rw * rh)
        cx, cy = cents[i]
        offset = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        cands.append({"area": area, "rect": rectangularity,
                      "cent": max(0.05, 1 - 2 * offset),
                      "bbox": stats[i, :4].tolist()})
    if not cands:
        return None

    seed = max(cands, key=lambda c: c["area"] * c["rect"] ** 2 * c["cent"])
    sx, sy, sw, sh = seed["bbox"]
    x0, y0, x1, y1 = sx, sy, sx + sw, sy + sh
    others = sorted((c for c in cands if c is not seed),
                    key=lambda c: abs((c["bbox"][1] + c["bbox"][3] / 2) - (sy + sh / 2)))
    for c in others:
        cx0, cy0, cw, ch = c["bbox"]
        cx1, cy1 = cx0 + cw, cy0 + ch
        overlap = max(0, min(x1, cx1) - max(x0, cx0))
        nx0, ny0 = min(x0, cx0), min(y0, cy0)
        nx1, ny1 = max(x1, cx1), max(y1, cy1)
        if (overlap > 0.6 * min(x1 - x0, cw)          # same column of the page
                and (nx1 - nx0) < 0.85 * (ny1 - ny0)  # still page-shaped
                and (nx1 - nx0) < 0.75 * w):          # never the whole spread
            x0, y0, x1, y1 = nx0, ny0, nx1, ny1

    mx = int((x1 - x0) * 0.10)
    my_top = int((y1 - y0) * 0.07)
    my_bot = int((y1 - y0) * 0.13)  # printed page numbers sit below the text block
    box = (max(0, x0 - mx), max(0, y0 - my_top), min(w, x1 + mx), min(h, y1 + my_bot))
    bw, bh = box[2] - box[0], box[3] - box[1]
    bcx, bcy = box[0] + bw / 2, box[1] + bh / 2
    ok = (0.40 < bw / bh < 0.95
          and 0.10 < (bw * bh) / (w * h) < 0.70
          and abs(bcx - w / 2) / w < 0.18      # a page is central; corner hits are
          and abs(bcy - h / 2) / h < 0.22)     # clutter (flyers next to blank pages)
    return box if ok else None


def refine_figure_bbox(color: np.ndarray,
                       prior: list[float]) -> list[float] | None:
    """Snap an approximate figure bbox to the actual photo on the page.

    Photo pixels are edge-dense OR darker than the paper (the darkness cue
    matters: smooth sky areas have no edges and would split the photo).
    Every substantial component lying mostly inside the slightly-expanded
    prior is union-merged into the refined box. None = not confident;
    caller keeps the existing crop.
    """
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    med = float(np.median(gray))
    edges = cv2.Canny(gray, 0.5 * med, 1.2 * med)
    dark = (gray < med - 25).astype(np.uint8) * 255
    mask = cv2.bitwise_or(cv2.dilate(edges, np.ones((3, 3), np.uint8)), dark)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    ex = 0.06
    px0, py0 = max(0.0, prior[0] - ex) * w, max(0.0, prior[1] - ex) * h
    px1, py1 = min(1.0, prior[2] + ex) * w, min(1.0, prior[3] + ex) * h
    ux0 = uy0 = float("inf")
    ux1 = uy1 = -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.004 * h * w:
            continue
        ox = max(0, min(px1, x + bw) - max(px0, x))
        oy = max(0, min(py1, y + bh) - max(py0, y))
        if ox * oy < 0.5 * bw * bh:
            continue
        ux0, uy0 = min(ux0, x), min(uy0, y)
        ux1, uy1 = max(ux1, x + bw), max(uy1, y + bh)
    if ux1 < 0:
        return None
    m = 4
    box = [max(0, ux0 - m) / w, max(0, uy0 - m) / h,
           min(w, ux1 + m) / w, min(h, uy1 + m) / h]
    prior_area = max(1e-6, (prior[2] - prior[0]) * (prior[3] - prior[1]))
    if (box[2] - box[0]) * (box[3] - box[1]) < 0.15 * prior_area:
        return None
    return box


def mask_outside(bgr: np.ndarray, mask: np.ndarray,
                 fill: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Paint everything outside the mask a flat color (hide desk clutter)."""
    out = bgr.copy()
    out[mask == 0] = fill
    return out


def order_quad(quad: np.ndarray) -> np.ndarray:
    """Order corners tl, tr, br, bl."""
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).flatten()
    return np.array([
        quad[np.argmin(s)], quad[np.argmin(d)],
        quad[np.argmax(s)], quad[np.argmax(d)],
    ])


def skin_fraction(bgr: np.ndarray, quad: np.ndarray | None = None) -> float:
    """Fraction of (page region of) the frame that looks like skin (thumb occlusion)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    h, w = mask.shape
    if quad is not None:
        region = np.zeros((h, w), np.uint8)
        pts = (quad * [w, h]).astype(np.int32)
        cv2.fillConvexPoly(region, pts, 255)
        inside = mask[region > 0]
        return float(np.count_nonzero(inside)) / max(1, inside.size)
    return float(np.count_nonzero(mask)) / (h * w)


def quad_crop(gray_or_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Axis-aligned bbox crop of the quad region (cheap; used for hashing)."""
    h, w = gray_or_bgr.shape[:2]
    xs, ys = quad[:, 0] * w, quad[:, 1] * h
    x0, x1 = int(max(0, xs.min())), int(min(w, xs.max()))
    y0, y1 = int(max(0, ys.min())), int(min(h, ys.max()))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return gray_or_bgr
    return gray_or_bgr[y0:y1, x0:x1]
