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
