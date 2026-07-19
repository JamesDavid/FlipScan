"""Generate synthetic two-pass book-flip test videos with known ground truth.

Produces odd.mp4 (pages N-1..1 back-to-front, i.e. reverse) and even.mp4
(pages 2..N front-to-back) that exercise scoring, clustering, parity merge,
and selection. Each page: rest frames (sharp, flat, unoccluded) separated by
turn transitions (motion blur + skin-toned thumb sweep).

Usage: python tools/make_test_video.py OUT_DIR [--pages 12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

W, H = 1920, 1080          # camera frame
PW, PH = 700, 980          # page pixels
REST_FRAMES = 18
TURN_FRAMES = 10
FPS = 60

WORDS = ("the quick brown fox jumps over a lazy dog while reading pages of "
         "an old book about optics and light in the late afternoon sun").split()


def render_page(n: int, total: int) -> np.ndarray:
    """A synthetic page: header, big printed page number, text lines, and a
    page-dependent dark block so perceptual hashes are clearly distinct."""
    rng = np.random.default_rng(n)
    img = np.full((PH, PW, 3), 245, np.uint8)
    cv2.putText(img, f"CHAPTER {1 + n // 4}", (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 60, 60), 2)
    # body text
    y = 130
    for line in range(24):
        words = [WORDS[(n * 7 + line * 3 + k) % len(WORDS)] for k in range(6)]
        cv2.putText(img, " ".join(words), (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        y += 34
    # page-dependent figure block (moves with page number -> distinct pHash)
    bx = 60 + (n * 53) % (PW - 360)
    by = 200 + (n * 97) % (PH - 500)
    cv2.rectangle(img, (bx, by), (bx + 280, by + 180), (120, 120, 120), -1)
    cv2.putText(img, f"Fig {n}", (bx + 10, by + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2)
    # printed page number
    cv2.putText(img, str(n), (PW - 120, PH - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3)
    return img


def compose(page: np.ndarray, rng: np.random.Generator,
            blur: float = 0.0, shift: int = 0, thumb: float | None = None) -> np.ndarray:
    """Place the page in the camera frame with jitter, noise, optional turn effects."""
    frame = np.full((H, W, 3), 25, np.uint8)
    angle = rng.normal(0, 0.5)
    scale = 1.0 + rng.normal(0, 0.005)
    m = cv2.getRotationMatrix2D((PW / 2, PH / 2), angle, scale)
    warped = cv2.warpAffine(page, m, (PW, PH), borderValue=(25, 25, 25))
    x0 = (W - PW) // 2 + int(rng.normal(0, 3)) + shift
    y0 = (H - PH) // 2 + int(rng.normal(0, 3))
    frame[y0:y0 + PH, x0:x0 + PW] = warped
    if blur > 0:
        k = max(3, int(blur) | 1)
        frame = cv2.blur(frame, (k * 3, k))  # horizontal-ish motion blur
    if thumb is not None:  # skin-toned thumb sweeping across during the turn
        tx = int(W * thumb)
        cv2.ellipse(frame, (tx, H // 2 + 100), (90, 170), 20, 0, 360, (105, 140, 215), -1)
    noise = rng.normal(0, 3, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def write_video(path: Path, page_numbers: list[int], total: int) -> None:
    rng = np.random.default_rng(hash(path.name) & 0xFFFF)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i, n in enumerate(page_numbers):
        page = render_page(n, total)
        for _ in range(REST_FRAMES):
            vw.write(compose(page, rng))
        if i < len(page_numbers) - 1:  # turn to next page
            nxt = render_page(page_numbers[i + 1], total)
            for t in range(TURN_FRAMES):
                frac = (t + 1) / (TURN_FRAMES + 1)
                src = page if frac < 0.5 else nxt
                vw.write(compose(src, rng, blur=18, shift=int(200 * np.sin(frac * np.pi)),
                                 thumb=frac))
    vw.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--pages", type=int, default=12)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = args.pages
    odd = list(range(total - 1, 0, -2))    # e.g. 11,9,7,5,3,1 — reverse capture
    even = list(range(2, total + 1, 2))    # 2,4,...,12 — forward capture
    write_video(args.out_dir / "odd.mp4", odd, total)
    write_video(args.out_dir / "even.mp4", even, total)
    print(f"wrote {args.out_dir}/odd.mp4 (pages {odd}) and even.mp4 (pages {even})")


if __name__ == "__main__":
    main()
