"""Target-device image profiles.

Small e-ink readers (Xteink X3/X4 are ESP32-class, 480x800 grayscale panels)
choke on large color images — figures and covers get downscaled to the panel,
converted to grayscale, and re-encoded as compact JPEGs so the EPUB/PDF
actually renders on-device.
"""

from __future__ import annotations

import cv2
import numpy as np

PROFILES: dict[str, dict] = {
    "xteink-x3": {"max_w": 480, "max_h": 800, "grayscale": True, "jpeg_q": 85},
    "xteink-x4": {"max_w": 480, "max_h": 800, "grayscale": True, "jpeg_q": 85},
    "eink-6in": {"max_w": 758, "max_h": 1024, "grayscale": True, "jpeg_q": 85},
    # reMarkable 2: 1404x1872 grayscale panel (canvas geometry from reCompose,
    # MIT-licensed, github.com/mrodger/reCompose)
    "remarkable-2": {"max_w": 1404, "max_h": 1872, "grayscale": True, "jpeg_q": 85},
    "tablet": {"max_w": 1200, "max_h": 1600, "grayscale": False, "jpeg_q": 88},
}

DEVICES = ["none"] + sorted(PROFILES)


def process_image(data: bytes, device: str) -> tuple[bytes, str]:
    """Re-encode image bytes for the device. Returns (bytes, media_type)."""
    profile = PROFILES.get(device)
    if profile is None:
        return data, "image/png" if data[:8].startswith(b"\x89PNG") else "image/jpeg"
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return data, "image/jpeg"
    h, w = img.shape[:2]
    scale = min(profile["max_w"] / w, profile["max_h"] / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    if profile["grayscale"]:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, profile["jpeg_q"]])
    return (buf.tobytes(), "image/jpeg") if ok else (data, "image/jpeg")
