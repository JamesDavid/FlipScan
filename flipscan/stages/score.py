"""Stage 3: score — per-frame quality metrics + perceptual hash.

Writes work/scores_<vid>.json: one record per frame with raw metrics.
Composite scoring happens at select time (needs per-video normalization).
"""

from __future__ import annotations

import json

import cv2

from ..imaging import detect_page_quad, phash64, quad_crop, sharpness, skin_fraction
from ..workspace import Workspace


def load_scores(ws: Workspace, video_id: str) -> list[dict]:
    with open(ws.work_file(f"scores_{video_id}.json"), encoding="utf-8") as f:
        return json.load(f)


def scores_by_frame_id(ws: Workspace) -> dict[str, dict]:
    out = {}
    for video in ws.manifest["videos"]:
        for rec in load_scores(ws, video["id"]):
            out[f"{video['id']}_{rec['frame']}"] = rec
    return out


def run(ws: Workspace, cfg: dict, log=print) -> None:
    center_crop = cfg["score"]["center_crop"]
    for video in ws.manifest["videos"]:
        vid = video["id"]
        out_path = ws.work_file(f"scores_{vid}.json")
        frames = sorted(ws.frames_dir(vid).glob("f*.jpg"))
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                if len(json.load(f)) == len(frames):
                    log(f"  {vid}: scores exist, skipping")
                    continue
        log(f"  {vid}: scoring {len(frames)} frames")
        records = []
        prev_small = None
        for i, fp in enumerate(frames):
            # half-res decode is plenty for scoring and much faster
            bgr = cv2.imread(str(fp), cv2.IMREAD_REDUCED_COLOR_2)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            small = cv2.resize(gray, (192, 108), interpolation=cv2.INTER_AREA)
            if prev_small is None:
                motion = 0.0
            else:
                motion = float(cv2.absdiff(small, prev_small).mean())
            prev_small = small

            quad, flatness = detect_page_quad(gray)
            page_gray = quad_crop(gray, quad) if quad is not None else gray

            records.append({
                "frame": fp.stem,
                "sharpness": round(sharpness(gray, center_crop), 2),
                "flatness": round(flatness, 4),
                "occlusion": round(skin_fraction(bgr, quad), 4),
                "motion": round(motion, 3),
                "phash": f"{phash64(page_gray):016x}",
                "quad": quad.round(4).tolist() if quad is not None else None,
            })
            if (i + 1) % 500 == 0:
                log(f"  {vid}: {i + 1}/{len(frames)}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f)
        log(f"  {vid}: done")
    ws.stage_done("score")
